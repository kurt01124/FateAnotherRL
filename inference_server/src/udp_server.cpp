#include "udp_server.h"

#include <cstring>
#include <stdexcept>
#include <iostream>
#include <sstream>

// ============================================================
// Platform init/cleanup
// ============================================================

void UdpServer::init_platform() {
#ifdef _WIN32
    WSADATA wsa;
    int err = WSAStartup(MAKEWORD(2, 2), &wsa);
    if (err != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(err));
    }
#endif
}

void UdpServer::cleanup_platform() {
#ifdef _WIN32
    WSACleanup();
#endif
}

// ============================================================
// Constructor / Destructor
// ============================================================

UdpServer::UdpServer(int listen_port, int send_port)
    : sock_(INVALID_SOCK), send_port_(send_port)
{
    init_platform();

    // Create UDP socket
    sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock_ == INVALID_SOCK) {
        cleanup_platform();
        throw std::runtime_error("Failed to create UDP socket");
    }

    // Disable WSAECONNRESET on UDP (Windows-specific: ICMP port unreachable)
#ifdef _WIN32
    {
        BOOL bNewBehavior = FALSE;
        DWORD dwBytesReturned = 0;
        WSAIoctl(sock_, SIO_UDP_CONNRESET, &bNewBehavior, sizeof(bNewBehavior),
                 NULL, 0, &dwBytesReturned, NULL, NULL);
    }
#endif

    // Set non-blocking
#ifdef _WIN32
    u_long mode = 1;
    if (ioctlsocket(sock_, FIONBIO, &mode) != 0) {
        closesocket(sock_);
        cleanup_platform();
        throw std::runtime_error("Failed to set non-blocking mode (ioctlsocket)");
    }
#else
    int flags = fcntl(sock_, F_GETFL, 0);
    if (flags < 0 || fcntl(sock_, F_SETFL, flags | O_NONBLOCK) < 0) {
        close(sock_);
        throw std::runtime_error("Failed to set non-blocking mode (fcntl)");
    }
#endif

    // Increase receive buffer size for burst packets (16 MB to prevent drops)
    int rcvbuf = 16 * 1024 * 1024;  // 16 MB
    setsockopt(sock_, SOL_SOCKET, SO_RCVBUF,
               reinterpret_cast<const char*>(&rcvbuf), sizeof(rcvbuf));

    // Allow address reuse
    int reuse = 1;
    setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    // Bind
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(listen_port));

    if (bind(sock_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
#ifdef _WIN32
        int err = WSAGetLastError();
        closesocket(sock_);
        cleanup_platform();
        throw std::runtime_error("Failed to bind UDP socket on port "
                                 + std::to_string(listen_port)
                                 + " (error " + std::to_string(err) + ")");
#else
        close(sock_);
        throw std::runtime_error("Failed to bind UDP socket on port "
                                 + std::to_string(listen_port));
#endif
    }

    std::cout << "[UdpServer] Listening on port " << listen_port
              << ", reply port " << send_port << std::endl;
}

UdpServer::~UdpServer() {
    if (sock_ != INVALID_SOCK) {
#ifdef _WIN32
        closesocket(sock_);
#else
        close(sock_);
#endif
    }
    cleanup_platform();
}

// ============================================================
// recv_all: drain all pending packets (non-blocking)
// ============================================================

std::vector<std::pair<std::string, std::vector<uint8_t>>> UdpServer::recv_all() {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> results;

    uint8_t buf[MAX_UDP_PACKET];
    struct sockaddr_in from_addr;
    socklen_t from_len = sizeof(from_addr);

    while (true) {
        std::memset(&from_addr, 0, sizeof(from_addr));
        from_len = sizeof(from_addr);

#ifdef _WIN32
        int n = recvfrom(sock_, reinterpret_cast<char*>(buf), MAX_UDP_PACKET, 0,
                         reinterpret_cast<struct sockaddr*>(&from_addr), &from_len);
        if (n == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK) break;  // No more data
            if (err == WSAECONNRESET) continue; // ICMP port unreachable (client died), skip
            std::cerr << "[UdpServer] recvfrom error: " << err << std::endl;
            break;
        }
#else
        ssize_t n = recvfrom(sock_, buf, MAX_UDP_PACKET, 0,
                             reinterpret_cast<struct sockaddr*>(&from_addr), &from_len);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            std::cerr << "[UdpServer] recvfrom error: " << errno << std::endl;
            break;
        }
#endif

        if (n <= 0) break;

        // Format source address as "ip:port"
        char ip_buf[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &from_addr.sin_addr, ip_buf, sizeof(ip_buf));
        int src_port = ntohs(from_addr.sin_port);

        std::string addr_str = std::string(ip_buf) + ":" + std::to_string(src_port);

        results.emplace_back(
            std::move(addr_str),
            std::vector<uint8_t>(buf, buf + n)
        );
    }

    return results;
}

// ============================================================
// send_to: send to "ip:port" address string
// ============================================================

void UdpServer::send_to(const std::string& addr_str, const uint8_t* data, size_t len) {
    // Parse "ip:port" or just "ip" (uses send_port_)
    std::string ip;
    int port = send_port_;

    auto colon_pos = addr_str.rfind(':');
    if (colon_pos != std::string::npos) {
        ip = addr_str.substr(0, colon_pos);
        // The part after ':' might be a port, try to parse it
        std::string port_str = addr_str.substr(colon_pos + 1);
        // Use send_port_ always for reply (C# plugin listens on a fixed port)
        (void)port_str;
    } else {
        ip = addr_str;
    }

    send_to_ip(ip, data, len);
}

// ============================================================
// send_to_ip: send to IP using send_port_
// ============================================================

void UdpServer::send_to_ip(const std::string& ip, const uint8_t* data, size_t len) {
    struct sockaddr_in dest;
    std::memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_port = htons(static_cast<uint16_t>(send_port_));

    if (inet_pton(AF_INET, ip.c_str(), &dest.sin_addr) != 1) {
        std::cerr << "[UdpServer] Invalid IP address: " << ip << std::endl;
        return;
    }

#ifdef _WIN32
    int sent = sendto(sock_, reinterpret_cast<const char*>(data),
                      static_cast<int>(len), 0,
                      reinterpret_cast<struct sockaddr*>(&dest), sizeof(dest));
    if (sent == SOCKET_ERROR) {
        std::cerr << "[UdpServer] sendto error: " << WSAGetLastError() << std::endl;
    }
#else
    ssize_t sent = sendto(sock_, data, len, 0,
                          reinterpret_cast<struct sockaddr*>(&dest), sizeof(dest));
    if (sent < 0) {
        std::cerr << "[UdpServer] sendto error: " << errno << std::endl;
    }
#endif
}
