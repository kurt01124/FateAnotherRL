#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

// Platform-specific socket types
#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <mswsock.h>
    #pragma comment(lib, "ws2_32.lib")
    using socket_t = SOCKET;
    constexpr socket_t INVALID_SOCK = INVALID_SOCKET;
    #ifndef SIO_UDP_CONNRESET
    #define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR, 12)
    #endif
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    using socket_t = int;
    constexpr socket_t INVALID_SOCK = -1;
#endif

// Maximum UDP datagram we expect (64 KB)
constexpr size_t MAX_UDP_PACKET = 65536;

class UdpServer {
public:
    /// listen_port: port to bind and receive STATE packets on.
    /// send_port:   port to send ACTION replies to (on the source address).
    UdpServer(int listen_port, int send_port);
    ~UdpServer();

    // Non-copyable
    UdpServer(const UdpServer&) = delete;
    UdpServer& operator=(const UdpServer&) = delete;

    /// Non-blocking receive of all pending packets.
    /// Returns vector of (source_addr_string, raw_data).
    /// source_addr_string is "ip:port" format (e.g. "127.0.0.1:12345").
    std::vector<std::pair<std::string, std::vector<uint8_t>>> recv_all();

    /// Send raw data to a specific address.
    /// addr_str is "ip:port" format or just "ip" (uses send_port_).
    void send_to(const std::string& addr_str, const uint8_t* data, size_t len);

    /// Send raw data to a specific IP (uses send_port_ as port).
    void send_to_ip(const std::string& ip, const uint8_t* data, size_t len);

private:
    socket_t sock_;
    int send_port_;

    void init_platform();
    void cleanup_platform();
};
