#!/bin/bash
# FateAnother RL - WC3 Container Entrypoint (Episode Restart Loop)
# tini + xvfb-run이 Xvfb를 자동 관리 (Dockerfile ENTRYPOINT에서 설정)
#
# v3: UDP 통신 (INFERENCE_HOST 환경변수로 GPU 컨테이너 주소 설정)
#     xdotool auto-dismiss 로딩 화면

WC3_DIR="/opt/wc3"
MAP_PATH="Z:\\opt\\wc3\\Maps\\rl\\fateanother_rl.w3x"
SPEED="${WC3_SPEED_MULTIPLIER:-}"

# Wine/Mesa 환경
export WINEDEBUG=${WINEDEBUG:--all}
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# C# RLCommPlugin UDP 환경변수 (컨테이너간 통신)
export INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
export INFERENCE_PORT="${INFERENCE_PORT:-7777}"
export RL_RECV_PORT="${RL_RECV_PORT:-7778}"

echo "=== FateAnother RL WC3 Container ==="
echo "  Map: ${MAP_PATH}"
echo "  Speed: ${SPEED:-1x (default)}"
echo "  Inference: ${INFERENCE_HOST}:${INFERENCE_PORT} (UDP)"
echo "  Recv port: ${RL_RECV_PORT} (UDP)"
echo "  DISPLAY: ${DISPLAY}"
echo ""

EPISODE=0

while true; do
    EPISODE=$((EPISODE + 1))
    echo ""
    echo "============================================"
    echo "=== Episode ${EPISODE} starting at $(date) ==="
    echo "============================================"

    # 이전 에피소드의 sentinel 파일 제거
    rm -f /tmp/rl_episode_done

    cd "${WC3_DIR}"

    # JNLoader로 war3.exe 시작 + DLL injection
    WC3_SPEED_MULTIPLIER="${SPEED}" \
      wine JNLoader.exe -loadfile "${MAP_PATH}" -window &
    WINE_PID=$!
    echo "  Wine/JNLoader PID: ${WINE_PID}"

    # JNLoader 종료 대기 (최대 30초)
    for i in $(seq 1 30); do
        if ! kill -0 $WINE_PID 2>/dev/null; then
            echo "  JNLoader finished (${i}s)"
            break
        fi
        sleep 1
    done

    # Auto-dismiss 로딩 화면 (xdotool key space 반복)
    (
        sleep 10  # 맵 로드 대기
        for i in $(seq 1 90); do
            xdotool key space 2>/dev/null || true
            sleep 1
        done
    ) &
    DISMISS_PID=$!

    # wineserver가 살아있는지 확인
    sleep 2
    if ! pgrep -x wineserver64 > /dev/null 2>&1 && ! pgrep -x wineserver > /dev/null 2>&1; then
        echo "WARNING: No wineserver running - WC3 failed to start, retrying..."
        kill "${DISMISS_PID}" 2>/dev/null || true
        sleep 3
        continue
    fi
    echo "  wineserver is running - WC3 is alive"

    # 타임아웃 없음 — 게임 자연 종료까지 대기 (sentinel / wineserver 종료 / war3 크래시만)
    WAR3_DEAD_COUNT=0
    WAR3_DEAD_THRESHOLD=30  # 30 * 0.5s = 15초 (크래시 감지)
    while true; do
        if [ -f /tmp/rl_episode_done ]; then
            echo "  Episode done signal detected, killing wineserver..."
            rm -f /tmp/rl_episode_done
            wineserver -k 2>/dev/null || true
            sleep 1
            break
        fi
        if ! pgrep -x wineserver64 > /dev/null 2>&1 && ! pgrep -x wineserver > /dev/null 2>&1; then
            echo "  wineserver already exited"
            break
        fi
        if ! pgrep -f "war3.exe" > /dev/null 2>&1; then
            WAR3_DEAD_COUNT=$((WAR3_DEAD_COUNT + 1))
            if [ $WAR3_DEAD_COUNT -ge $WAR3_DEAD_THRESHOLD ]; then
                echo "  WARNING: war3.exe crashed, killing wineserver..."
                wineserver -k 2>/dev/null || true
                sleep 1
                break
            fi
        else
            WAR3_DEAD_COUNT=0
        fi
        sleep 0.5
    done

    echo "=== Episode ${EPISODE} ended at $(date) ==="

    # Cleanup
    kill "${DISMISS_PID}" 2>/dev/null || true
    pkill -9 -f war3 2>/dev/null || true
    pkill -9 -f wine 2>/dev/null || true

    # 안정화 대기
    sleep 2
done
