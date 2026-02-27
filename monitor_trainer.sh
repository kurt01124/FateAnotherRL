#!/bin/bash
# Monitor fate-trainer for crashes (2 hours, check every 60s)
SSH="sshpass -p Gagaseoro2!# ssh -o StrictHostKeyChecking=no -p 54311 lhcho@10.0.77.186"

for i in $(seq 1 120); do
    STATUS=$($SSH "docker ps --filter name=fate-trainer --format '{{.Status}}'" 2>/dev/null)

    if [ -z "$STATUS" ]; then
        EXIT_STATUS=$($SSH "docker ps -a --filter name=fate-trainer --format '{{.Status}}'" 2>/dev/null)
        if echo "$EXIT_STATUS" | grep -qi "exited"; then
            LOGS=$($SSH "docker logs fate-trainer --tail 50 2>&1" 2>/dev/null)
            echo "CRASH at $(date): $EXIT_STATUS"
            echo "$LOGS"
            exit 1
        fi
        echo "SSH_FAIL at $(date), retrying..."
    else
        LOGS=$($SSH "docker logs fate-trainer --tail 3 2>&1" 2>/dev/null)
        if echo "$LOGS" | grep -qi "OutOfMemoryError\|CUDA out of memory\|Traceback"; then
            FULL=$($SSH "docker logs fate-trainer --tail 50 2>&1" 2>/dev/null)
            echo "ERROR at $(date)"
            echo "$FULL"
            exit 1
        fi
        if [ $((i % 5)) -eq 0 ]; then
            echo "CHECK $i - $(date): OK ($STATUS)"
            LAST_LINE=$(echo "$LOGS" | tail -1)
            echo "  $LAST_LINE"
        fi
    fi
    sleep 60
done
echo "ALL_CLEAR after 2 hours"
exit 0
