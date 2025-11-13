#!/bin/bash
# TCP Server 监控和自动恢复脚本

CONTAINER_NAME="research_agent"
TARGET_PORT=18379

while true; do
    # 获取容器ID
    CONTAINER_ID=$(docker ps --filter "name=${CONTAINER_NAME}" --format "{{.ID}}")
    
    if [ -z "$CONTAINER_ID" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  容器未运行"
        sleep 10
        continue
    fi
    
    TCP_COUNT=$(docker exec $CONTAINER_ID ps aux | grep "tcp_server.*port ${TARGET_PORT}" | grep -v grep | wc -l)
    
    if [ $TCP_COUNT -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ tcp_server端口${TARGET_PORT}未运行，正在重启..."
        docker exec -d $CONTAINER_ID python3 /workspace/tcp_server.py \
            --workspace workspace \
            --conda_path /home/user/micromamba \
            --port ${TARGET_PORT}
        sleep 2
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ tcp_server已重启"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ tcp_server运行正常"
    fi
    
    sleep 30  # 每30秒检查一次
done

