# Docker Execution Environment

## 概述

experiment_agent系统**强制**在Docker容器中执行所有代码，不允许在本地环境运行任何用户代码。这是出于安全考虑的设计。

所有代码执行都通过TCP连接到已运行的Docker容器中的TCP server来完成。

## 架构

```
[Experiment Agent] --TCP Socket--> [Docker Container with TCP Server]
     (本地)                              (隔离环境)
```

- **本地**: Agent、工具系统、文件操作
- **Docker**: 所有代码执行、包安装、环境查询

## 前提条件

### 1. Docker容器运行

需要一个已经运行的Docker容器，其中：
- 安装了Python环境
- 运行了TCP server（监听指定端口）
- 挂载了工作目录

### 2. TCP Server

Docker容器内需要运行`tcp_server.py`（参考AI-Researcher-v2实现）：

```bash
# 在Docker容器内运行
python /path/to/tcp_server.py --workplace /workspace --port 8000
```

TCP server功能：
- 监听TCP连接
- 接收shell命令
- 在容器内执行命令
- 流式返回输出
- 返回退出码

## 环境配置

### 环境变量

在运行experiment_agent前设置：

```bash
export DOCKER_HOST="localhost"      # Docker主机地址
export DOCKER_PORT="8000"           # TCP server端口
export DOCKER_TIMEOUT="3600"        # 命令超时时间（秒）
```

### Python代码配置

```python
from src.agents.experiment_agent.tools.execution_tools import set_docker_client
from src.agents.experiment_agent.environment import create_docker_client

# 创建Docker客户端
client = create_docker_client(
    host="localhost",
    port=8000,
    timeout=3600
)

# 设置为全局客户端
set_docker_client(client)

# 测试连接
from src.agents.experiment_agent.tools import test_docker_connection
result = test_docker_connection()
print(result)
```

## 使用示例

### 1. 运行Python脚本

```python
from src.agents.experiment_agent.tools import run_python_script

result = run_python_script(
    script_path="/workspace/train.py",
    args="--epochs 10 --batch_size 32",
    working_dir="/workspace",
    stream_output=True  # 实时打印输出
)

if result['success']:
    print("✓ Script executed successfully")
    print(result['stdout'])
else:
    print("✗ Execution failed")
    print(result['stderr'])
```

### 2. 运行Shell命令

```python
from src.agents.experiment_agent.tools import run_shell_command

result = run_shell_command(
    command="pip list | grep torch",
    working_dir="/workspace"
)

print(result['stdout'])
```

### 3. 运行Python代码片段

```python
from src.agents.experiment_agent.tools import run_python_code

code = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
"""

result = run_python_code(code, stream_output=True)
```

### 4. 安装包

```python
from src.agents.experiment_agent.tools import install_package

result = install_package(
    package_name="numpy==1.24.0",
    upgrade=True,
    stream_output=True
)

if result['installed']:
    print("✓ Package installed successfully")
```

### 5. 检查Python语法

```python
from src.agents.experiment_agent.tools import check_python_syntax

result = check_python_syntax("/workspace/script.py")

if result['valid_syntax']:
    print("✓ Syntax is valid")
else:
    print(f"✗ Syntax error: {result['syntax_error']['message']}")
```

### 6. 获取环境信息

```python
from src.agents.experiment_agent.tools import get_environment_info

info = get_environment_info()
print(f"Python version: {info['python_version']}")
print(f"Platform: {info['platform']}")
print(f"Working directory: {info['working_directory']}")
print(f"Execution mode: {info['execution_mode']}")  # "Docker Container"
```

## Docker容器设置

### 方法1: 使用现有容器

如果已有运行的容器：

```bash
# 1. 进入容器
docker exec -it <container_name> bash

# 2. 启动TCP server
python /path/to/tcp_server.py --workplace /workspace --port 8000

# 3. 保持容器运行
```

### 方法2: 创建新容器

```bash
# 创建并启动容器
docker run -d \
    --name experiment_container \
    --net=host \
    -v /path/to/workspace:/workspace \
    -w /workspace \
    python:3.10 \
    python /workspace/tcp_server.py --workplace /workspace --port 8000
```

### 方法3: 使用docker-compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  experiment_env:
    image: python:3.10
    container_name: experiment_container
    network_mode: host
    volumes:
      - ./workspace:/workspace
      - ./tcp_server.py:/app/tcp_server.py
    working_dir: /workspace
    command: python /app/tcp_server.py --workplace /workspace --port 8000
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## TCP Server实现

参考`/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/AI-Researcher-v2/docker/tcp_server.py`

关键功能：
- 接收命令字符串
- 在指定工作目录执行
- 流式发送输出（每行立即发送）
- 发送最终状态（退出码、完整输出）

响应格式：
```json
// 流式输出（每行）
{"type": "chunk", "data": "output line\n"}

// 最终响应
{"type": "final", "status": 0, "result": "complete output"}
```

## 安全性说明

### 为什么强制使用Docker？

1. **隔离性**: 用户代码在容器中运行，无法访问主机系统
2. **可控性**: 容器环境完全可控，可以限制资源
3. **可重现性**: 统一的执行环境确保结果可重现
4. **安全性**: 避免恶意代码对主机造成影响

### 本地执行已禁用

所有execution_tools中的工具都**只能**在Docker中执行：

```python
# ❌ 这不会在本地执行
result = run_python_code("import os; os.system('rm -rf /')")

# ✓ 只会在Docker容器中执行，主机系统安全
```

如果Docker不可用，工具会返回错误：
```python
{
    "success": False,
    "error": "Cannot connect to Docker container at localhost:8000. Code execution requires Docker environment.",
    "exit_code": -1
}
```

## 故障排除

### 连接失败

**问题**: `Cannot connect to Docker container`

**解决方案**:
1. 检查Docker容器是否运行: `docker ps | grep experiment`
2. 检查TCP server是否启动: `docker exec <container> ps aux | grep tcp_server`
3. 检查端口是否正确: 环境变量DOCKER_PORT
4. 检查网络模式: 使用`--net=host`或正确的端口映射

### 命令超时

**问题**: `Command execution timeout`

**解决方案**:
1. 增加超时时间: `export DOCKER_TIMEOUT="7200"`
2. 检查命令是否真的需要很长时间
3. 优化代码减少执行时间

### 输出乱码

**问题**: Unicode decode error

**解决方案**:
1. 确保容器使用UTF-8编码
2. 在Docker中设置: `ENV LANG=C.UTF-8`
3. 检查输出中是否有二进制数据

## 最佳实践

### 1. 工作目录管理

```python
# ✓ 好: 使用绝对路径
run_python_script("/workspace/train.py", working_dir="/workspace")

# ✗ 避免: 相对路径可能导致混淆
run_python_script("train.py")
```

### 2. 流式输出

对于长时间运行的任务，使用流式输出：

```python
result = run_python_script(
    "/workspace/long_running_task.py",
    stream_output=True  # 实时显示进度
)
```

### 3. 错误处理

```python
result = run_python_script("/workspace/script.py")

if not result['success']:
    if "Cannot connect to Docker" in result.get('error', ''):
        print("Docker环境未配置")
        # 提示用户配置Docker
    elif result['exit_code'] != 0:
        print("脚本执行失败")
        print(result['stderr'])
```

### 4. 资源清理

```python
# 定期清理不需要的包
run_shell_command("pip cache purge")

# 清理临时文件
run_shell_command("rm -rf /tmp/*")
```

## 与Agent集成

experiment_agent中的所有agent会自动使用这些工具：

- **experiment_execute_agent**: 运行实验代码
- **code_implement_agent**: 测试生成的代码
- **code_judge_agent**: 验证代码语法

无需手动配置，只需确保Docker环境运行。

## 监控和日志

### 检查Docker状态

```python
from src.agents.experiment_agent.tools import test_docker_connection, get_environment_info

# 测试连接
conn_result = test_docker_connection()
print(f"Connected: {conn_result['success']}")

# 获取环境信息
env_info = get_environment_info()
print(f"Python: {env_info['python_version']}")
print(f"Platform: {env_info['platform']}")
```

### 日志记录

执行工具的输出会包含在返回的字典中：

```python
result = run_python_script("train.py")

# 保存日志
with open("execution.log", "w") as f:
    f.write(f"Command: {result['command']}\n")
    f.write(f"Exit Code: {result['exit_code']}\n")
    f.write(f"Output:\n{result['stdout']}\n")
    if result['stderr']:
        f.write(f"Errors:\n{result['stderr']}\n")
```

## 参考资料

- AI-Researcher-v2 Docker实现: `/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/AI-Researcher-v2/research_agent/inno/environment/`
- TCP Server实现: `/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/AI-Researcher-v2/docker/tcp_server.py`
- Docker客户端: `src/agents/experiment_agent/environment/docker_client.py`
- 执行工具: `src/agents/experiment_agent/tools/execution_tools.py`

---

**更新日期**: 2025-11-05  
**状态**: ✅ 生产就绪

