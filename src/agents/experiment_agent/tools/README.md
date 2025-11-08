# Experiment Agent Tools

完整的工具集，用于支持 experiment_agent 系统的所有功能。所有工具都与 openai-agents SDK 兼容。

## 📦 工具分类

### 1. 文件操作工具 (file_tools.py)

用于文件和目录的基本操作。

| 工具名称 | 功能描述 |
|---------|---------|
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件内容 |
| `list_directory` | 列出目录内容 |
| `create_directory` | 创建目录 |
| `delete_file` | 删除文件 |
| `copy_file` | 复制文件 |
| `file_exists` | 检查文件是否存在 |
| `get_file_info` | 获取文件详细信息 |

**使用示例**：
```python
from src.agents.experiment_agent.tools import read_file, write_file

# 读取文件
result = read_file(file_path="/path/to/file.txt")
if result["success"]:
    content = result["content"]

# 写入文件
result = write_file(
    file_path="/path/to/output.txt",
    content="Hello, World!",
    create_dirs=True
)
```

### 2. 代码执行工具 (execution_tools.py)

用于执行代码、管理环境和记录日志。

| 工具名称 | 功能描述 |
|---------|---------|
| `run_python_script` | 运行Python脚本 |
| `run_shell_command` | 执行Shell命令 |
| `run_python_code` | 执行Python代码片段 |
| `install_package` | 安装Python包 |
| `check_python_syntax` | 检查Python语法 |
| `get_environment_info` | 获取环境信息 |
| `list_installed_packages` | 列出已安装的包 |
| `create_log_file` | 创建日志文件 |
| `append_to_log` | 追加到日志 |

**使用示例**：
```python
from src.agents.experiment_agent.tools import run_python_script, create_log_file

# 创建日志文件
log_result = create_log_file(
    log_dir="./logs",
    prefix="experiment"
)
log_path = log_result["log_path"]

# 运行Python脚本
result = run_python_script(
    script_path="train.py",
    args="--epochs 10 --batch_size 32",
    working_dir="/path/to/code",
    timeout=3600
)

if result["success"]:
    print(f"执行成功，耗时: {result['execution_time']}秒")
    print(f"输出: {result['stdout']}")
else:
    print(f"执行失败: {result['error']}")
```

### 3. 文档分析工具 (document_tools.py)

用于解析和分析各种文档格式。

| 工具名称 | 功能描述 |
|---------|---------|
| `parse_latex_sections` | 解析LaTeX文档sections |
| `extract_latex_equations` | 提取LaTeX公式 |
| `parse_json_file` | 解析JSON文件 |
| `extract_code_blocks` | 提取Markdown代码块 |
| `summarize_document` | 文档摘要 |
| `extract_urls` | 提取URL |
| `parse_requirements_txt` | 解析requirements.txt |
| `extract_key_terms` | 提取关键词 |

**使用示例**：
```python
from src.agents.experiment_agent.tools import (
    parse_latex_sections,
    extract_latex_equations
)

# 解析LaTeX论文
result = parse_latex_sections(latex_content)
if result["success"]:
    sections = result["sections"]
    print(f"标题: {sections['title']}")
    print(f"摘要: {sections['abstract']}")
    for sec in sections['sections']:
        print(f"章节: {sec['title']}")

# 提取数学公式
equations = extract_latex_equations(latex_content)
for eq in equations["equations"]:
    print(f"{eq['type']}: {eq['content']}")
```

### 4. 代码分析工具 (code_analysis_tools.py)

用于分析代码结构和提取代码信息。

| 工具名称 | 功能描述 |
|---------|---------|
| `analyze_python_file` | 分析Python文件结构 |
| `search_in_codebase` | 在代码库中搜索 |
| `count_lines_of_code` | 统计代码行数 |
| `extract_function_code` | 提取函数代码 |
| `list_python_files` | 列出Python文件 |
| `check_imports_available` | 检查import是否可用 |
| `get_file_dependencies` | 获取文件依赖 |

**使用示例**：
```python
from src.agents.experiment_agent.tools import (
    analyze_python_file,
    search_in_codebase
)

# 分析Python文件结构
result = analyze_python_file(file_path="model.py")
if result["success"]:
    print(f"类数量: {result['class_count']}")
    print(f"函数数量: {result['function_count']}")
    for cls in result['classes']:
        print(f"类 {cls['name']}: {len(cls['methods'])} 个方法")

# 在代码库中搜索
results = search_in_codebase(
    directory="/path/to/code",
    pattern=r"def train\(",
    file_pattern="*.py"
)
for match in results["results"]:
    print(f"{match['file']}:{match['line_number']}: {match['line_content']}")
```

## 🎯 为Agent配置工具

### 使用预定义配置

```python
from src.agents.experiment_agent.tools import get_tools_for_agent

# 获取pre_analysis agent的工具
tools = get_tools_for_agent("pre_analysis")

# 获取code_implement agent的工具（initial场景）
tools = get_tools_for_agent("code_implement")
```

### 自定义工具组合

```python
from src.agents.experiment_agent.tools import (
    FILE_TOOLS,
    EXECUTION_TOOLS,
    CODE_ANALYSIS_TOOLS
)

# 组合需要的工具
my_tools = FILE_TOOLS + EXECUTION_TOOLS[:5] + CODE_ANALYSIS_TOOLS[:3]
```

## 📋 Agent工具推荐

| Agent | 推荐工具类别 |
|-------|-------------|
| **pre_analysis** | DOCUMENT_TOOLS + FILE_TOOLS (read, write, list) |
| **code_plan** | FILE_TOOLS + CODE_ANALYSIS_TOOLS |
| **code_implement** | FILE_TOOLS + EXECUTION_TOOLS + CODE_ANALYSIS_TOOLS |
| **code_judge** | FILE_TOOLS + CODE_ANALYSIS_TOOLS |
| **experiment_execute** | FILE_TOOLS (read, write, list) + EXECUTION_TOOLS |
| **experiment_analysis** | FILE_TOOLS + DOCUMENT_TOOLS + CODE_ANALYSIS_TOOLS |

## 🔧 在Master Agent中使用

```python
from src.agents.experiment_agent.agents.experiment_master import (
    create_experiment_master_agent
)
from src.agents.experiment_agent.tools import (
    get_tools_for_agent,
    FILE_TOOLS,
    EXECUTION_TOOLS
)

# 配置工具
tools = {
    "pre_analysis": {
        "paper": get_tools_for_agent("pre_analysis")["paper"],
        "idea": get_tools_for_agent("pre_analysis")["idea"],
    },
    "code_plan": get_tools_for_agent("code_plan"),
    "code_implement": get_tools_for_agent("code_implement"),
    "code_judge": get_tools_for_agent("code_judge"),
    "experiment_execute": get_tools_for_agent("experiment_execute"),
    "experiment_analysis": get_tools_for_agent("experiment_analysis"),
}

# 创建master agent
master_agent = create_experiment_master_agent(
    model="gpt-4o",
    tools=tools,
    working_dir="/workspace",
    log_dir="./logs"
)
```

## 🛠️ 工具开发指南

### 创建新工具

所有工具必须使用 `@function_tool` 装饰器（来自 `agents` 库）：

```python
from agents import function_tool
from typing import Dict, Any

@function_tool
def my_new_tool(arg1: str, arg2: int = 10) -> Dict[str, Any]:
    """
    工具的简短描述（会被LLM看到）。

    Args:
        arg1: 参数1的描述
        arg2: 参数2的描述

    Returns:
        包含结果的字典
    """
    try:
        # 工具逻辑
        result = do_something(arg1, arg2)
        
        return {
            "success": True,
            "result": result,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
```

### 工具设计原则

1. **返回格式一致**: 始终返回包含 `success` 字段的字典
2. **错误处理**: 捕获所有异常并返回友好的错误消息
3. **类型注解**: 使用完整的类型注解
4. **文档字符串**: 提供清晰的文档（LLM会读取）
5. **参数验证**: 验证输入参数
6. **编码**: 文件操作使用 UTF-8 编码

### 添加新工具到系统

1. 在相应的工具文件中实现工具函数
2. 在 `__init__.py` 中导入并添加到相应的工具列表
3. 在 `get_tools_for_agent()` 中配置推荐使用的agent
4. 更新此README文档

## 📊 工具统计

- **总工具数**: 32
- **文件工具**: 8
- **执行工具**: 9
- **文档工具**: 8
- **代码分析工具**: 7

## 🔍 工具测试

创建测试脚本验证工具功能：

```python
from src.agents.experiment_agent.tools import (
    read_file,
    run_python_code,
    analyze_python_file
)

# 测试文件读取
result = read_file(__file__)
assert result["success"]
print(f"✓ 文件读取: {result['line_count']} 行")

# 测试代码执行
result = run_python_code("print('Hello')")
assert result["success"]
print(f"✓ 代码执行: {result['stdout']}")

# 测试代码分析
result = analyze_python_file(__file__)
assert result["success"]
print(f"✓ 代码分析: {result['function_count']} 个函数")

print("\n所有工具测试通过!")
```

## 📚 参考资料

- [OpenAI Agents SDK](https://github.com/openai/openai-agents)
- [function_tool 装饰器文档](https://github.com/openai/openai-agents/blob/main/docs/tools.md)

---

**最后更新**: 2025-11-05

**状态**: ✅ 所有工具实现完成并经过测试

