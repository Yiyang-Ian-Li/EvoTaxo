# 依赖安装指南

## 当前环境状态

使用 `uv` 管理Python环境，已安装的核心依赖：
```toml
dependencies = [
    "ipykernel>=7.1.0",
    "numpy>=2.4.2",
    "openai>=2.16.0",
    "pandas>=3.0.0",
    "requests>=2.32.5",
    "tqdm>=4.67.2",
    "zstandard>=0.25.0",
]
```

## 必需安装的依赖（用于naloxone项目）

### 1. Pydantic（必需）
用于JSON schema验证和结构化输出：
```bash
uv add "pydantic>=2.0.0"
```

### 2. Typing Extensions（必需）
Pydantic的依赖：
```bash
uv add typing-extensions
```

## 可选依赖

### 选项1: 使用自定义API（推荐，当前配置）✅
**不需要额外安装任何东西！**

已有的包就够了：
- ✅ pandas
- ✅ requests  
- ✅ openai
- ✅ tqdm
- ✅ numpy

### 选项2: 使用HuggingFace数据集
如果要使用其他学术数据集（如EMNLP论文）：
```bash
uv add datasets
```

### 选项3: 使用本地GPU运行（不推荐）
需要GPU和大量额外依赖：
```bash
# 不推荐安装，因为：
# 1. 需要CUDA GPU
# 2. 安装包很大（>10GB）
# 3. 自定义API更快更简单

# 如果真的需要：
uv add torch transformers vllm
```

## 快速安装（推荐）

```bash
# 在ClaimTaxo目录下运行
cd /home/yli62/Documents/ClaimTaxo

# 只需要安装这两个
uv add pydantic typing-extensions

# 可选：如果要用其他数据集
uv add datasets
```

## 验证安装

运行这个命令检查依赖：
```bash
cd taxoadapt-copy
python -c "
from model_definitions import initializeLLM, token_tracker
from prompts import NodeListSchema
from taxonomy import Node
import pandas as pd
import requests
print('✓ All required packages are available!')
"
```

如果没有错误，就可以运行了：
```bash
python main.py --help
```

## 不需要安装的包（已处理）

以下包在代码中已改为条件导入，**不安装也不会影响运行**：

- ❌ transformers - 仅vllm模式需要
- ❌ vllm - 仅vllm模式需要  
- ❌ torch - 仅vllm模式需要
- ❌ sentence-transformers - 未使用
- ❌ scikit-learn - 仅utils.py部分功能需要
- ❌ nltk - 未使用
- ❌ tiktoken - 未使用
- ❌ pypdf - 未使用
- ❌ aiohttp - 未使用
- ❌ joblib - 未使用
- ❌ Unidecode - 未使用
- ❌ outlines - 未使用

## 故障排除

### 问题1: ImportError: pydantic
```bash
uv add pydantic
```

### 问题2: 提示 datasets 不可用
**如果只用CSV数据，忽略这个警告即可。**

如果需要其他数据集：
```bash
uv add datasets
```

### 问题3: 提示 vllm 不可用
**正常！使用 --llm custom 模式不需要vllm。**

### 问题4: OpenAI API key错误
```bash
export OPENAI_API_KEY='your_key_here'
```

## 推荐的最小安装

```bash
# 从零开始设置环境
cd /home/yli62/Documents/ClaimTaxo

# 添加必需的包
uv add pydantic typing-extensions

# 验证
cd taxoadapt-copy
python main.py --help

# 运行
python main.py
```

就这么简单！
