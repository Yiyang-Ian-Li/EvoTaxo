# TaxoAdapt for Naloxone Reddit Data

## 修改说明

### 1. 数据输入
- **输入源**: CSV文件 (`naloxone_mentions.csv`)
- **必需字段**: `text`, `title` (可选), `subreddit`, `kind`
- 自动处理submissions和comments，为comments生成描述性title

### 2. API配置
- **使用模型**: `gpt-oss:120b`
- **API地址**: `https://openwebui.crc.nd.edu/api/v1/chat/completions`
- **API Key**: 从环境变量 `OPENAI_API_KEY` 读取

### 3. 统计功能
✅ **Token使用统计**:
  - 总API调用次数
  - Prompt tokens总数
  - Completion tokens总数
  - 总token数

✅ **运行时间统计**:
  - 总执行时间（秒）
  - 格式化的时间显示（HH:MM:SS）

所有统计数据保存在 `datasets/naloxone_reddit/statistics.json`

### 4. 单一分类体系
- **简化架构**: 不再使用多维度分类
- **单一taxonomy**: 构建一个统一的主题分类体系
- 自动根据数据内容迭代扩展和细化分类

## 使用方法

### 方法1: 使用脚本运行
```bash
cd /home/yli62/Documents/ClaimTaxo
bash run_naloxone_taxonomy.sh
```

### 方法2: 直接运行
```bash
cd /home/yli62/Documents/ClaimTaxo/taxoadapt-copy
export OPENAI_API_KEY='your_api_key'
python main.py --topic "naloxone discussion" \
               --dataset "naloxone_reddit" \
               --csv_path "../naloxone_mentions.csv" \
               --llm "custom" \
               --max_depth 2 \
               --init_levels 1 \
               --max_density 40
```

### 自定义参数
- `--topic`: 主题描述
- `--max_depth`: 最大分类层级深度
- `--init_levels`: 初始生成的层级数
- `--max_density`: 触发扩展的最大文档数阈值

## 输出文件

所有结果保存在 `datasets/naloxone_reddit/`:

1. **internal.txt**: 预处理后的输入数据
2. **initial_taxo.txt**: 初始分类体系（文本）
3. **final_taxonomy.txt**: 最终分类体系（文本）
4. **final_taxonomy.json**: 最终分类体系（JSON格式）
5. **statistics.json**: 完整统计信息
   ```json
   {
     "total_prompt_tokens": 12345,
     "total_completion_tokens": 6789,
     "total_tokens": 19134,
     "api_calls": 42,
     "execution_time_seconds": 123.45,
     "execution_time_formatted": "0:02:03"
   }
   ```

## 修改的文件

### model_definitions.py
- 添加 `TokenUsageTracker` 类追踪token使用
- 添加 `promptCustomAPI()` 函数支持自定义API
- 修改 `initializeLLM()` 支持 `llm='custom'` 模式
- 修改 `promptGPT()` 添加token追踪
- 修改 `promptLLM()` 路由到custom API

### main.py
- **移除多维度支持**：简化为单一taxonomy构建
- 修改 `construct_dataset()` 支持读取CSV
- 添加pandas和time导入
- **添加运行时间统计**：完整的开始到结束计时
- 修改默认参数为naloxone项目
- 简化DAG初始化和分类逻辑
- 统一输出文件命名
- 合并统计信息到单个文件

## 故障排除

### 问题1: "Model not found"
确保使用正确的模型名称 `gpt-oss:120b`

### 问题2: API认证失败
检查环境变量:
```bash
echo $OPENAI_API_KEY
```

### 问题3: CSV文件路径错误
确保 `naloxone_mentions.csv` 在 `/home/yli62/Documents/ClaimTaxo/` 目录下

### 问题4: pandas未安装
```bash
pip install pandas
# 或
uv add pandas
```

## 架构简化说明

**旧版本**: 多维度分类
- 需要先将每个post分类到不同维度（topics, attitudes, use_cases等）
- 为每个维度独立构建分类树
- 输出多个taxonomy文件

**新版本**: 单一taxonomy
- 所有posts从同一个根节点开始
- 根据内容自动迭代扩展和细化
- 输出一个统一的分类体系
- 更简单、更快、更容易理解

## 性能优势

✅ 更快的运行速度（减少分类步骤）
✅ 更少的API调用
✅ 更简洁的输出
✅ 完整的时间和token统计
