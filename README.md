# FinKB 金融知识库

面向普通用户的金融知识问答系统：用自然语言提问，系统从已导入的金融资料（上市公司季报、基金产品资料概要、理财风险揭示书、宏观政策报告、投资者教育问答等）中查找依据，流式给出带引用来源、符合合规要求的回答。

定位是资料查询与解读工具：**不提供个性化投资建议，不承诺收益**；无资料时明确告知，不编造数据。

## 架构

两张 LangGraph 图，节点逻辑写在 `processor/*/nodes/`，工具函数在 `utils/`。

```text
导入图（一次处理一个文件，进度记在 Mongo documents.stage，中断后重跑即从断点续跑）
node_entry → node_parse → node_normalize → node_document_split → node_bge_embedding → node_import_milvus → node_enrich
  登记/去重    MinerU 解析   标题层级/表格/图片   正文按章节、表格独立    BGE-M3 稠密+稀疏     先写新版本再删旧版本   财务事实+文档摘要

查询图
node_query_plan → node_entity_confirm ─┬→ node_answer_output（澄清 / 库外拒答 / 投资建议 / 实时行情：固定话术）
                                       └→ node_fact_lookup ‖ node_summary_fetch ‖ node_search_embedding（按需并行）
                                             → node_rerank → node_answer_output（流式生成 + 合规守卫 + 代码渲染引用）
```

设计要点：

- **按问题类型取证据**：数值题查结构化财务事实（季报表格确定性解析），摘要题取文档摘要，其余走稠密 + 稀疏混合检索；LLM 只负责输出结构化查询计划和组织回答。
- **表格友好的切分**：表格独立成块，合并单元格展开、子表头识别、单位行并入，按“列名=值”线性化后计算向量。
- **可追溯**：模型只写证据编号 `[E1]`，来源列表（资料名称、内容类型、产品代码、文件名、页码、原件链接）由代码渲染。
- **合规**：禁用表达按句拦截（识别否定前缀与“描述骗局”语境），风险提示与时效提示由代码追加，证据不足时输出固定拒答话术。
- **多轮与澄清**：会话记录焦点对象；对象有歧义时反问，用户回复“第一个 / 债券那只”由代码匹配。

## 目录结构

```text
cli.py                    命令行入口
api/                      query_service.py（问答，端口 8001）  file_import_service.py（上传导入，端口 8000）
page/chat.html            聊天页面
processor/                import_processor/  query_processor/（state.py、main_graph.py、nodes/）
utils/                    clients/（Milvus、Mongo、MinIO、MinerU、会话）  lm/（对话、BGE-M3、精排）  其他工具函数
common/                   config/  logging/  models/  prompt/*.prompt  answer_templates.py
evaluation/               评测集自检、检索评测、答案评测
data/entities.json        实体表（手写）
data/eval/                评测集 fin_eval_set.jsonl 与历次结果 results/
tests/                    unit/  integration/
```

## 环境准备

- Windows x64，Python 3.12，[uv](https://docs.astral.sh/uv/)；本地 BGE-M3 模型（CPU 即可）
- 中间件：Milvus 2.5、MongoDB（开启认证）、MinIO
- 外部服务：MinerU 精准解析 API、阿里云百炼（OpenAI 兼容模式，LLM + qwen3-rerank + 视觉模型）

```bash
uv sync
```

复制 `.env.example` 为 `.env` 并填入真实值（`.env` 不入库）：

```bash
cp .env.example .env
```

检查中间件与外部服务连通性：

```bash
uv run python cli.py check
```

## 使用

导入资料目录（可重复执行：已就绪且内容未变的文件跳过，未完成的从断点续跑；解析结果按文件哈希缓存在 `data/artifacts/`）：

```bash
uv run python cli.py ingest "D:\path\to\金融\数据"
```

查看导入状态、调试检索、命令行问答（不带问题进入多轮交互）：

```bash
uv run python cli.py status
```

```bash
uv run python cli.py search "茅台一季度营业收入"
```

```bash
uv run python cli.py ask "华夏债券C的托管费是多少？"
```

启动问答服务后打开 <http://127.0.0.1:8001>：

```bash
uv run python -m api.query_service
```

启动导入服务（接口文档见 <http://127.0.0.1:8000/docs>）：

```bash
uv run python -m api.file_import_service
```

### 接口

| 服务 | 接口 | 说明 |
|---|---|---|
| 问答 8001 | `POST /query` | `{"question", "session_id"?}` → `text/event-stream`：`delta`（增量文本）/ `sources`（引用来源）/ `final`（session_id、kind、完整回答）/ `error`（面向用户的提示） |
| | `GET /sessions` | 最近的会话 |
| | `GET /sessions/{id}/messages` | 会话的问答记录 |
| 导入 8000 | `POST /documents` | 上传文件（pdf/doc/docx/ppt/pptx/md），后台串行导入 |
| | `GET /documents` | 文档与导入状态 |

`kind` 取值：`answer`（依据资料回答）、`refuse`（资料不足）、`clarify`（需要确认对象）、`decline_advice`（不提供投资建议）、`realtime_notice`（非实时数据）、`chitchat`。

## 测试与评测

```bash
uv run pytest
```

```bash
uv run pytest -m integration
```

评测集 67 题（产品、公告资讯、风险、知识、流程、负例、合规探针、4 组多轮链），金标为“文件名 + 关键原句”。默认只评检索，`--mode answer` 走完整问答链路：

```bash
uv run python -m evaluation.runner --tag <标签> --mode answer
```

## 评测结果

规则打分（不用 LLM 裁判），67 题 75 个轮次，存档 `data/eval/results/2026-09-19_M3-final.json`：

| 指标 | 结果 |
|---|---|
| 行为准确率（回答 / 拒答 / 澄清 / 不提供建议 / 时效提示是否符合预期） | 0.960 |
| 负例拒答率 / 正例误拒率 | 0.923 / 0.036 |
| 合规违规（禁用表达、投资建议措辞） | 0 |
| 必含内容通过率（三份季报与统计公报的数值题全部原值命中） | 0.96 |
| 多轮链（含澄清流程） | 4 / 4 |
| 证据含金标原句 / 引用文件含金标文件 | 0.909 / 1.0 |
| 延迟中位 / p90 | 7.3 s / 39.5 s（该轮评测期间精排接口多次超时；M2 同口径为 7.3 s / 17 s） |

纯检索基线（稠密 + 稀疏 + RRF，无实体过滤与精排）：Hit@5 0.873、MRR@10 0.670，文档级 Hit@5 1.0（`2026-09-19_M1-baseline.json`）。

LLM 输出在温度 0 下仍有波动：M2 以来各轮答案评测的行为准确率在 0.96～0.99 之间，差异主要来自规划分类与“依据不足”判断。

## 已知问题

- MinerU vlm 偶尔漏解析正文数字（招商银行季报少数正文数字缺失；《基金基础知识》为伪粗体排版，数字召回很低），表格数值不受影响。
- 负例“风险等级 R3 代表什么”模型回答“资料未披露”而非固定拒答话术（未编造，但未走标准话术）。
- 个别产品费率题检索未召回对应费用表（如易方达智造 C 的销售服务费）。
- CPU 上 BGE-M3 编码约 1.7 秒/切片，全量导入 17 个文件约 10 分钟；问答中位延迟约 7 秒。
