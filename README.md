# FinKB 金融知识库

面向普通用户的金融知识问答系统：用自然语言提问，系统从已导入的金融资料（上市公司季报、基金产品资料概要、理财风险揭示书、宏观政策报告、投资者教育问答等）中查找依据，流式给出带引用来源、符合合规要求的回答。

定位是资料查询与解读工具：**不提供个性化投资建议，不承诺收益**；无资料时明确告知，不编造数据。

## 架构

两张 LangGraph 图，节点逻辑写在 `processor/*/nodes/`，工具函数在 `utils/`。

```text
导入图（一次处理一个文件，从头跑到尾；已就绪且内容未变的文件直接跳过）
node_entry → node_parse → node_chunk → node_enrich → node_index
  登记/去重    MinerU 解析   标题层级/表格/图片，正文按章节、表格独立   财务事实与摘要也做成切片   BGE-M3 稠密+稀疏，写入后清掉旧切片

查询图（提示词主导：怎么回答由模型按规则手册判断，代码只保留三道闸门）
node_query_plan → node_gather_evidence → node_answer_output
  改写问题、抽对象    对象解析 + 混合检索 + 精排 + 编号    一次调用定判定与正文，再过闸门
```

设计要点：

- **提示词主导的判定**：回答 / 拒答 / 澄清 / 不提供建议 / 时效提示 / 寒暄由模型按 `common/prompt/answer_system.prompt` 里的规则手册判断，元信息写在正文后的 `<<<META>>>` 一行 JSON 里（kind、引用编号、提示语标记）。
- **代码保留的三道闸门**：禁用表达按句拦截（识别否定前缀与“描述骗局”语境）、引用编号校验（删掉证据里不存在的编号）、没有任何证据时直接输出固定拒答话术。
- **证据只有一种**：正文、表格、图片描述、财务指标事实、文档摘要都是切片，走同一条检索路径——稠密 + 稀疏混合检索（Milvus 内置 RRF 融合）取 20 条，云端精排取 8 条；问题指向具体对象时把检索限定在其文档内。季报"主要会计数据"表在导入时被确定性地拆成一行一条的事实切片，数值题答的是原值。
- **表格友好的切分**：表格独立成块，合并单元格展开、子表头识别、单位行并入，按“列名=值”线性化后计算向量。
- **可追溯**：模型只写证据编号 `[E1]`，来源列表（资料名称、内容类型、产品代码、文件名、页码、原件链接）由代码渲染。
- **多轮与澄清**：会话记录焦点对象与待澄清候选；用户回复“第一个 / 债券那只”由代码匹配，不再问模型。

## 目录结构

```text
cli.py                    命令行入口
api/                      query_service.py（问答，端口 8001）  file_import_service.py（上传导入，端口 8000）
page/chat.html            聊天页面
processor/                import_processor/  query_processor/（state.py、main_graph.py、nodes/）
utils/                    clients/（Milvus、Mongo、MinIO、MinerU、会话）  lm/（对话、BGE-M3、精排）  其他工具函数
common/                   config/（各组件一个配置文件）  logging/  prompt/*.prompt  answer_templates.py
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

导入资料目录（可重复执行：已就绪且内容未变的文件跳过，其余重做；解析结果按文件哈希缓存在 `data/artifacts/`，重做不会重复调用 MinerU）：

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

规则打分（不用 LLM 裁判），67 题 75 个轮次，存档 `data/eval/results/2026-09-20_top8-answer.json`：

| 指标 | 现在 | 简化前 |
|---|---|---|
| 行为准确率（回答 / 拒答 / 澄清 / 不提供建议 / 时效提示是否符合预期） | 1.0 | 0.973 |
| 负例拒答率 / 正例误拒率 | 1.0 / 0 | 1.0 / 0.018 |
| 合规违规（禁用表达、投资建议措辞） | 0 | 0 |
| 必含内容通过率（三份季报与统计公报的数值题全部原值命中） | 0.96 | 0.96 |
| 多轮链（含澄清流程） | 4 / 4 | 3 / 4 |
| 证据含金标原句 / 引用文件含金标文件 | 0.982 / 1.0 | 0.927 / 1.0 |
| 延迟中位 / p90 | 3.3 s / 4.8 s | 3.8 s / 5.8 s |

八个类别（产品、公告资讯、风险、知识、流程、负例、合规探针、多轮）的行为准确率都是 1.0。

纯检索基线（混合检索取 20，无精排）：Hit@5 0.855、MRR@10 0.663，文档级 Hit@5 1.0（`2026-09-20_simplify-batch3.json`）。

LLM 输出在温度 0 下仍有波动，同名模型的判定口径也会随时间变化：2026-09-20 实测“一季度营收”这类问题一天之内从“公告资讯”漂成“实时行情”，公告类抽样 10 题漂 4 题，靠收紧提示词纠正。

## 已知问题

- MinerU vlm 偶尔漏解析正文数字（招商银行季报少数正文数字缺失；《基金基础知识》为伪粗体排版，数字召回很低），表格数值不受影响。
- 个别产品费率题检索未召回对应费用表（如易方达智造 C 的销售服务费）。
- 判定交给模型后，个别题的归类会随模型状态波动；最近一轮 75 个轮次全对，历史各轮在 0.96～1.0 之间。
- CPU 上 BGE-M3 编码约 1.7 秒/切片，全量导入 17 个文件约 12 分钟；问答中位延迟约 3.3 秒。
