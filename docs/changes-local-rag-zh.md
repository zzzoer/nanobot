# 本次 nanobot 完善说明

本次在原有“自适应工具路由、会话管理”未提交实现上继续完善，并新增本地文档 RAG。保留项目原有模型提供商和运行入口，无需新建 embedding 服务或向量数据库。

## 改了哪些文件

| 文件 | 修改与原因 |
|---|---|
| `nanobot/agent/tools/knowledge.py`（新增） | 本地 SQLite FTS5/BM25 检索，支持中英文、重叠分块、来源路径与行号、内容哈希更新和删除同步。限制索引与输出大小，验证知识库路径边界。 |
| `nanobot/agent/tools/session_manage.py`（完善原草稿） | 实现创建、切换、列表、当前、回默认会话；保留中文名称，校验会话归属，使用 ContextVar 隔离并发聊天。 |
| `nanobot/session/manager.py` | 持久化当前会话选择；重启恢复；补标题和旧会话检查；只读查询不再创建空会话；拒绝冲突存储键覆盖其他会话。 |
| `nanobot/agent/tool_router.py`（完善原草稿） | 英文词边界匹配、中文关键词、工具依赖、动态 MCP 工具保留；有界使用历史；准确统计；默认关闭诊断日志且不记用户原文。 |
| `nanobot/agent/loop.py` | 注册 RAG 与会话工具；让后续消息真正使用选中的会话；记录实际执行工具，删除重复写入的人工 assistant trace；多模态请求只用用户文本做路由。 |
| `nanobot/agent/context.py` | 指导模型使用知识库检索、引用文件行号、说明证据不足，并把检索内容视作参考资料。 |
| `nanobot/config/schema.py` | 新增 `tools.routing.enabled` 和 `logDecisions` 配置。 |
| `nanobot/cli/commands.py`、`nanobot/nanobot.py` | 将路由配置传入 CLI、gateway、API 和 SDK 的 AgentLoop。 |
| `tests/agent/test_tool_router.py`、`test_knowledge_integration.py`、`test_loop_session_routing.py`（新增） | 测路由边界、真实 runner 检索与引用、会话下一轮切换/重启、线程作用域和并发隔离。模型使用本地确定性替身。 |
| `tests/tools/test_knowledge_tool.py`、`test_session_manage.py`（新增） | 真实临时文件、SQLite、中文与边界分块、索引更新/删除、路径检查、引用截断、会话归属与持久化。 |
| `tests/tools/test_tool_validation.py` | 超时测试使用当前 Python 的 `time.sleep`，替代 Windows 不支持的 shell `sleep` 命令。 |
| `.github/workflows/knowledge-sessions.yml`（新增） | 为当前工作分支配置 Linux Python 3.11/3.12/3.13 和 Windows Python 3.12 的定向回归测试。 |
| `.gitignore`、`README.md`、`docs/local-knowledge-and-sessions.md` | 忽略知识库原文/索引/路由日志，增加功能入口、配置、运行命令和边界说明。 |

## 如何使用 RAG

1. 找到配置中的 `agents.defaults.workspace`，默认是 `~/.nanobot/workspace`。它与 `D:\code\nanobot` 源码目录不同。
2. 在 workspace 下创建 `knowledge/`，放入 UTF-8 的 `.md`、`.txt`、`.rst` 文件。
3. 正常运行 nanobot，提问：“根据知识库总结这份项目的实验步骤，并给出来源。”
4. agent 调用 `search_knowledge(query, limit=5, refresh=True)`，将找到的文本交给原有模型生成回答。

首次检索自动建索引，后续默认按内容哈希刷新；移除文件会同步移除索引记录。结果包含 `source`、`start_line`、`end_line` 和 `content`，便于检查引用。缓存位于 workspace 的 `.nanobot/knowledge.sqlite3`，其中含文档原文。

这是关键词/BM25 RAG，不是 embedding 语义检索。PDF、Word、图片需先转为支持的文本格式。检索在本地完成；引用片段仍会作为工具结果发送给你已配置的模型提供商。

## 会话和路由

可以自然语言请求“新建一个叫论文阅读的会话”“列出当前聊天的会话”“切换到论文阅读”“回到默认会话”。切换从下一条普通用户消息开始生效，不删除任何历史；后台 system/subagent 消息沿用原来的默认会话。

路由可在已有配置中关闭或开启元数据日志：

```json
{
  "tools": {
    "routing": {
      "enabled": true,
      "logDecisions": false
    }
  }
}
```

没有明确关键词时回退到全部工具；动态 MCP、会话和知识库工具保持可发现。路由是启发式筛选，复杂任务如果需要完整工具集，可以设 `enabled: false`。

## 验证方法与边界

```shell
python -m pip install -e ".[dev]"
python -m pytest tests/agent tests/tools tests/config tests/cli tests/test_nanobot_facade.py tests/test_openai_api.py -q
```

新增集成测试执行真实 runner、工具注册表、检索索引和会话存储，使用假模型提供商以避免外部调用。这证明功能连接与数据流，不代表已测量真实模型的回答准确率。部分真实 symlink 测试依赖系统权限，缺权限时跳过；Windows junction 越界检查另有实际测试。

2026-11-01 本地 Python 3.14 验证：上述相关回归测试 **495 passed、5 skipped、32 subtests passed**；5 项跳过来自 Windows symlink 创建权限。所有本次修改的 Python 文件通过 Ruff，`git diff --check` 通过。另在原始 `D:\code\nanobot` 同步目录执行 RAG 主循环集成测试，4 项通过。测试存在依赖弃用、aiohttp AppKey 和 Windows 异步子进程清理警告，未将它们描述为无警告的完整上游测试。

工作从用户的四个未提交文件继续，原稿已备份在工作区 `research/nanobot-original/`。GitHub 成果属于 `HKUDS/nanobot` 的个人 fork 改进，应区分上游原项目与本次新增功能。
