# 会话压缩与交接规则

本规则适用于 `/home/lfr/se3_trajopt`，会话压缩阈值为 **5 次**。

- 每个新会话独立计数。每轮用户请求开始、上下文压缩后恢复工作时，使用
  `.agents/skills/inspect-context-compactions/SKILL.md` 规定的只读检查器：

  ```bash
  python3 .agents/skills/inspect-context-compactions/scripts/inspect_context_compactions.py --threshold 5
  ```

- 只有报告的 `status` 为 `available` 时，才使用 `compaction_count` 和
  `threshold_reached`。报告为 `unavailable` 或 `inconsistent` 时，明确说明
  无法核实次数，不根据摘要、会话长度或 token 用量估算。
- 当前会话核实达到或超过 5 次后，在当轮回复中提供交接文档和可直接复制的
  新会话 Prompt，保存到 `.agents/handoffs/<时间戳>/HANDOFF.md` 和
  `NEW_SESSION_PROMPT.md`，同时保存检查器 JSON 报告作为计数证据。
- 交接文档写明当前目标、用户授权及约束、修改与验证结果、有效配置、关键
  文件和结果路径、未完成工作、下一步命令，以及正在运行的进程状态。
  新会话 Prompt 必须引用该交接文档，并携带本规则。
- 不因达到阈值而擅自终止正在运行的程序或取消已授权任务；先将当前任务
  推进到可检查、可恢复的状态，再如实交接。未完成的任务必须明确列出。
- 同一会话首次达到阈值时生成一次交接；后续若继续工作，必要时更新这份
  交接，不为每轮实验重复创建文档。新会话使用自身线程重新核验计数。
- 在回复中链接 `HANDOFF.md` 和 `NEW_SESSION_PROMPT.md`。不得声称已经自动
  新建会话、修改系统压缩策略或安装了后台监控。
- 检查会话日志时不得修改、截断、移动或删除 rollout 文件，也不得打印
  日志中的对话、工具输出或 `replacement_history`。


# 代码变更方案与审批规则

本规则适用于 `/home/lfr/se3_trajopt` 中所有涉及代码变更的操作。

- 代码新增、修改、删除、移动、重构和测试代码修改，均须先向用户提交具体修改方案；不得因改动简单、可逆或属于清理工作而跳过审批。
- 可以先进行只读诊断，核实当前文件、Git 状态和依赖，保护用户已有脏修改。
- 修改方案应明确涉及的文件、修改内容、保留范围和验证方式；**方案最后必须说明修改的目的或优势**。
- **只有用户明确批准同意方案后，才可执行代码变更**。批准前不得先行修改。
- 批准后仅执行获批范围；超出范围的代码变更须重新提交方案并等待用户批准。
