# Project learnings

## [LRN-20260926-001] best_practice

**Logged**: 2026-09-26
**Status**: verified for the recorded diagnostic case
**Area**: trajectory serialization

For piecewise contact controls, split both numerical integration and serialized polynomial intervals at every control breakpoint. Carry the exact previous integrated state into the next interval. Aligning only the saved grid can leave dense-output derivative errors above the original dynamics gate.

The recorded 200ms LW support comparison reduced saved-curve Newton error from about 0.0087N (unaligned), to 3.6e-5N (storage aligned), to 4.25e-9N (integration and storage aligned). Original COM floor, stopping reserve, and force-slew violations remained and independently rejected the trajectory.

Evidence: `trajopt_solutions_batch/leg_to_wheel_buffer_closure_20260926_123509/support_200ms_{initial,aligned,segmented}/final/independent_acceptance.json`.

Do not infer complete feasibility from the serialization improvement or from a single instantaneous support solve.


## [LRN-20261010-001] correction

**Logged**: 2026-10-10T10:54:23.920671+08:00
**Priority**: high
**Status**: promoted
**Promoted**: AGENTS.md
**Area**: docs

### Summary
任何涉及代码修改的操作，先提交具体方案，在方案最后说明修改目的或优势，得到用户明确批准后才执行。

### Details
用户明确要求：“任何涉及到代码修改的操作都要先给出修改方案并在方案最后给出修改的目的或者优势，经过我批准同意后才可以修改”。
根据用户进一步明确要求，已将本规则写入项目根目录 AGENTS.md，供后续会话读取。此要求立即作为当前协作规则生效，适用于代码新增、修改、删除、移动、重构及测试代码修改，不以改动简单、可逆或属于清理工作为由跳过审批。只读诊断可以先进行。

### Suggested Action
1. 先只读检查当前文件、Git状态与依赖，保护历史脏修改。
2. 提交具体修改方案：涉及的文件、修改内容、保留范围和验证方式。
3. 方案最后单独说明修改的目的或优势。
4. 等待用户明确批准方案；收到批准前不执行涉及代码变更的操作。
5. 批准后仅执行获批范围；超出范围的代码变更另行提交方案并等待批准。
学习日志记录本身不授权任何代码修改，也不授权求解、提交或推送。

### Metadata
- Source: user_feedback
- Related Files: .learnings/LEARNINGS.md, AGENTS.md
- Tags: approval, change-plan, purpose, preserve-dirty-worktree
- Pattern-Key: workflow.code_change_plan_and_explicit_approval
- Recurrence-Count: 1
- First-Seen: 2026-10-10
- Last-Seen: 2026-10-10

---
