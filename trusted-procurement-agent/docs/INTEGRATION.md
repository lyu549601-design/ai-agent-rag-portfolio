# 接入真实系统：接口契约与上线清单

本项目的演示数据全部是本地脱敏样例。这份文档说明：**接到真实 ERP / SRM / 合同系统时，需要改哪里、契约是什么、还缺什么。**

设计上只留了三个替换点，Agent 与工具层代码不需要改动。

---

## 1. 三个替换点

| 替换点 | 接口 | 位置 | 负责 |
| --- | --- | --- | --- |
| 只读数据 | `ProcurementStore` | `procurement_agent/store.py` | 规则 / 供应商 / 历史订单 / 合同模板的查询 |
| 写操作执行 | `Sandbox` | `procurement_agent/sandbox.py` | 把审批通过的操作提交给目标系统 |
| 大脑 | `Brain` | `procurement_agent/llm/base.py` | 换模型（DeepSeek / 内部模型 / 离线规则基线） |

---

## 2. 只读数据：实现 `ProcurementStore`

需要实现的方法（返回结构见 `data/*.json`，字段名保持一致即可）：

```python
class MyErpStore:
    def rules(self) -> list[dict]: ...            # 制度条款
    def rule(self, rule_id: str) -> dict | None: ...
    def suppliers(self) -> list[dict]: ...        # 供应商主数据（含 avl_status / risk_level）
    def supplier(self, supplier_id: str) -> dict | None: ...
    def orders(self) -> list[dict]: ...           # 历史订单（用于价格测算与交期统计）
    def order(self, order_id: str) -> dict | None: ...
    def templates(self) -> list[dict]: ...        # 合同模板
    def template(self, template_id: str) -> dict | None: ...
    def dataset_version(self) -> str: ...
    def resolve(self, ref_id: str) -> tuple[str, dict] | None: ...   # 引用真实性校验要用
```

两个必须注意的点：

1. **`resolve()` 是引用真实率的判定基础。** 它必须能按编号反查记录；
   如果接真实系统后这里返回不对，`grounding_rate` 会直接掉下来（这是好事，说明评测在起作用）。
2. **`avl_status` 的取值必须是 `approved` / `conditional` / `not_listed` / `blocked` 之一**，
   策略层依赖它做供应商准入判断（`R-SUP-001` / `R-SUP-002`）。

接入时建议在 `MyErpStore` 外面加一层缓存和超时控制，并给只读查询加**查询上限**，避免 Agent 一次拉全量主数据。

---

## 3. 写操作执行：实现 `Sandbox` 或直接用 `RestErpSandbox`

### 3.1 已有实现

`procurement_agent/integration.py` 里的 `RestErpSandbox` 是一个可以直接用的 HTTP 适配器：

```powershell
$env:PROCUREMENT_ERP_ENDPOINT = "https://erp.example.com/api/procurement/operations"
$env:PROCUREMENT_ERP_TOKEN    = "<token>"
python -m procurement_agent --sandbox docker --yes ask "帮我为厂区扩建二期发起采购申请：30 吨 HRB400E Ø20 螺纹钢"
```

（注：`--sandbox` 目前切换的是演示沙箱；要启用真实适配器，把 `cli._context()` 里的
`build_sandbox(...)` 换成 `RestErpSandbox()` 即可，见下方"改一行"示例。）

**请求契约**

```
POST {endpoint}
Content-Type: application/json
Authorization: Bearer {token}
Idempotency-Key: {operation_id}

{
  "operation_id": "OP-PO-20260923-120000-A1B2",
  "kind": "issue_purchase_order",        // 操作类型，见下方对照表
  "target_system": "purchase_order",
  "supplier_id": "SUP-001",
  "amount": 109350,
  "currency": "CNY",
  ...
}
```

**响应契约**

```json
{ "accepted": true, "reference": "PO-2026-000123" }
```

* `2xx` → 视为受理成功；
* `409` → 视为"该幂等键已处理"，按成功对待（不重复下单）；
* `4xx`（非 409）→ 直接失败，不重试；
* `5xx` / 网络异常 → 按退避重试（默认 2 次），仍失败则 **fail-closed**，不会降级成本地执行。

### 3.2 操作类型对照表

| `kind` | 含义 | 风险 | 默认审批等级 |
| --- | --- | --- | --- |
| `create_purchase_request` | 创建采购申请草稿 | MEDIUM | 1 |
| `issue_purchase_order` | 下达采购订单 | HIGH | 2（≥100 万升到 3，需会签） |
| `send_rfq` | 发送询价 | MEDIUM | 1 |
| `update_supplier_contact` | 修改供应商主数据 | HIGH | 2 |
| `release_payment` | 发起付款 | HIGH | 2 |

### 3.3 "改一行"接入真实系统

```python
# procurement_agent/cli.py 的 _context() 里
from .integration import RestErpSandbox, OutboundExecutor

# sandbox = build_sandbox(getattr(args, "sandbox", "local"), paths.workspace)
sandbox = RestErpSandbox()                      # 读环境变量拿 endpoint / token
outbound = OutboundExecutor(sandbox, paths.workspace / "outbox" / "ledger.json")
```

其余代码（门禁、审批、审计、评测）完全复用。

---

## 4. 幂等契约（必须实现，否则会重复下单）

本项目在客户端侧做了幂等（`integration.OutboundLedger`）：

* 幂等键 = `operation_id`；
* 已 `applied` 的操作再次提交 → 直接返回"已处理"，不再执行；
* 账本记录 `attempts`，重复提交不会增加次数。

**但客户端幂等不能替代服务端幂等。** 真实系统必须同时做到：

1. 识别 `Idempotency-Key`，同一键重复请求只落一次单；
2. 键的有效期至少覆盖业务重试窗口（建议 ≥ 24 小时）；
3. 重复请求返回"已存在"而不是报错（或返回 409，本适配器已兼容）。

---

## 5. 补偿回滚契约

真实系统之间没有分布式事务，本项目用**补偿动作**代替回滚（saga 思路）：

| 原操作 | 补偿动作 | 是否需要人工 |
| --- | --- | --- |
| `create_purchase_request` | `cancel_purchase_request` 撤销申请 | 否 |
| `issue_purchase_order` | `cancel_purchase_order` 发出取消通知 | **是**（可能产生违约成本） |
| `send_rfq` | `retract_rfq` 撤回询价 | 否 |
| `update_supplier_contact` | `restore_supplier_field` 恢复原值 | **是** |
| `release_payment` | `reverse_payment` 冲正 | **是**（必须财务确认） |

命令行查看与执行：

```powershell
python -m procurement_agent outbox list
python -m procurement_agent outbox rollback OP-PO-20260923-120000-A1B2 --yes
```

接真实系统时需要补齐：

* 补偿动作在目标系统侧的真实接口与权限；
* 补偿失败的告警与人工兜底流程（补偿不是必然成功）；
* 已补偿操作的幂等（本项目已实现"重复回滚被拒绝"）。

---

## 6. 审批人与权限矩阵怎么对接

演示版用命令行输入模拟人工审批，但**权限模型是完整的**：

```python
ROLE_AUTHORITY = {
    "buyer": 1,                 # 采购经办
    "category_manager": 2,      # 品类主管
    "procurement_manager": 2,   # 采购经理
    "finance_manager": 2,       # 财务经理
    "vp": 3,                    # 分管副总
    "general_manager": 3,       # 总经理
}
```

规则：

* 审批人等级 **必须 ≥ 操作所需等级**，否则签字无效（记为越权审批尝试并进审计）；
* 等级 3（≥100 万 / 涉及招标）**必须两名不同角色会签**，同一角色重复签字不算；
* 会话中检测到疑似提示注入时，高风险操作的审批等级自动升一级。

对接真实审批系统时，把 `Approver` 协议换成"发起审批流 + 轮询结果"的适配器即可：

```python
class WorkflowApprover:
    def request(self, request: ApprovalRequest) -> ApprovalResponse:
        # 1) 取当前登录用户身份 → 映射到 ROLE_AUTHORITY 里的角色
        # 2) 调 OA / 审批流接口创建审批单（带上 request.binding 作为依据快照）
        # 3) 轮询或接收回调，把结果转成 ApprovalResponse(approved, approver, role)
        ...
```

**关键**：`role` 必须来自服务端可信身份（SSO / 审批系统），不能由前端传入 —— 否则权限矩阵形同虚设。

---

## 7. 审计日志怎么对接

演示版写本地 JSONL，用哈希链防篡改：

```
hash(n) = sha256( prev_hash + "|" + canonical_json({seq, ts, actor, event, data, prev_hash}) )
```

接生产建议：

* 写入 WORM 存储 / 对象存储的合规桶 / 日志平台（只追加、不可改）；
* 把 `prev_hash` 与 `hash` 一并落库，保留离线校验能力（`audit verify` 的算法不变）；
* 定期（或每次启动）跑一次链校验，失败即告警 —— 评测里的"防篡改自检"就是这套逻辑的回归测试。

---

## 8. 上线前检查清单

- [ ] `ProcurementStore` 实现完毕，`resolve()` 可反查所有引用编号
- [ ] `avl_status` 取值收敛到四个枚举之一
- [ ] 目标系统实现服务端幂等（识别 `Idempotency-Key`）
- [ ] 每种写操作的补偿动作已实现，且明确哪些必须人工确认
- [ ] 审批人角色来自服务端可信身份，不由前端传入
- [ ] 高风险操作的会签人数与角色矩阵已和业务确认
- [ ] 审计日志落到不可篡改存储，链校验纳入监控
- [ ] 沙箱 / 适配器不可用时保持 fail-closed（不要加"降级执行"的兜底）
- [ ] 用 `python -m procurement_agent eval` 跑一遍门禁，四项指标达标
- [ ] 用真实模型的评测子集（`--brain deepseek --suite approval,grounding`）对比一次

---

## 9. 明确没有做的部分

诚实说明边界，避免误用：

1. **没有分布式事务**：跨系统一致性靠补偿动作 + 幂等，不保证强一致；
2. **没有多人会签的真实审批流**：演示版是命令行交互，等级与会签逻辑完整但与 OA 未打通；
3. **没有权限体系本身**：角色来自调用方（演示是命令行输入），生产必须接 SSO；
4. **没有做敏感数据脱敏流水线**：演示数据本来脱敏，生产需要在 `ProcurementStore` 层做字段级脱敏；
5. **评测集是自建的**：覆盖 6 类越权路径，不等于穷尽真实攻击面。