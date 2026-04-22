# Task18 Harness Connectivity

## 目标

把 `task_18_spreadsheet_summary` 的 focused DPO 数据采集跑在真实三机链路上，而不是本地意淫 prompt。

拓扑：

- Mac / control host
  - 启动 `ModelProxy`
  - 把 OpenClaw 的请求转发到 vLLM
- ECS
  - 跑 `openclaw agent`
- RunPod L40S
  - 跑 `vLLM + tool parser`
- RunPod A100
  - 跑训练

## 已确认的事实

### 1. 三台机器可连

- ECS:
  - `ssh root@8.163.82.224 -p 22 -i ~/.ssh/id_ed25519`
  - `openclaw` 在 `/usr/bin/openclaw`
- L40S:
  - `ssh root@195.26.232.162 -p 28610 -i ~/.ssh/id_ed25519`
- A100:
  - `ssh root@64.247.196.128 -p 13130 -i ~/.ssh/id_ed25519`

### 2. L40S 上 vLLM 本机可访问

- `curl http://127.0.0.1:8000/v1/models` 正常

### 3. ECS 不能直接访问 L40S 暴露的 vLLM 端口

直接让 ECS 打 `http://195.26.232.162:8000/v1` 不可靠，至少当前实测超时。

这意味着：

- 不能把 `baseUrl` 直接写成 L40S 公网地址
- 必须走 `ModelProxy + ssh -R`

### 4. 本地到 L40S 的 SSH 隧道可用，但本机验证要注意权限口径

- 采用本地转发：
  - `127.0.0.1:18021 -> L40S:8000`
- 在 Codex sandbox 里直接 `curl 127.0.0.1:18021` 会报：
  - `Operation not permitted`
- 这不是隧道坏了，而是本地 loopback 访问被 sandbox 限制。
- 用提权后的 `curl` 验证，`/v1/models` 正常返回。

结论：

- 本地 SSH 隧道是否可用，不能只看 sandbox 内的 `curl`
- 要结合：
  - `lsof -nP -iTCP:<port> -sTCP:LISTEN`
  - 远端 `lsof -nP -iTCP:8000 -sTCP:LISTEN`
  - 提权后的 `curl`

## 正确的连通方案

### 方案

1. Mac 建一个到 L40S 的本地转发
   - 例如 `127.0.0.1:18021 -> L40S:8000`
2. Mac 本机启动 `ModelProxy`
3. ECS 通过 `ssh -R` 把 `localhost:<proxy_port>` 反向映射到 Mac 的 `ModelProxy`
4. ECS 上的 OpenClaw agent 配置成：
   - `baseUrl = http://127.0.0.1:<proxy_port>/v1`
5. `ModelProxy` 再把请求转发到：
   - `http://127.0.0.1:18021/v1`

### 为什么这样最稳

- ECS 不需要直接打 RunPod 公网端口
- 只依赖 SSH 已经打通的链路
- 训练时的 `ModelProxy` 架构和现有 RL 代码一致
- 代码复用 `rl/test_e2e_vllm.py` / `rl/agent_loop/openclaw_agent_loop.py`

## 当前 collector 的实现约束

`rl/data_construction/collect_task18_harness_focused_dpo.py` 现在应该走：

- 本机 `ModelProxy`
- `ssh -R`
- ECS OpenClaw
- 本机转发到 `vLLM_BASE_URL`

而不是：

- `execute_openclaw_task(... base_url=公网地址 ...)`

因为后者在当前三机环境下不满足真实连通性。

## Smoke 实录

### 已打通的部分

`task18` 的 harness smoke 已经证明三机链路可工作：

1. Mac 启动 `ModelProxy`
2. ECS 通过 `ssh -R` 打回 Mac 上的 `ModelProxy`
3. `ModelProxy` 再转发到 `L40S vLLM`
4. ECS transcript 成功写出

真实 transcript 里已经拿到了首跳坏动作：

- `read quarterly_sales.csv`
- `read company_expenses.xlsx`

同时，这次真实 request 还暴露了一个关键纠偏：

- `tools = 24`

不是之前静态抓模板时看到的 `17`。

这说明：

- OpenClaw 真实 runtime prompt 仍会随环境和技能池变化
- 后续 focused DPO 数据构造，不能再盲信旧的静态 `runtime_prompt_template.json`
- 应优先使用 harness 当次实际抓回的：
  - `messages`
  - `tools`

这说明：

- harness-driven focused DPO 数据采集链路已经不是“概念”
- 已经能采到真实 `rejected` 首跳

### 发现的新坑

如果让 OpenClaw 继续往下跑完整题，第二轮常出现：

- `Connection error.`

表现是：

- 第一跳 transcript 已经落盘
- 但后续 turn 不稳定
- collector 会无谓地卡在收尾阶段

### 对 collector 的结论

对于 focused DPO，这个采集器不应该等完整 task 结束。

更合理的策略是：

1. 只采第一跳
2. 第一条 assistant action 落盘后，短暂等待 transcript flush
3. 主动结束远端 run
4. 立即构造 DPO pair

原因：

- 我们训练的就是 first-step policy repair
- 不需要完整轨迹
- 这样更快，也更稳

这已经体现在：

- `rl/data_construction/collect_task18_harness_focused_dpo.py`

当前实现会在第一跳响应后尽快收尾，而不是等待整题自然结束。

## 复用建议

后续类似工作都按这个分工：

- A100: 训练
- L40S: vLLM 推理服务
- ECS: OpenClaw runtime
- Mac: 控制面 / ModelProxy / 数据采集脚本

不要再尝试：

- ECS 直连 RunPod vLLM 公网端口
- 直接把公网 `base_url` 写进 remote OpenClaw agent

除非先证明该端口从 ECS 侧稳定可达。

## 操作清单

下次复用这条链，按这个顺序：

1. 在 L40S 确认 vLLM 本机监听 `8000`
2. 在 Mac 起本地 SSH 隧道到 `127.0.0.1:18021`
3. 用提权 `curl http://127.0.0.1:18021/v1/models` 验证
4. 在 Mac 运行 harness collector
5. collector 内部再起：
   - `ModelProxy`
   - `ssh -R`
   - ECS OpenClaw
6. 只采第一跳 transcript，立即落 pair
