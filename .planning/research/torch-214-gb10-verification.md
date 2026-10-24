# torch>=2.12 GB10 验证 runbook（起草 2026-10-24）

目的：在改 pyproject 的 torch 上限之前，先在 GB10（`dnallm-nightly` self-hosted runner）实测
mamba-ssm/causal_conv1d 对 torch 2.14.x 的源码编译与运行兼容性。**本机 Mac 只负责起草与准备
分支；全部 CUDA/mamba 验证在 GB10 上通过 workflow dispatch 完成，Mac 不跑任何 GPU 内容。**

## 已核查事实（Mac 侧，全部实测，2026-10-24）

- 上限来历：`5721ec3`（2026-05-08）为 RTX 5090 放宽 `<=2.7 → <2.12`，是当时的前瞻性边界，
  不是"2.12 有缺陷"
- Mac 本地 venv 已升 torch 2.14.1：MPS forward 通（无需 checkpoint 补丁，max|mps-cpu|=1.3e-5）、
  CPU logits 与 2.11 逐位一致、496 离线测试全绿（详见 STATE.md 2026-10-24 Session Continuity）
- wheels：mac arm64 与 cu130 索引均有 2.14.1（cp310–314，x86_64+aarch64）
- 依赖链无上界冲突：transformers `torch>=2.6` / accelerate `>=2.0` / bitsandbytes `<3,>=2.4` / peft `>=1.13`
- `tests/test_extras_guard.py` 不钉 torch 版本边界 → spike 分支只改 pyproject，无需同改测试
- ty 门禁不受影响：stub environment 钉死 `torch==2.10.0`（ty-stub-constraints.txt）
- **唯一真正的未知数**：`mamba-ssm==2.3.2.post1` + `causal_conv1d==1.7.0` 对 2.14 的编译/运行。
  PyPI 元数据两者均无 torch 上界声明（风险纯在源码编译层）。test-mamba 每次
  `--no-cache-dir --no-build-isolation` 重编 → dispatch 一次即实测。

## 预检（owner 动作，dispatch 前必须完成）

0. ⚠️ **runner 现状：`edgexpert-3eee` offline**（2026-10-24 检查；main 上两个 scheduled CI
   run 已排队 13h+，即此原因）。开机 / 恢复 runner 服务后再 dispatch；恢复后排队中的
   scheduled run 也会自动开跑（注意它们跑的是 main 的旧代码，rice/ClinVar 类失败属预期）。

   ```bash
   gh api repos/zhangtaolab/DNALLM/actions/runners --jq '.runners[] | [.name,.status,.busy]'
   ```

## 验证步骤（全部发生在 GB10）

1. **push spike 分支**（分支已在本机 Mac 备好：`spike/torch-214-verification`，单 commit，
   pyproject 7 处 `<2.12` → `<2.15`：base + cpu/cuda124/cuda126/cuda128/cuda130/mamba extras）：

   ```bash
   git push -u origin spike/torch-214-verification
   ```

2. **dispatch**（owner；workflow_dispatch 满足 test-mamba 的 `if` disjunct；owner-repo 分支
   而非 fork PR，符合 dispatch-only 自托管策略）：

   ```bash
   gh workflow run CI --ref spike/torch-214-verification
   ```

3. **盯跑**（内核编译是长杆，预计 2–3h；job 自带 180min timeout 兜底）：

   ```bash
   gh run list --branch spike/torch-214-verification --limit 1
   gh run watch <run-id>
   ```

4. **通过标准**（三条全满足才算绿）：
   - a. `uv pip install -e ".[full,fla]"` 解析成功且 torch 落在 **2.14.x**（看 install 日志确认）
   - b. `uv pip install -e ".[mamba]" --no-cache-dir --no-build-isolation` 源码编译绿
     （mamba-ssm 2.3.2.post1 + causal_conv1d 1.7.0 在 2.14 下编译通过）
   - c. `pytest tests/ -v -m "not slow"` census 绿（与常规 mamba lane 同口径）

## 失败诊断矩阵

| 症状 | 含义 | 处置 |
|---|---|---|
| resolver 找不到可用 torch 2.14 / 装上后 import 报 no kernel image | base 依赖走 PyPI 默认索引，默认 CUDA variant 不含 GB10（sm_120）kernel | spike 分支追加一 commit：test-mamba 的 install 行加 `cuda130` extra（`[[tool.uv.index]]` 映射已存在），重新 dispatch |
| mamba-ssm / causal_conv1d **编译失败** | 真正的阻断：2.3.2.post1 不兼容 2.14 的 C++/CUDA API | 抓编译错误全文；调研 mamba-ssm 上游新版本（其元数据已引入 tilelang/tvm-ffi/quack-kernels 依赖，上游在演进，可能有面向新 torch 的 release）；结论回写本文件 |
| 编译绿但 mamba 测试红（kernel numerics / API 运行时错误） | 运行时不兼容 | `mamba-test-logs-*` artifact 自动上传；按 run id 取回，按 A.5 分类纪律定性 |
| dispatch 后长时间 queued | runner 又掉线 | 回到预检第 0 步 |

## 通过后（真实 ceiling raise，落在 v1.4）

- 同样 7 处 `<2.12` → `<2.15` 落到 v1.4 正式 commit，**message 引用 spike run id**（仓库引 run id 惯例）
- nightly 无锁文件 → 首个 24h 的 coverage-nightly（03:00 UTC）与 example-nightly（05:30 UTC）
  会隐式全体跳到 2.14.x —— 盯一整轮（v1.3 的 2026-10-07 教训：长寿命分支零矩阵暴露首晚爆三类问题）
- ty 门禁与 stub env 不受影响（torch==2.10.0 钉死）

## 回滚

- spike 阶段：删远端/本地 spike 分支即可，v1.4 未动
- 真实 raise 后首晚爆红：revert 单 commit 即回到 `<2.12`

## 证据落档

dispatch 时在 `.planning/async-jobs/` 建 manifest（沿用 19-07-nightly.json 字段格式，
`plan_id` 用 `spike-torch-214`），终态回填 run id + 三条通过标准的逐项结论。
