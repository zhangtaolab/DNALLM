# WINDOWS-KDA-VERIFY — Windows 真机 KDA 内核验证手册(WINDOWS #27 / WPAR-07)

> 目标机器:Windows + RTX 5080(sm_120)。本手册自包含:克隆本仓库的 Windows 机器按步骤执行即可。
> 背景:v1.3 的 18-01 提交(`399f9f1`)把 `triton-windows` 声明在 `fla` extra(现已退役)的 Windows
> 平台标记后;v1.4 的 28-01 提交(D-09)把 flash-linear-attention 与该 Windows 标记整体上移为
> `[project] dependencies` 基础依赖(`triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'`)。
> 本验证证明:真实 Windows 机器上基础安装(`-e .`)即自动拉入 triton-windows,且 KDA chunk 内核
> 能编译并执行。完成即销 WINDOWS #18/#27、标记 WPAR-07、退 T-18-SC SUS 旗。

## 前置条件

- Windows 10/11 + NVIDIA RTX 5080,驱动可用(`nvidia-smi` 正常)
- Python 3.11 或 3.12 + [uv](https://docs.astral.sh/uv/)(或只用 pip 也行,命令见注)
- 网络可达 PyPI

## 六步执行(在克隆的仓库根目录,PowerShell)

**Step 1 — 全新 venv 安装(关键:不许手动装 triton-windows,平台标记必须自动拉它)**

```powershell
uv venv .venv-kda --python 3.11
uv pip install --python .venv-kda\Scripts\python.exe -e .
```

(纯 pip 等价:`python -m venv .venv-kda` → `.venv-kda\Scripts\python.exe -m pip install -e .`)

**Step 2 — 验证 triton-windows 被自动拉入(版本必须是 3.7.1.post27)**

```powershell
uv pip show --python .venv-kda\Scripts\python.exe triton-windows
# 期望含:Version: 3.7.1.post27
```

**Step 3 — 记录 torch 版本与 CUDA 构建**

```powershell
.venv-kda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

**Step 4 — KDA import 契约(等价于 pytest 的 test_chunk_kda_importable_when_fla_installed)**

```powershell
.venv-kda\Scripts\python.exe -c "from fla.ops.kda.chunk import chunk_kda; print('import OK:', callable(chunk_kda))"
# 期望:import OK: True
```

**Step 5 — 编译并执行一个 KDA chunk 内核(真编译,不是 import;并与独立 fp32 参考对数)**

把下面存为 `kda_probe.py` 后运行 `.venv-kda\Scripts\python.exe kda_probe.py`:

```python
"""KDA chunk kernel compile+execute+correctness probe (WINDOWS #27, RTX 5080 sm_120).

Calls chunk_kda exactly the way fla.layers.kda does (fused gate). fla 0.5.2's raw
pre-computed-g path (use_gate_in_kernel=False) is defective — do not use it here.
"""

import torch
import torch.nn.functional as F
from fla.ops.kda.chunk import chunk_kda

assert torch.cuda.is_available(), "CUDA unavailable"
dev, dtype = "cuda", torch.bfloat16
B, T, H, K, V = 2, 256, 4, 64, 64
torch.manual_seed(0)
q = torch.randn(B, T, H, K, device=dev, dtype=dtype)
k = torch.randn(B, T, H, K, device=dev, dtype=dtype)
v = torch.randn(B, T, H, V, device=dev, dtype=dtype)
g_raw = torch.randn(B, T, H, K, device=dev, dtype=dtype)  # f_proj(x), pre-gate
beta_raw = torch.randn(B, T, H, device=dev, dtype=dtype)  # b_proj(x), pre-sigmoid
A_log = torch.log(torch.empty(H, device=dev, dtype=torch.float32).uniform_(1, 16))
dt_bias = torch.log(
    torch.expm1(torch.empty(H * K, device=dev, dtype=torch.float32).uniform_(0.1, 1.0))
)

o, final_state = chunk_kda(
    q,
    k,
    v,
    g_raw,
    beta_raw,
    A_log=A_log,
    dt_bias=dt_bias,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    output_final_state=True,
)
torch.cuda.synchronize()
assert o.shape == (B, T, H, V), o.shape
assert torch.isfinite(o).all(), "non-finite kernel output"
assert torch.isfinite(final_state).all(), "non-finite final state"

# independent fp32 sequential reference (same semantics as fla.layers.kda)
qf = F.normalize(q.float(), dim=-1)
kf = F.normalize(k.float(), dim=-1)
g_eff = -A_log.view(1, 1, H, 1).exp() * F.softplus(g_raw.float() + dt_bias.view(1, 1, H, K))
b_eff = beta_raw.float().sigmoid()
S = torch.zeros(B, H, K, V, device=dev)
outs = []
for t in range(T):
    Sd = S * torch.exp(g_eff[:, t]).unsqueeze(-1)
    u = b_eff[:, t, :, None] * (v[:, t].float() - torch.einsum("bhkv,bhk->bhv", Sd, kf[:, t]))
    S = Sd + torch.einsum("bhk,bhv->bhkv", kf[:, t], u)
    outs.append(torch.einsum("bhkv,bhk->bhv", S, qf[:, t]))
r = torch.stack(outs, dim=1) * (K**-0.5)
maxdiff = (o.float() - r).abs().max().item()
assert maxdiff < 0.01, f"kernel disagrees with reference: maxdiff={maxdiff}"
print("KDA kernel OK:", tuple(o.shape), o.dtype, "| final_state:", tuple(final_state.shape))
print(f"maxdiff vs fp32 reference: {maxdiff:.6f}")
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
```

(2026-10-19 修订:探针改为与 `fla.layers.kda` 完全等价的 fused-gate 调用——fla 0.5.2 的
裸 g 路径(`use_gate_in_kernel=False`,直接传预算 decay)有缺陷,见失败分诊;且该版本
`g` 形状契约为 `[B, T, HV, K]`。真实模型层从不走裸 g 路径,故本探针才是对 WPAR-07
「KDA chunk 内核可用」的忠实检验。真机复验:RTX 5080,maxdiff 0.000788。)

**Step 6 — 汇总证据**(六样,缺一不可):

1. venv 路径与创建命令(Step 1 原文)
2. `triton-windows` 版本(Step 2 输出)
3. torch 版本 + CUDA 构建(Step 3 输出)
4. import 结果(Step 4 输出)
5. 内核证据(Step 5 完整输出)
6. 日期

## 失败分诊

- **不要把探针改回裸 g 路径**(`use_gate_in_kernel=False`,直接传预算 decay)→ fla 0.5.2 该路径有缺陷(2026-10-19 RTX 5080 实测:负 g 首个 chunk 输出即 ~1e26、后续 inf/NaN;正 g 全 NaN)。真实调用方 `fla.layers.kda` 恒走 fused-gate 路径;同栈 `chunk_gla` 正常,缺陷仅限 KDA 裸 g 路径(值得报 fla 上游)。若探针的 fused-gate 调用与未来 fla 版本签名不符,以 `python -c "import inspect, fla.ops.kda.chunk as m; print(inspect.signature(m.chunk_kda))"` 输出为准做最小适配,并保持 fused-gate 形态。
- **Step 1 后 Step 2 查无 triton-windows** → 平台标记未生效,是回归:记录 `uv pip install` 的完整输出与 `pyproject.toml` 中 `[project] dependencies` 段的 triton-windows 声明原文,回报主仓。
- **Step 5 编译报错** → 逐字捕获报错栈(NVCC/ptas 报错尤其要全),回报主仓;不要自行降级 torch 或手装别的 triton。
- **CUDA 不可用 / 驱动问题** → 先 `nvidia-smi` 自检,非本验证范围。

## 完成后

把六样证据带回主仓库(Linux 侧):在 v1.3 分支上直接跑 `/gsd-execute-phase 18`——18-03 Task 2 将据此销 WINDOWS #18、标记 WPAR-07、退 T-18-SC SUS 旗(执行清单在
`.planning/milestones/v1.3-phases/18-packaging-install-honesty-atomic-extras-surgery/18-03-SUMMARY.md`)。

---

*Runbook 落盘于 2026-10-18,基线 commit `bf3b441`(分支 v1.3);适用于任何 ≥`399f9f1` 的后续 tip。*
*Step 5 修订于 2026-10-19:探针改为 `fla.layers.kda` 等价的 fused-gate 调用并加 fp32 参考对数(裸 g 路径在 fla 0.5.2 有缺陷,详见失败分诊);RTX 5080 真机复验通过,maxdiff 0.000788。*
*安装路线修订于 2026-10-29(28-05,D-09):两处可编辑安装命令改为基础安装形式 `-e .`(flash-linear-attention 与 triton-windows 声明已随 28-01 上移为 `[project] dependencies` 基础依赖,原 extra 已删除);KDA 探针代码与数值证据不变。*
