# WINDOWS-KDA-VERIFY — Windows 真机 KDA 内核验证手册(WINDOWS #27 / WPAR-07)

> 目标机器:Windows + RTX 5080(sm_120)。本手册自包含:克隆本仓库的 Windows 机器按步骤执行即可。
> 背景:v1.3 的 18-01 提交(`399f9f1`)把 `triton-windows` 声明在 `fla` extra 的 Windows 平台标记后
> (`triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'`),本验证证明该声明在真实
> Windows 机器上自动生效,且 KDA chunk 内核能编译并执行。完成即销 WINDOWS #18/#27、标记 WPAR-07、
> 退 T-18-SC SUS 旗。

## 前置条件

- Windows 10/11 + NVIDIA RTX 5080,驱动可用(`nvidia-smi` 正常)
- Python 3.11 或 3.12 + [uv](https://docs.astral.sh/uv/)(或只用 pip 也行,命令见注)
- 网络可达 PyPI

## 六步执行(在克隆的仓库根目录,PowerShell)

**Step 1 — 全新 venv 安装(关键:不许手动装 triton-windows,平台标记必须自动拉它)**

```powershell
uv venv .venv-kda --python 3.11
uv pip install --python .venv-kda\Scripts\python.exe -e ".[fla]"
```

(纯 pip 等价:`python -m venv .venv-kda` → `.venv-kda\Scripts\python.exe -m pip install -e ".[fla]"`)

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

**Step 5 — 编译并执行一个 KDA chunk 内核(真编译,不是 import)**

把下面存为 `kda_probe.py` 后运行 `.venv-kda\Scripts\python.exe kda_probe.py`:

```python
"""KDA chunk kernel compile+execute probe (WINDOWS #27, RTX 5080 sm_120)."""
import torch
from fla.ops.kda.chunk import chunk_kda

assert torch.cuda.is_available(), "CUDA unavailable"
dev, dtype = "cuda", torch.bfloat16
B, T, H, D = 2, 256, 4, 64
torch.manual_seed(0)
q = torch.randn(B, T, H, D, device=dev, dtype=dtype)
k = torch.randn(B, T, H, D, device=dev, dtype=dtype)
v = torch.randn(B, T, H, D, device=dev, dtype=dtype)
g = -torch.rand(B, T, H, device=dev, dtype=dtype) - 1e-3   # log decay, 负值
beta = torch.rand(B, T, H, device=dev, dtype=dtype).sigmoid()

o, final_state = chunk_kda(q, k, v, g, beta, output_final_state=True)
torch.cuda.synchronize()
assert o.shape == (B, T, H, D), o.shape
assert torch.isfinite(o).all(), "non-finite kernel output"
print("KDA kernel OK:", tuple(o.shape), o.dtype, "| final_state:", tuple(final_state.shape))
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
```

(若该参数组合与安装的 fla 版本签名不符,以 `python -c "import inspect, fla.ops.kda.chunk as m; print(inspect.signature(m.chunk_kda))"` 的输出为准做最小适配——5 个位置参数 q/k/v/g/beta 是稳定契约。)

**Step 6 — 汇总证据**(六样,缺一不可):

1. venv 路径与创建命令(Step 1 原文)
2. `triton-windows` 版本(Step 2 输出)
3. torch 版本 + CUDA 构建(Step 3 输出)
4. import 结果(Step 4 输出)
5. 内核证据(Step 5 完整输出)
6. 日期

## 失败分诊

- **Step 1 后 Step 2 查无 triton-windows** → 平台标记未生效,是回归:记录 `uv pip install` 的完整输出与 `pyproject.toml` 中 fla extra 段原文,回报主仓。
- **Step 5 编译报错** → 逐字捕获报错栈(NVCC/ptas 报错尤其要全),回报主仓;不要自行降级 torch 或手装别的 triton。
- **CUDA 不可用 / 驱动问题** → 先 `nvidia-smi` 自检,非本验证范围。

## 完成后

把六样证据带回主仓库(Linux 侧):在 v1.3 分支上直接跑 `/gsd-execute-phase 18`——18-03 Task 2 将据此销 WINDOWS #18、标记 WPAR-07、退 T-18-SC SUS 旗(执行清单在
`.planning/milestones/v1.3-phases/18-packaging-install-honesty-atomic-extras-surgery/18-03-SUMMARY.md`)。

---

*Runbook 落盘于 2026-10-18,基线 commit `bf3b441`(分支 v1.3);适用于任何 ≥`399f9f1` 的后续 tip。*
