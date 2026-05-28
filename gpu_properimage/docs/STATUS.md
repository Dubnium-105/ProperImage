# ProperImage GPU Acceleration Status

## 当前状态 (2026-05-27)

### 两条加速路线

| 方案 | 路径 | 方法 | 仓库 | 进度 |
|------|------|------|------|------|
| 方案7 | FFT频域 | CuPy/cuFFT | **本仓库** | 📋 规划中 |
| 方案8 | 空间域 | CUDA fused kernel | Dubnium-105/ois | 🔧 gpu-ois-v0.2分支 完成TDD |

### 方案7 详细状态

**目标**: 加速 properimage/operations.py 的 subtract() 函数

**瓶颈分析** (codex-proxy + 人工交叉验证):
- scipy.optimize 回调内反复 IFFT: 50-90%
- 方差校正 V_en/V_er FFT/IFFT: 20-40%
- PSF 渲染 + 背景: 5-20%

**加速方案** (按改动量):

| # | 改动量 | 方案 | 预期加速 |
|---|--------|------|---------|
| 1 | 小 | pyfftw 显式配置 plan 缓存/线程 | 1.2-3x |
| 2 | 小 | 减少 beta/shift 优化迭代 | 2-10x |
| 3 | 大 | CuPy/cuFFT 全路径常驻 GPU | 8-20x |

**关键挑战**:
- scipy.optimize 在 CPU，cost() 必须最小化 GPU↔CPU 搬运
- sep.Background (C 库) 无 GPU 等价实现
- fourier_shift 内部用 numpy FFT，是 pyfftw/CuPy 盲区
- SingleImage 内部状态 (interped_hat, PSF) 需 GPU 化

**未开始**: 代码实现尚未启动，优先完成方案8验证后再开。

### 完整分析文档

GPU 加速方案: 见本文件及 PLAN.md
项目规格: Dubnium-105/ois/gpu_ois/docs/ (SPEC.md, ARCH.md, API.md 等)
