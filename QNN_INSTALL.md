# 🔨 qai appbuilder 编译安装

QNN 后端依赖 `qai_appbuilder` Python 包，需从 [quic/ai-engine-direct-helper](https://github.com/quic/ai-engine-direct-helper) 源码编译为 `.whl` 后安装。

## ✅ 前置条件

- **硬件：** Qualcomm Dragonwing™ 开发板
- **Python：** 3.10+
- **工具：** Git、CMake、build-essential
- **网络：** 可联网下载依赖

## 1. 安装系统依赖

```bash
sudo apt update
sudo apt install -y cmake gcc g++ build-essential python3-dev

#假如你在使用python虚拟环境，请在你正在使用的环境下安装
pip install wheel setuptools pybind11 build
```

## 2. 下载并配置 QAIRT SDK

网页下载: [Qualcomm Software Center](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community?osArch=Any&osType=All&version=2.39.0.250926)<br>
链接下载: [Qualcomm_AI_Runtime_SDK_2.38.0.250901.zip](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.39.0.250926/v2.39.0.250926.zip)

> - 推理模型所使用的SDK版本建议**高于等于**转换时所用的SDK版本<br>
> - 设备代号 (如 QCS6490 -> Hexagon v68) 可在QAIRT附赠的文档查找
> qairt/VERISON_NAME/docs/QNN/general/overview.html#supported-snapdragon-devices

```bash
# 下载 SDK
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.39.0.250926/v2.39.0.250926.zip

# 解压
unzip v2.39.0.250926.zip

# 配置环境变量（替换为实际路径）
source ./qairt/v2.39.0.250926/bin/envsetup.sh
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2:$LD_LIBRARY_PATH

# 按平台设置 Hexagon 库路径
# QCS6490 (Hexagon v68):
export ADSP_LIBRARY_PATH=$QNN_SDK_ROOT/lib/hexagon-v68/unsigned

```


## 3. 克隆源码

```bash
git clone https://github.com/quic/ai-engine-direct-helper.git --recursive
cd ai-engine-direct-helper
git submodule update --init --recursive
```

## 4. 编译 .whl

在项目根目录执行：

```bash
python setup.py bdist_wheel

```

编译完成后，`dist/` 目录下会生成 `qai_appbuilder-*.whl` 文件。

## 5. 安装 .whl

```bash
pip install dist/qai_appbuilder-*.whl
```

## ❓ 常见问题

| 问题 | 解决方法 |
|------|----------|
| qai-appbuilder 更更爆 | 不知道 |
| 编译找不到 QNN SDK | 确认 `QNN_SDK_ROOT` 和 `LD_LIBRARY_PATH` 环境变量已正确设置 |
| `setup.py` 报错 | 确保已安装 `cmake` 和 `build-essential` |



---

### ⚠️ 高通 QAI AppBuilder 并发推理的局限性

QNN 并发推理面临一系列由 `QNNContext` 内部复杂性引发的固有限制：

**根本原因：** 部分 soc(如QCS6490) 的 HTP（DSP）通常不具备多个独立逻辑核心，其"并发"本质上是通过提高运算器（如矩阵乘法累加器）利用率与内存带宽利用率来实现的，而非真正的多核并行。

**Python 端的限制：**
- `QNNContext` 内部状态复杂，**不支持多线程并发访问**，必须加锁保护
- 由于 Python GIL 与 QNN 锁的叠加，线程池在 QNN 场景下负收益
- 因此只能退而求其次，在 **Python 端使用多进程（`multiprocessing`）** 来手动并发
- qai appBuilder 用于并发推理的 `QNNContextProc` 存在bug, 创建和释放时可能会触发。<br>
同时基于`QNNContextProc`的并发推理速度比本小姐写的`QnnProcessPool` **Python multiprocessing based 多进程QNN推理器** 快不了几个ms（不过内存占用少了些）

**隐藏的稳定性风险：**

| 风险 | 表现 | 推测原因 |
|------|------|------|
| **QNNContext 创建失败** | 进程池初始化时无法创建上下文 | 不知道<br>~~高通工程师更更爆~~ |
| **cDSP 内存申请失败** | 推理中途报错、进程异常退出 | 不知道<br>~~高通工程师更更爆~~ |
| **与其他库冲突** | 与opencv库等同时使用时崩溃或卡死 | 多进程争抢同一内存资源（DSP、CMA 内存池）<br>>~~高通工程师更更爆~~ |
| **轮询开销** | 实际大部分时间耗在轮询多个推理任务而非真实并行计算 | HTP 的核心调度策略是时分复用而非空间并行 |

**其他问题：** 小女子手上的 `Radxa Dragon Q6A` (QCS6490)的 `dsp(HTP)` 可能并**非较为完整的神经网络加速器**，部分计算需要**在CPU端完成**，不仅并发性能受限，还**影响其他CPU任务**<br>
(例：在并发时，由于偷用大量CPU资源导致预处理和后处理时间明显拉长，从而拖慢整组并发推理任务)

> **建议：** 若追求并发稳定性，优先使用`QnnProcessPool` **Python multiprocessing based 多进程QNN推理器**，叶姐姐🍃亲手维护。<br>
>
> 叶姐姐🍃: 不是哥们，你这大高通的soc高出来的性能就被这些奇妙开销给磨掉了，rk恩情还不完✋😭🤚
