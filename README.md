# Edge_Inferencer
![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=pink)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Rockchip%20|%20Qualcomm%20|%20x86-orange)
![Backend](https://img.shields.io/badge/Backend-RKNN%20|%20QNN%20|%20ONNX-red)
![Inference](https://img.shields.io/badge/Mode-Single%20|%20Thread%20Pool%20|%20Process%20Pool-lightgrey)

## 📖 概述
统一边缘设备推理引擎 —— 一套 Python API，自动适配 Rockchip NPU、Qualcomm HTP 和 ONNX Runtime<br>
无需修改业务代码，切换模型文件即可在不同硬件后端间无缝迁移

> 同时本项目也是本小姐🍃的项目[Focus-Finder](https://github.com/YeWenxuan64/Focus-Finder)的模型推理后端喵~

---

### ✨ 功能亮点

- **三后端统一接口** — RKNN（Rockchip NPU）/ QNN（Qualcomm HTP）/ ONNX Runtime（CPU），`put()` / `get()` / `release()` 三方法
- **自动模型识别** — 根据文件后缀 (`.rknn` / `.bin` / `.onnx`) 自动选择推理后端，零配置
- **多模式推理** — 单线程执行、线程池并发、进程池 + 共享内存，按需选择性能与开销的平衡
- **多核 NPU 支持** — 指定 NPU 核心 (Core 0/1/2/ALL)，线程池/进程池自动轮询分发
- **格式自动转换** — 支持 NHWC / NCHW 输入，后端自动处理维度转置
- **延迟初始化** — 首次 `put()` 时才加载模型，减少启动开销
- **易于集成** — 三行代码接入自有项目，作为 [Edge_ModelDeploy](https://github.com/YeWenxuan64/Edge_ModelDeploy) 和 [Focus-Finder](https://github.com/YeWenxuan64/Focus-Finder) 的推理后端



| 模块 | 说明 |
|------|------|
| [`ai_inferencer.py`](./ai_inferencer.py) | 统一入口 `AIInferencer`，自动识别模型类型并路由 |
| [`rknn_inferencer.py`](./rknn_inferencer.py) | Rockchip NPU 推理：`RknnExecutor`（单线程）/ `RknnThreadPool`（线程池） |
| [`qnn_inferencer.py`](./qnn_inferencer.py) | Qualcomm HTP 推理：`QnnExecutor`（单线程）/ `QnnProcessPool`（进程池+共享内存） |
| [`onnx_inferencer.py`](./onnx_inferencer.py) | ONNX Runtime 推理：`OnnxExecutor`（CPU） |

> **底层 SDK：** Rockchip NPU 使用 **rknn-toolkit-lite2**；Qualcomm HTP 使用 **QAI AppBuilder**（基于 QAIRT SDK）；ONNX 使用 **onnxruntime**。



## 🏗️ 架构

```
    ┌────────────────────────────────────────┐
    │        AIInferencer (Unified)          │
    │   Auto-detect .rknn / .bin / .onnx     │
    └─────┬──────────────┬─────────────┬─────┘
          │              │             │
    ┌─────▼──────┐ ┌─────▼──────┐ ┌────▼─────┐
    │  RKNPU     │ │  HTP       │ │  ONNX    │
    │ (Rockchip) │ │ (Qualcomm) │ │ (CPU)    │
    ├────────────┤ ├────────────┤ ├──────────┤
    │ Executor   │ │ Executor   │ │ Executor │
    │ ThreadPool │ │ ProcessPool│ │          │
    │            │ │            │ │          │
    └────────────┘ └────────────┘ └──────────┘
```



## 📁 项目结构

```
Edge_Inferencer/
├── ai_inferencer.py        # 统一入口：AIInferencer，自动路由
├── rknn_inferencer.py      # Rockchip NPU 推理后端
├── qnn_inferencer.py       # Qualcomm HTP 推理后端
├── onnx_inferencer.py      # ONNX Runtime 推理后端
├── requirements_rknn.txt   # RKNN 依赖
├── requirements_qnn.txt    # QNN 依赖
├── requirements_onnx.txt   # ONNX 依赖
├── README.md
└── LICENSE
```


## 📦 安装

### 🍴 1. 克隆

```bash
git clone https://github.com/YeWenxuan64/Edge_Inferencer.git
cd Edge_Inferencer
```

### 🔧 2. 安装依赖

根据目标平台安装对应依赖：

| 平台 | 命令 |
|------|------|
| **Rockchip NPU** (RK3588/RK3576) | `pip install -r requirements_rknn.txt` |
| **Qualcomm HTP** (QCS6490) | `pip install -r requirements_qnn.txt` + 从源码编译安装 `qai_appbuilder` |
| **ONNX Runtime** (x86 / 通用) | `pip install -r requirements_onnx.txt` |

> ⚠️ QNN 的 `qai_appbuilder` 需从源码编译安装，可参考 [QNN (qai_appbuilder) 编译安装](QNN_INSTALL.md)


## 🏃 快速开始

### 🔌 统一接口

```python
from ai_inferencer import AIInferencer
import numpy as np

# 初始化 — 根据模型文件后缀自动选择后端
model = AIInferencer(
    model_path='model.rknn',      # .rknn / .bin / .onnx
    cores=(0,),                    # NPU 核心，默认 (0,)
    mult_task=False                # 是否启用线程池并发
)

# 推理 — 输入 NHWC 格式的 numpy 数组
input_data = [np.random.randint(0, 255, (1, 320, 640, 3), dtype=np.uint8)]
result = model.put(input_data, input_format='nhwc')

# 释放资源
model.release()
```

### ⚖️ 多模式对比

| 模式 | 适用场景 | 创建方式 |
|------|----------|----------|
| **单线程 Executor** | 简单推理，最低开销 | `AIInferencer(model_path, mult_task=False)` |
| **线程池 ThreadPool** | 多核 NPU 并发推理 | `AIInferencer(model_path, cores=(0, 1), mult_task=True)`（RKNN/QNN） |
| **进程池 ProcessPool** | QNN 极致性能，绕过 GIL | `AIInferencer(model_path, cores=(0, 1), mult_task=True)`（QNN 自动使用进程池） |

### 🧵 线程池 / 进程池用法

```python
from ai_inferencer import AIInferencer
import numpy as np

# 双核并发推理
model = AIInferencer(
    model_path='model.rknn',
    cores=(0, 1),          # 使用 NPU Core 0 和 Core 1
    mult_task=True          # 启用线程池
)

for frame in video_stream:
    input_data = [preprocess(frame)]
    model.put(input_data)          # 提交推理任务（轮询分发到不同核心）
    result = model.get(block=True) # 获取结果（阻塞等待）

model.release()
```

> **RKNN** 并发使用线程池 `ThreadPoolExecutor`，**QNN** 并发使用进程池 `multiprocessing.Process` + 共享内存拷贝

---

### ⚠️ 多任务模式帧错位说明
由于RKNPU和QNN的并发限制。如RKNPU的宣传算力为所有核心的算力之和，且rk3588、rk3576等又是多核心的NPU，所以单NPU核心的算力有限，且确认了在同一个NPU上并发虽不报错是负收益。<br>而在高通HTP上情况则要复杂，为了不报错，只能使用进程隔离。<br>所以只能以帧为单位分发去并发，让提交当前帧任务之后可以立即获取上一帧的结果。

`RknnThreadPool` 和 `QNNProcessPool` 的 `put()` / `get()` 存在**固定的帧偏移**，偏移量等于任务数 (`thread_num`)。

**原因：** 首次调用 `put()` 初始化线程池时，会用同一帧向每个核心提交一个推理任务（填满 `thread_num` 个队列槽位），后续每次 `put()` 只追加一个任务。因此 `get()` 返回的结果始终滞后于当前 `put()` 的帧。

**以任务 (`cores=(0, 1), mult_task=True`) 为例：**

```
time line:  put(Frame 0) → put(Frame 1) → put(Frame 2) → put(Frame 3) 
                 ↓              ↓              ↓              ↓      
returns:    get():Frame 0  get():Frame 0  get():Frame 1  get():Frame 2
```

| 操作 | get() 返回 | 说明 |
|------|-----------|------|
| 第 1 次 `put(Frame 0)` + `get()` | Frame 0 | 首次 put 用 Frame 0 填满 2 个核心槽位 |
| 第 2 次 `put(Frame 1)` + `get()` | Frame 0 | 偏移 1 帧 |
| 第 3 次 `put(Frame 2)` + `get()` | Frame 1 | 偏移 1 帧 |
| 第 n 次 `put(Frame n)` + `get()` | Frame n-1 | **稳定偏移 = thread_num - 1** |

**总偏移量 = `thread_num`**（含首次初始化）。即：

| 核心数 | 偏移帧数 |
|--------|----------|
| 1 核 (`cores=(0,)`) | 0/1 帧 |
| 2 核 (`cores=(0, 1)`) | 1 帧 |
| 3 核 (`cores=(0, 1, 2)`) | 2 帧 |

**影响与应对：**

- **视频流实时推理** — 偏移仅造成几帧延迟，通常可忽略，不影响可视化效果
- **需要帧级对齐的场景**（如逐帧后处理、结果与帧号严格对应）— 需在应用层手动补偿偏移量，或使用单线程 `Executor` 模式（`mult_task=False`）

---

### ⚠️ 高通 QAI AppBuilder 并发推理的局限性

QNN 并发推理面临一系列由 `QNNContext` 内部复杂性引发的固有限制：

**根本原因：** 部分 soc(如QCS6490) 的 HTP（DSP）通常不具备多个独立逻辑核心，其"并发"本质上是通过提高运算器（如矩阵乘法累加器）利用率与内存带宽利用率来实现的，而非真正的多核并行。

**Python 端的限制：**
- `QNNContext` 内部状态复杂，**不支持多线程并发访问**，必须加锁保护
- 由于 Python GIL 与 QNN 锁的叠加，线程池在 QNN 场景下负收益
- 因此只能退而求其次，在 **Python 端使用多进程（`multiprocessing`）** 来手动并发

**隐藏的稳定性风险：**

| 风险 | 表现 | 推测原因 |
|------|------|------|
| **QNNContext 创建失败** | 进程池初始化时无法创建上下文 | 多进程同时争抢 DSP 资源，HTP 固件资源耗尽<br>~~高通工程师更更爆~~ |
| **cDSP 内存申请失败** | 推理中途报错、进程异常退出 | DSP 侧共享内存碎片化，多进程各自持有独立上下文导致内存压力倍增<br>~~高通工程师更更爆~~ |
| **与其他库冲突** | 与opencv库等同时使用时崩溃或卡死 | 多进程争抢同一硬件资源（DSP、CMA 内存池），缺乏全局调度<br>>~~高通工程师更更爆~~ |
| **轮询开销** | 实际大部分时间耗在轮询多个推理任务而非真实并行计算 | HTP 的核心调度策略是时分复用而非空间并行 |

**其他问题：** 小女子手上的 `Radxa Dragon Q6A` (QCS6490)的 `dsp(HTP)` 可能并非较为完整的神经网络加速器，部分计算需要在CPU端完成，不仅并发性能受限，还影响其他CPU任务<br>(例：在并发时，由于偷用大量CPU资源导致预处理和后处理时间明显拉长，从而拖慢整组并发推理任务)

> **建议：** 若追求稳定性，优先使用**单线程 `QnnExecutor` 模式**（`mult_task=False`），避免多进程带来的资源竞争。仅在性能压力测试或明确知晓硬件余量时，才考虑开启多进程并发。<br>
>
> 叶姐姐🍃: 不是哥们，你这大高通的soc高出来的性能就被这些奇妙开销给磨掉了，rk恩情还不完✋😭🤚

---


## 📚 API 参考

### 🎯 `AIInferencer`

统一入口，自动识别模型类型并路由到对应后端。

```python
AIInferencer(model_path: str, cores: tuple[int] = (0,), mult_task: bool = False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | — | 模型文件路径，根据后缀自动识别类型 |
| `cores` | `tuple[int]` | `(0,)` | NPU 核心编号，支持 `0`/`1`/`2`/`-1`(ALL) |
| `mult_task` | `bool` | `False` | 是否启用线程池/进程池并发模式 |

**方法：**

| 方法 | 签名 | 说明 |
|------|------|------|
| `put()` | `put(input_data: list[np.ndarray], input_format: str = 'nhwc')` | 提交推理输入，返回结果（单线程）或 `None`（并发模式） |
| `get()` | `get(block: bool = True)` | 获取推理结果（并发模式），`block=False` 非阻塞 |
| `release()` | `release() -> bool` | 释放所有资源 |

**支持的模型格式：**

| 后缀 | 后端 | 平台 |
|------|------|------|
| `.rknn` | RKNN Lite | Rockchip NPU (RK3588/RK3576/RK3566) |
| `.bin` | QAIRT | Qualcomm HTP (QCS6490 QCS8550 QCS9075) |
| `.onnx` | ONNX Runtime | CPU (通用) |

---

### `RknnExecutor` / `RknnThreadPool`

Rockchip NPU 推理后端。

```python
# 单线程
from rknn_inferencer import RknnExecutor
executor = RknnExecutor(model_path='model.rknn', cores=(0,))
result = executor.put(input_data, input_format='nhwc')
executor.release()

# 线程池（多核并发）
from rknn_inferencer import RknnThreadPool
pool = RknnThreadPool(model_path='model.rknn', cores=(0, 1))
pool.put(input_data)
result = pool.get(block=True)
pool.release()
```


### `QnnExecutor` / `QnnProcessPool`

Qualcomm HTP 推理后端。

```python
# 单线程
from qnn_inferencer import QnnExecutor
executor = QnnExecutor(model_path='model.bin')
result = executor.put(input_data, input_format='nhwc')
executor.release()

# 进程池（共享内存零拷贝，绕过 GIL）
from qnn_inferencer import QnnProcessPool
pool = QnnProcessPool(model_path='model.bin', cores=(0, 1))
pool.put(input_data)
result = pool.get(block=True)
pool.release()
```


### `OnnxExecutor`

ONNX Runtime CPU 推理后端 (用于在转换为 onnx 模型后进行测试用)。

```python
from onnx_inferencer import OnnxExecutor

executor = OnnxExecutor(model_path='model.onnx')
executor.set_providers(['CPUExecutionProvider'])  # 可选：指定执行提供者

result = executor.put(input_data, input_format='nhwc')

executor.release()
```

---

### ⏱️ `timeit` 装饰器

内置推理耗时 / FPS 统计工具，滑动窗口计算平均耗时（窗口大小 30 次调用）。

支持两种用法：

| 形式 | 测量内容 |
|------|----------|
| `@timeit`<br>或(func = timeit(func)) | 单次函数执行耗时（`end - start`） |
| `@timeit(measure_cycle_time=True)`<br>或(func = timeit(func, measure_cycle_time=True)) | 两次调用之间的周期时间（`start - last_start`），包含主线程空闲/等待时间 |


输出示例（每 ~1 秒打印一次）：

| 模式 | 输出格式 |
|------|----------|
| 默认 | `infer_frame: 12.345 ms, fps: 81.002` |
| 周期模式 | `infer_frame (per cycle): 15.678 ms, fps: 63.783` |

---

## 🔗 集成示例

### 🎯 在 YOLO 推理中使用
以本小姐的项目[Yolo11_ModelDeploy](https://github.com/YeWenxuan64/Yolo11_ModelDeploy)为例

```python
import numpy as np
import cv2
from ai_inferencer import AIInferencer

model_path = "path/to/your/model.rknn" # "path/to/your/model.bin"

# 初始化推理引擎
model = AIInferencer(
    model_path=model_path,
    cores=(0,),
    mult_task=False
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 320))
    input_tensor = np.expand_dims(resized, axis=0)  # (1, 320, 640, 3)

    # 推理
    result = model.put([input_tensor], input_format='nhwc')

    # 后处理...
    # frame = draw(result)

    # 可视化...
    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
model.release()
```

### 🚀 多核并发推理

```python
from ai_inferencer import AIInferencer

model_path = "path/to/your/model.rknn"

# 双核并发，交替使用 Core 0 和 Core 1
model = AIInferencer(
    model_path=model_path,
    cores=(0, 1),
    mult_task=True
)

frames:list[np.ndarray, ...] = [...]  # 视频帧列表

for frame in frames:
    model.put([preprocess(frame)])

    result = model.get(block=True)
    if result is not None:
        postprocess(result)

model.release()
```


## ⚠️ 注意事项

- **首次调用 `put()` 才会加载模型** — 延迟初始化减少启动时间
- **并发模式下的 `put()` 返回 `None`** — 需要通过 `get()` 获取结果
- **`get(block=False)` 非阻塞** — 结果未就绪时返回 `None`，不会卡死主线程
- **QNN 进程池使用共享内存** — 自动管理创建与清理，正常退出时自动释放
- **NCHW 输入** — 后端会自动转置为 NHWC（RKNN/QNN）或保持 NCHW（ONNX）


## 📄 License

[MIT License](./LICENSE) — Copyright (c) 2026 叶文轩
