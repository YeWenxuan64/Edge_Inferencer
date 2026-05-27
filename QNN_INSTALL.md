

# 🔨 QNN (qai_appbuilder) 编译安装

QNN 后端依赖 `qai_appbuilder` Python 包，需从 [quic/ai-engine-direct-helper](https://github.com/quic/ai-engine-direct-helper) 源码编译为 `.whl` 后安装。

## ✅ 前置条件

- **硬件：** Qualcomm Dragonwing™ IQ9075 或 QCS6490 开发板
- **系统：** Ubuntu 24.04（ARM64）
- **Python：** 3.8+（推荐 3.12）
- **工具：** Git、CMake、build-essential
- **网络：** 可联网下载依赖

## 1️⃣ Step 1: 安装系统依赖

```bash
sudo apt update
sudo apt install -y cmake build-essential python3.12-dev

pip install requests==2.32.3 \
    py3-wget==1.0.12 \
    tqdm==4.67.1 \
    importlib-metadata==8.5.0 \
    qai-hub==0.30.0
```

## 2️⃣ Step 2: 下载并配置 QAIRT SDK

从 [Qualcomm Software Center](https://softwarecenter.qualcomm.com/#/catalog/item/Qualcomm_AI_Runtime_SDK) 下载 **Qualcomm AI Runtime (QAIRT) SDK**，当前推荐版本 `v2.40.0.251030`：

```bash
# 下载 SDK
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip

# 解压
unzip v2.40.0.251030.zip

# 配置环境变量（替换为实际路径）
export QNN_SDK_ROOT=/path/to/v2.40.0.251030
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2:$LD_LIBRARY_PATH

# 按平台设置 Hexagon 库路径
# QCS6490 (Hexagon v68):
export ADSP_LIBRARY_PATH=$QNN_SDK_ROOT/lib/hexagon-v68/unsigned
# IQ9075 (Hexagon v73):
# export ADSP_LIBRARY_PATH=$QNN_SDK_ROOT/lib/hexagon-v73/unsigned
```


## 3️⃣ Step 3: 克隆源码

```bash
git clone https://github.com/quic/ai-engine-direct-helper.git --recursive
cd ai-engine-direct-helper
```

## 4️⃣ Step 4: 编译 .whl

在项目根目录执行，根据平台选择对应命令：

```bash
# QCS6490 (Hexagon v68)
python setup.py bdist_wheel --toolchains aarch64-oe-linux-gcc11.2 --hexagonarch 68

# IQ9075 (Hexagon v73)
# python setup.py bdist_wheel --toolchains aarch64-oe-linux-gcc11.2 --hexagonarch 73
```

编译完成后，`dist/` 目录下会生成 `qai_appbuilder-*.whl` 文件。

## 5️⃣ Step 5: 安装 .whl

```bash
pip install dist/qai_appbuilder-*.whl
```

## 6️⃣ Step 6: 验证安装

```bash
python -c "import qai_appbuilder; print('QAI AppBuilder installed successfully')"
```

## ❓ 常见问题

| 问题 | 解决方法 |
|------|----------|
| `undefined symbol: _ZN3tbb...` (TBB 冲突) | `sudo apt remove libtbb-dev` 或 `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtbb.so.12` |
| 编译找不到 QNN SDK | 确认 `QNN_SDK_ROOT` 和 `LD_LIBRARY_PATH` 环境变量已正确设置 |
| `setup.py` 报错 | 确保已安装 `cmake` 和 `build-essential` |


