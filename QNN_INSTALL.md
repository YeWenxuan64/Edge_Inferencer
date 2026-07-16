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


