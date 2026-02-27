# Qwen3.5-35B-RTX5070-Laptop-Guide
Qwen3.5, MoE, RTX5070, 8GB VRAM, llama-cpp-python, Blackwell
🚀 Qwen3.5-35B on RTX 5070 Laptop (8GB VRAM): 极限部署指南
🏆 项目成就：全球首批在 8GB 显存笔记本 (RTX 5070 Blackwell 架构) 上成功运行 Qwen3.5-35B-MoE 模型的完整实践记录。
⚠️ 核心结论：通过手动编译源码、暴力修补 API、锁定 n_gpu_layers=1，实现了稳定中文对话。本指南解决了 CUDA 内部错误、数值溢出乱码、Windows 编码死结三大难题。
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Model: Qwen3.5-35B](https://img.shields.io/badge/Model-Qwen3.5--35B--MoE-blue)
![Hardware: RTX 5070 Laptop](https://img.shields.io/badge/Hardware-RTX%205070%20Laptop%20(8GB)-green)
![Status: Stable](https://img.shields.io/badge/Status-Stable%20(CPU+1%20GPU%20Layer)-brightgreen)


📖 目录
背景与挑战
硬件与软件环境
核心解决方案 (The Golden Path)
避坑指南：两大致命陷阱
快速开始：一键运行脚本
性能数据与实测报告
致谢与贡献


🔍 背景与挑战
Qwen3.5-35B 是阿里巴巴最新发布的 MoE 架构大模型，官方推荐显存 >24GB。然而，对于拥有 RTX 5070 Laptop (8GB VRAM) 的用户来说，直接运行几乎是不可能的任务：
架构滞后：官方 llama-cpp-python 包尚未支持 qwen35moe 架构。
API 断裂：底层 llama.cpp 重构导致大量 Python 绑定函数失效。
显存墙：8GB 显存无法容纳模型权重，混合加载易导致 CUDA 错误或数值溢出。
新架构 Bug：RTX 50 系列 (Blackwell) 驱动与 cuBLAS 存在兼容性隐患。
显示乱码：Windows 控制台无法正确渲染 UTF-8 输出。
本项目记录了如何通过手动编译、代码修补、极限参数调优，在 8GB 显存笔记本上实现稳定、可用的中文实时对话。


🛠️ 硬件与软件环境
表格
组件	规格	备注
GPU	NVIDIA GeForce RTX 5070 Laptop	8GB GDDR7, Blackwell 架构
CPU	AMD Ryzen 9 9955HX	16 核心 32 线程
RAM	32GB DDR5	双通道
OS	Windows 11 Pro	PowerShell / CMD
Model	Qwen3.5-35B-A3B-Q4_K_S.gguf	4-bit 量化版 (~20GB)
Backend	llama-cpp-python (Custom Build)	手动替换 vendor 源码编译

软件环境需要miniconda，自己装好后如下配置：
这是为您整理的 **Conda 环境配置全记录**，您可以直接将其添加到您的实践报告或 GitHub README 的“环境准备”章节中。

---


如果你想自己编译，就需要如下的conda环境，否则python运行run_qwen35.py即可


### 🐍 Conda 环境配置指南 (Qwen35 专属)

#### 1. 创建独立环境
为了保证系统 Python 不受影响，必须创建一个独立的虚拟环境。我们指定 **Python 3.11**，因为它是目前 `llama-cpp-python` 和 CUDA 13.1 兼容性最好的版本。

```bash
# 创建名为 qwen35 的环境，指定 Python 3.11
conda create -n qwen35 python=3.11 -y

# 激活环境
conda activate qwen35
```

#### 2. 安装核心编译依赖 (Windows 必做)
在手动编译 `llama-cpp-python` 之前，必须安装 C++ 编译工具和 CMake。如果不安装，`pip install` 时会报错找不到编译器。

```bash
# 安装 CMake 和 Ninja (构建工具)
conda install -c conda-forge cmake ninja -y

# 安装 Git (用于拉取源码，如果尚未安装)
conda install -c conda-forge git -y
```

#### 3. 设置 CUDA 编译环境变量 (关键步骤)
为了让编译出的库支持 GPU 加速（即使是只开 1 层），需要告诉编译器使用 CUDA。
*注意：对于 RTX 5070 (Blackwell)，我们需要确保编译器能识别新架构。*

```bash
# 设置环境变量，启用 CUDA 加速
# Windows PowerShell 语法:
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"

# 如果是 CMD (命令提示符)，请使用:
# set CMAKE_ARGS=-DGGML_CUDA=on
# set FORCE_CMAKE=1
```

#### 4. 安装 llama-cpp-python (手动编译版)
**不要**直接使用 `pip install llama-cpp-python`，因为那样会下载预编译包（不支持 Qwen3.5 MoE）。
您需要先克隆源码、替换 `vendor` 目录、修补 API，然后再安装。

```bash
# 1. 克隆仓库
git clone https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python

# 2. [重要] 此时请执行您之前的操作：
#    - 下载最新 llama.cpp 源码并替换 vendor/llama.cpp
#    - 修改 llama_cpp/llama.py 注释掉废弃函数

# 3. 执行本地安装 (会触发编译)
pip install . -v
```
*注：`-v` 参数会显示详细编译日志，方便排查错误。编译过程可能需要 5-10 分钟。*

#### 5. 安装其他运行依赖
模型运行还需要一些基础库。

```bash
pip install numpy torch --index-url https://download.pytorch.org/whl/cu121
# 注意：虽然我们有 CUDA 13.1，但 PyTorch 通常兼容 cu121 即可，或者只安装 numpy 也足够跑 llama-cpp
pip install numpy
```

#### 6. 验证环境
安装完成后，运行以下命令验证环境是否就绪：

```bash
python -c "import llama_cpp; print('✅ llama-cpp 导入成功'); print(f'版本：{llama_cpp.__version__}')"
```

---

> 为了避免依赖冲突，我使用 **Conda** 构建了名为 `qwen35` 的独立虚拟环境，并严格锁定 **Python 3.11** 版本。针对 Windows 平台编译难的问题，我通过 `conda-forge` 渠道安装了 **CMake** 和 **Ninja** 构建工具，解决了 `build.ninja not found` 的常见错误。
>
> 更重要的是，为了适配 RTX 5070 的 Blackwell 架构，我在编译前显式设置了 `CMAKE_ARGS="-DGGML_CUDA=on"` 环境变量，强制开启 CUDA 加速支持。这一系列标准化的环境配置流程，为后续复杂的源码编译和 API 修补奠定了坚实的基础，确保了整个开发环境的纯净与可复现性。

---


### 💡 给后来者的 `environment.yml` 文件 (可选)
创建一个 `environment.yml` 文件，一键还原我的环境：

```yaml
name: qwen35
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - cmake
  - ninja
  - git
  - pip
  - pip:
      - numpy
      # 注意：llama-cpp-python 需要手动克隆源码并编译，不能直接写在 pip 列表里
```
*(用户只需运行 `conda env create -f environment.yml` 即可)*



💡 核心解决方案 (The Golden Path)
1. 手动注入架构支持
由于官方包不支持 Qwen3.5 MoE，必须手动编译：
克隆 llama-cpp-python 仓库。
下载最新 llama.cpp Master 分支源码。
关键步骤：替换项目中的 vendor/llama.cpp 目录。
执行 pip install . 进行本地编译。
2. “外科手术式”API 修补
编译后导入报错 function not found？这是因为 llama.cpp 移除了旧接口。
操作：打开 llama_cpp/llama.py (或 .cpp 绑定文件)。
动作：搜索并注释掉以下废弃函数调用（约 60+ 处）：
llama_get_kv_self 系列
llama_adapter_lora 系列
llama_sampler_init_* 系列
结果：绕过 API 不兼容，成功导入库。
3. 黄金参数配置 (稳定性奇点)
经过数十次梯度测试，发现 n_gpu_layers=1 是唯一稳定点。
n_gpu_layers >= 2：导致数值溢出，输出全问号 ?????? 或英文胡话。
n_gpu_layers = 0：纯 CPU，稳定但速度慢 (~3 t/s)。
n_gpu_layers = 1：完美平衡，速度 ~4.5-5.5 t/s，无乱码，无崩溃。


⚠️ 避坑指南：两大致命陷阱
陷阱 1：CUDA Internal Error / 静默卡死
现象：运行时报 cublasGemmStridedBatchedEx ... internal operation failed 或程序假死。
原因：Blackwell 架构驱动不稳定 + Flash Attention 兼容性差。
✅ 解法：启动时强制设置 flash_attn=False。
陷阱 2：输出全是问号 ?????? 或乱码
现象：模型加载成功，但生成内容全是问号。
原因：
GPU 层数过高导致数值错误（最常见）。
Windows 控制台编码问题。
✅ 解法：
将 n_gpu_layers 降为 1。
代码中执行 os.system('chcp 65001 > nul')。
强烈建议：将输出写入文件 (answer.txt)，用记事本查看，绕过控制台限制。
陷阱 3：模型说胡话（英文角色扮演）
现象：输入“你好”，模型回答 "user, I feel sad..."。
原因：缺少 Chat Template，模型在进行文本续写而非对话。




快速开始：一键运行脚本
脚本应该名为run_qwen35.py，我放在仓库文件里。
把zip解压放到一个你方便的地方。
不会运行的把这个阅读文档丢给ai让它教你。
另外两个后缀为py的都是测试用的，你应该用不着，如果你是开发者，感兴趣可以看一看





📊 性能数据实测报告
本项目的性能测试基于 AMD Ryzen 9 9955HX 处理器与 NVIDIA RTX 5070 Laptop (8GB VRAM) 组合。测试模型为 Qwen3.5-35B-A3B-Q4_K_S.gguf (4-bit 量化)。
表格
测试指标	实测数据	备注
生成速度	4.5 - 5.5 Tokens/s	得益于 DDR5 双通道内存带宽与 GPU 单层加速的协同
显存占用	7.2 GB - 7.6 GB	严格控制在 8GB 物理显存内，未发生 Swap 交换
内存占用	~21.5 GB	模型权重主要驻留于系统内存
首字延迟	~3.8 秒	首次推理需加载部分权重至缓存
稳定性测试	连续运行 2 小时无崩溃	在复杂多轮对话场景下，未出现 CUDA Error 或数值溢出
温度表现	GPU 72°C / CPU 85°C	笔记本满载正常温度范围，风扇全速运转


🙏 致谢与贡献
本项目得以实现，离不开开源社区的无私奉献：
阿里巴巴通义千问团队：发布了强大的 Qwen3.5 系列模型，并开源了高质量的权重。
llama.cpp 社区：提供了高效的 C++ 推理后端，使得在消费级硬件上运行大模型成为可能。
GitHub 开发者们：各种关于 MoE 架构支持和 Windows 兼容性修复的 Issue 讨论为本项目提供了关键线索。


项目愿景：
本指南旨在证明，显存大小不应成为探索大模型的绝对门槛。即使只有 8GB 显存的笔记本电脑，通过合理的架构理解、参数调优和代码修补，依然可以流畅运行参数量巨大的前沿模型。希望这份文档能帮助更多资源受限的开发者、学生和研究者低门槛地体验本地大模型的魅力。
欢迎提交 Issue 或 Pull Request，共同完善在黑石架构（Blackwell）上的部署细节！
© 2026 Qwen3.5-LowVRAM-Project. Licensed under MIT.
