<h1 align="center">ACE-Step</h1>
<h1 align="center">A Step Towards Music Generation Foundation Model</h1>
<p align="center">
    <a href="https://ace-step.github.io/">Project</a> |
    <a href="https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B">Hugging Face</a> |
    <a href="https://modelscope.cn/models/ACE-Step/ACE-Step-v1-3.5B">ModelScope</a> |
    <a href="https://huggingface.co/spaces/ACE-Step/ACE-Step">Space Demo</a> |
    <a href="https://discord.gg/PeWDxrkdj7">Discord</a> 
</p>
<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README_CN.md"><img alt="简体中文版自述文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="./README_JA.md"><img alt="日本語のREADME" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
</p>

<p align="center">
    <img src="./assets/orgnization_logos.png" width="100%" alt="StepFun 标志">
</p>

## 目录

- [✨ 功能特性](#-功能特性)
- [📦 安装](#-安装)
- [🚀 使用](#-使用)
- [📱 用户界面指南](#-用户界面指南)
- [🔨 训练](#-训练)

## 📝 摘要

我们推出了 ACE-Step，这是一款新颖的开源音乐生成基础模型，它克服了现有方法的关键局限性，并通过整体架构设计实现了最先进的性能。当前的方法在生成速度、音乐连贯性和可控性之间存在固有的权衡。例如，基于LLM的模型（如Yue、SongGen）在歌词对齐方面表现出色，但推理速度慢且存在结构失真。而扩散模型（如DiffRhythm）虽然能实现更快的合成，但通常缺乏长程结构连贯性。

ACE-Step 通过将基于扩散的生成与 Sana 的深度压缩自动编码器 (DCAE) 和轻量级线性 Transformer 相结合，弥合了这一差距。它还在训练过程中利用 MERT 和 m-hubert 对齐语义表示 (REPA)，从而实现快速收敛。因此，我们的模型在 A100 GPU 上仅需 20 秒即可合成长达 4 分钟的音乐——比基于 LLM 的基线模型快 15 倍——同时在旋律、和声和节奏指标上实现了卓越的音乐连贯性和歌词对齐。此外，ACE-Step 保留了细致的声学细节，从而实现了高级控制机制，如声音克隆、歌词编辑、混音和音轨生成（例如，歌词到人声、歌唱到伴奏）。

我们的愿景并非构建又一个端到端的文本到音乐的流水线，而是为音乐人工智能建立一个基础模型：一个快速、通用、高效且灵活的架构，使其易于在其之上训练子任务。这为开发能无缝集成到音乐艺术家、制作人和内容创作者创作流程中的强大工具铺平了道路。简而言之，我们旨在打造音乐领域的 Stable Diffusion 时刻。

## 📢 新闻与更新

- 🚀 **2025.05.08:** [ComfyUI_ACE-Step](https://t.co/GeRSTrIvn0) 节点现已可用！在 ComfyUI 中探索 ACE-Step 的强大功能。🎉
![图片](https://github.com/user-attachments/assets/0a13d90a-9086-47ee-abab-976bad20fa7c)

- 🚀 2025.05.06: 开源演示代码和模型

## ✨ 功能特性

<p align="center">
    <img src="./assets/application_map.png" width="100%" alt="ACE-Step 应用图">
</p>

### 🎯 基线质量

#### 🌈 多样风格与流派

- 🎸 支持所有主流音乐风格，描述格式多样，包括短标签、描述性文本或使用场景
- 🎷 能够生成不同流派的音乐，并配备合适的乐器和风格

#### 🌍 多语言支持

- 🗣️ 支持19种语言，表现较好的前10种语言包括：
  - 🇺🇸 英语, 🇨🇳 中文, 🇷🇺 俄语, 🇪🇸 西班牙语, 🇯🇵 日语, 🇩🇪 德语, 🇫🇷 法语, 🇵🇹 葡萄牙语, 🇮🇹 意大利语, 🇰🇷 韩语
- ⚠️ 由于数据不平衡，不太常见的语言可能表现不佳

#### 🎻 乐器风格

- 🎹 支持各种不同流派和风格的器乐生成
- 🎺 能够为每种乐器制作逼真的乐器音轨，并具有适当的音色和表现力
- 🎼 能够生成包含多种乐器的复杂编曲，同时保持音乐连贯性

#### 🎤 人声技巧

- 🎙️ 能够高质量地呈现各种人声风格和技巧
- 🗣️ 支持不同的声乐表达，包括各种歌唱技巧和风格

### 🎛️ 可控性

#### 🔄 变奏生成

- ⚙️ 通过免训练、推理时优化技术实现
- 🌊 流匹配模型生成初始噪声，然后使用 trigFlow 的噪声公式添加额外的高斯噪声
- 🎚️ 可调节原始初始噪声和新高斯噪声之间的混合比例，以控制变奏程度

#### 🎨 局部重绘 (Repainting)

- 🖌️ 通过向目标音频输入添加噪声并在ODE（常微分方程）过程中应用掩码约束来实现
- 🔍 当输入条件与原始生成不同时，可以仅修改特定方面，同时保留其余部分
- 🔀 可以与变奏生成技术结合，在风格、歌词或人声方面创建局部变化

#### ✏️ 歌词编辑

- 💡 创新性地应用流编辑 (flow-edit) 技术，实现局部歌词修改，同时保留旋律、人声和伴奏
- 🔄 适用于生成内容和上传的音频，极大地增强了创作可能性
- ℹ️ 当前限制：一次只能修改少量歌词片段以避免失真，但可以顺序应用多次编辑

### 🚀 应用

#### 🎤 歌词到人声 (Lyric2Vocal) (LoRA)

- 🔊 基于在纯人声数据上微调的 LoRA 模型，允许直接从歌词生成人声样本
- 🛠️ 提供众多实际应用，如人声小样、引导音轨、歌曲创作辅助和人声编排实验
- ⏱️ 提供一种快速测试歌词演唱效果的方法，帮助歌曲创作者更快地迭代

#### 📝 文本到采样 (Text2Samples) (LoRA)

- 🎛️ 类似于歌词到人声，但在纯乐器和采样数据上进行微调
- 🎵 能够从文本描述生成概念性的音乐制作采样
- 🧰 可用于快速创建乐器循环、音效和用于制作的音乐元素

### 🔮 即将推出

#### 🎤 RapMachine

- 🔥 在纯说唱数据上进行微调，以创建一个专门从事说唱生成的 AI 系统
- 🏆 预期能力包括 AI 说唱对战和通过说唱进行叙事表达
- 📚 说唱具有出色的叙事和表达能力，提供了非凡的应用潜力

#### 🎛️ StemGen

- 🎚️ 一个在多轨数据上训练的 controlnet-lora 模型，用于生成单个乐器分轨
- 🎯 以参考音轨和指定乐器（或乐器参考音频）作为输入
- 🎹 输出与参考音轨互补的乐器分轨，例如为长笛旋律创作钢琴伴奏或为领奏吉他添加爵士鼓

#### 🎤 歌唱到伴奏 (Singing2Accompaniment)

- 🔄 StemGen 的逆过程，从单个人声音轨生成混合的母带音轨
- 🎵 以人声音轨和指定风格作为输入，生成完整的人声伴奏
- 🎸 创建与输入人声互补的完整乐器背景，可以轻松地为任何录制的人声添加专业水准的伴奏

## 📋 路线图

- [x] 发布训练代码 🔥
- [x] 发布 LoRA 训练代码 🔥
- [ ] 发布 RapMachine LoRA 🎤
- [ ] 发布 ControlNet 训练代码 🔥
- [ ] 发布 Singing2Accompaniment ControlNet 🎮
- [ ] 发布评估性能和技术报告 📄

## 🖥️ 硬件性能

我们评估了 ACE-Step 在不同硬件配置下的性能，得到以下吞吐量结果：

| 设备          | 实时率 (27步) | 渲染1分钟音频所需时间 (27步) | 实时率 (60步) | 渲染1分钟音频所需时间 (60步) |
| --------------- | -------------- | ------------------------------------- | -------------- | ------------------------------------- |
| NVIDIA RTX 4090 | 34.48 ×        | 1.74 秒                               | 15.63 ×        | 3.84 秒                               |
| NVIDIA A100     | 27.27 ×        | 2.20 秒                               | 12.27 ×        | 4.89 秒                               |
| NVIDIA RTX 3090 | 12.76 ×        | 4.70 秒                               | 6.48 ×         | 9.26 秒                               |
| MacBook M2 Max  | 2.27 ×         | 26.43 秒                              | 1.03 ×         | 58.25 秒                              |

我们使用 RTF (Real-Time Factor, 实时率) 来衡量 ACE-Step 的性能。数值越高表示生成速度越快。27.27x 表示生成1分钟的音乐需要2.2秒 (60/27.27)。性能是在单个 GPU、批处理大小为1、27步的条件下测量的。

## 📦 安装

### 1. 克隆仓库
首先，将 ACE-Step 仓库克隆到您的本地计算机，并进入项目目录：
```bash
git clone https://github.com/ace-step/ACE-Step.git
cd ACE-Step
```

### 2. 先决条件
确保您已安装以下软件：

* `Python`: 推荐使用 3.10 或更高版本。您可以从 [python.org](https://www.python.org/) 下载。
* `Conda` 或 `venv`: 用于创建虚拟环境（推荐使用 Conda）。

### 3. 设置虚拟环境

强烈建议使用虚拟环境来管理项目依赖项并避免冲突。选择以下方法之一：

#### 选项 A：使用 Conda

1.  **创建环境**，命名为 `ace_step`，使用 Python 3.10：
    ```bash
    conda create -n ace_step python=3.10 -y
    ```

2.  **激活环境：**
    ```bash
    conda activate ace_step
    ```

#### 选项 B：使用 venv

1.  **导航到克隆的 ACE-Step 目录。**

2.  **创建虚拟环境** (通常命名为 `venv`)：
    ```bash
    python -m venv venv
    ```

3.  **激活环境：**
    * **Windows (cmd.exe):**
        ```bash
        venv\Scripts\activate.bat
        ```
    * **Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
        *（如果遇到执行策略错误，您可能需要先运行 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`）*
    * **Linux / macOS (bash/zsh):**
        ```bash
        source venv/bin/activate
        ```

### 4. 安装依赖
虚拟环境激活后：
**a.** (仅限 Windows) 如果您在 Windows 上并计划使用 NVIDIA GPU，请先安装支持 CUDA 的 PyTorch：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
（如果您有不同的 CUDA 版本，请调整 cu126。有关其他 PyTorch 安装选项，请参阅 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)）。

**b.** 安装 ACE-Step 及其核心依赖项：
```bash
pip install -e .
```

ACE-Step 应用程序现已安装。GUI 可在 Windows、macOS 和 Linux 上运行。有关如何运行它的说明，请参阅 [使用](#-使用) 部分。

## 🚀 使用

![演示界面](assets/demo_interface.png)

### 🔍 基本用法

```bash
acestep --port 7865
```

### ⚙️ 高级用法

```bash
acestep --checkpoint_path /path/to/checkpoint --port 7865 --device_id 0 --share true --bf16 true
```

如果您使用的是 macOS，请使用 `--bf16 false` 以避免错误。

#### 🔍 API 用法
如果您打算将 ACE-Step 作为库集成到您自己的 Python 项目中，可以使用以下 pip 命令直接从 GitHub 安装最新版本。

**通过 pip 直接安装：**

1.  **确保已安装 Git：** 此方法要求您的系统上已安装 Git，并且在系统的 PATH 中可访问。
2.  **执行安装命令：**
    ```bash
    pip install git+https://github.com/ace-step/ACE-Step.git
    ```
    建议在虚拟环境中使用此命令，以避免与其他软件包发生冲突。

#### 🛠️ 命令行参数

- `--checkpoint_path`: 模型检查点路径 (默认：自动下载)
- `--server_name`: Gradio 服务器绑定的 IP 地址或主机名 (默认：'127.0.0.1')。使用 '0.0.0.0' 使其可从网络上的其他设备访问。
- `--port`: Gradio 服务器运行的端口 (默认：7865)
- `--device_id`: 要使用的 GPU 设备 ID (默认：0)
- `--share`: 启用 Gradio 共享链接 (默认：False)
- `--bf16`: 使用 bfloat16 精度以加快推理速度 (默认：True)
- `--torch_compile`: 使用 `torch.compile()` 优化模型，加快推理速度 (默认：False)。**Windows 不支持**

## 📱 用户界面指南

ACE-Step 界面提供了几个选项卡，用于不同的音乐生成和编辑任务：

### 📝 文本到音乐 (Text2Music) 选项卡

1. **📋 输入字段**:
   - **🏷️ 标签 (Tags)**: 输入描述性标签、流派或场景描述，用逗号分隔
   - **📜 歌词 (Lyrics)**: 输入歌词，并使用 [verse], [chorus], [bridge] 等结构标签
   - **⏱️ 音频时长 (Audio Duration)**: 设置生成音频的目标时长 (-1 表示随机)

2. **⚙️ 设置 (Settings)**:
   - **🔧 基本设置 (Basic Settings)**: 调整推理步数、引导强度 (guidance scale) 和种子 (seeds)
   - **🔬 高级设置 (Advanced Settings)**: 微调调度器类型、CFG 类型、ERG 设置等

3. **🚀 生成 (Generation)**: 点击 "Generate" 根据您的输入创建音乐

### 🔄 重录 (Retake) 选项卡

- 🎲 使用不同的种子重新生成音乐，并带有轻微变化
- 🎚️ 调整变异程度 (variance) 以控制重录与原始版本的差异程度

### 🎨 局部重绘 (Repainting) 选项卡

- 🖌️ 选择性地重新生成音乐的特定部分
- ⏱️ 指定要重绘部分的开始和结束时间
- 🔍 选择源音频 (文本到音乐的输出、上次重绘结果或上传的音频)

### ✏️ 编辑 (Edit) 选项卡

- 🔄 通过更改标签或歌词来修改现有音乐
- 🎛️ 选择 "only_lyrics" 模式 (保留旋律) 或 "remix" 模式 (改变旋律)
- 🎚️ 调整编辑参数以控制保留原始内容的程度

### 📏 扩展 (Extend) 选项卡

- ➕ 在现有乐曲的开头或结尾添加音乐
- 📐 指定左侧和右侧扩展长度
- 🔍 选择要扩展的源音频

## 📂 示例

`examples/input_params` 目录包含可用作生成音乐参考的示例输入参数。

## 🏗️ 架构

<p align="center">
    <img src="./assets/ACE-Step_framework.png" width="100%" alt="ACE-Step 框架图">
</p>

## 🔨 训练

### 先决条件
1. 按照安装部分的说明准备环境。

2. 如果您计划训练 LoRA 模型，请安装 PEFT 库：
   ```bash
   pip install peft
   ```

3. 以 Huggingface 格式准备您的数据集 ([Huggingface Datasets 文档](https://huggingface.co/docs/datasets/index))。数据集应包含以下字段：
   - `keys`: 每个音频样本的唯一标识符
   - `filename`: 音频文件的路径
   - `tags`: 描述性标签列表 (例如：`["pop", "rock"]`)
   - `norm_lyrics`: 规范化的歌词文本
   - 可选字段：
     - `speaker_emb_path`:说话人嵌入文件的路径 (如果不可用，则使用空字符串)
     - `recaption`: 各种格式的附加标签描述

示例数据集条目：
```json
{
	"keys": "1ce52937-cd1d-456f-967d-0f1072fcbb58",
	"filename": "data/audio/1ce52937-cd1d-456f-967d-0f1072fcbb58.wav",
	"tags": ["流行", "原声", "情歌", "浪漫", "情感"],
	"speaker_emb_path": "",
	"norm_lyrics": "我爱你，我爱你，我爱你",
	"recaption": {
		"simplified": "流行",
		"expanded": "流行, 原声, 情歌, 浪漫, 情感",
		"descriptive": "声音轻柔，如同静夜中的微风。它舒缓而充满渴望。",
		"use_cases": "适用于浪漫电影的背景音乐或私密时刻。",
		"analysis": "流行, 情歌, 钢琴, 吉他, 慢节奏, 浪漫, 情感"
	}
}
```

### 训练参数

#### 通用参数
- `--dataset_path`: Huggingface 数据集的路径 (必需)
- `--checkpoint_dir`: 包含基础模型检查点的目录
- `--learning_rate`: 训练学习率 (默认：1e-4)
- `--max_steps`: 最大训练步数 (默认：2000000)
- `--precision`: 训练精度，例如："bf16-mixed" (默认) 或 "fp32"
- `--devices`: 使用的 GPU 数量 (默认：1)
- `--num_nodes`: 使用的计算节点数量 (默认：1)
- `--accumulate_grad_batches`: 梯度累积步数 (默认：1)
- `--num_workers`: 数据加载工作线程数 (默认：8)
- `--every_n_train_steps`: 检查点保存频率 (默认：2000)
- `--every_plot_step`: 生成评估样本的频率 (默认：2000)
- `--exp_name`: 用于日志记录的实验名称 (默认："text2music_train_test")
- `--logger_dir`: 保存日志的目录 (默认："./exps/logs/")

#### 基础模型训练
使用以下命令训练基础模型：
```bash
python trainer.py --dataset_path "path/to/your/dataset" --checkpoint_dir "path/to/base/checkpoint" --exp_name "your_experiment_name"
```

#### LoRA 训练
对于 LoRA 训练，您需要提供一个 LoRA 配置文件：
```bash
python trainer.py --dataset_path "path/to/your/dataset" --checkpoint_dir "path/to/base/checkpoint" --lora_config_path "path/to/lora_config.json" --exp_name "your_lora_experiment"
```

示例 LoRA 配置文件 (lora_config.json):
```json
{
	"r": 16,
	"lora_alpha": 32,
	"target_modules": [
		"speaker_embedder",
		"linear_q",
		"linear_k",
		"linear_v",
		"to_q",
		"to_k",
		"to_v",
		"to_out.0"
	]
}
```

### 高级训练选项
- `--shift`: 流匹配 (Flow matching) 位移参数 (默认：3.0)
- `--gradient_clip_val`: 梯度裁剪值 (默认：0.5)
- `--gradient_clip_algorithm`: 梯度裁剪算法 (默认："norm")
- `--reload_dataloaders_every_n_epochs`: 重新加载数据加载器的频率 (默认：1)
- `--val_check_interval`: 验证检查间隔 (默认：None)

## 📜 许可与免责声明

本项目根据 [Apache License 2.0](./LICENSE) 获得许可。

ACE-Step 能够生成各种流派的原创音乐，应用于创意制作、教育和娱乐领域。虽然旨在支持积极和艺术性的用例，但我们承认潜在的风险，例如由于风格相似性导致的无意版权侵犯、文化元素的不当融合以及滥用于生成有害内容。为确保负责任地使用，我们鼓励用户验证生成作品的原创性，明确披露 AI 的参与，并在改编受保护的风格或材料时获得适当的许可。使用 ACE-Step 即表示您同意遵守这些原则，尊重艺术完整性、文化多样性和法律合规性。作者不对模型的任何滥用负责，包括但不限于侵犯版权、文化不敏感或生成有害内容。

🔔 重要声明
ACE-Step 项目的唯一官方网站是我们的 GitHub Pages 网站。
我们不运营任何其他网站。
🚫 虚假域名包括但不限于：
ac\*\*p.com, a\*\*p.org, a\*\*\*c.org
⚠️ 请务必谨慎。不要访问、信任或在任何这些网站上付款。

## 🙏 致谢

本项目由 ACE Studio 和 StepFun 共同领导。

## 📖 引用

如果您发现此项目对您的研究有用，请考虑引用：

```BibTeX
@misc{gong2025acestep,
	title={ACE-Step: A Step Towards Music Generation Foundation Model},
	author={Junmin Gong, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
	howpublished={\url{https://github.com/ace-step/ACE-Step}},
	year={2025},
	note={GitHub 仓库}
}
```