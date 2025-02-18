# Video Summarizer

一个基于Python的工具，可以从YouTube视频或本地音频文件中提取音频，使用Whisper进行转录，并可选择使用通过Ollama本地运行的DeepSeek进行摘要。

## 功能

- 支持YouTube视频和本地音频文件
- 使用`yt-dlp`从YouTube视频下载音频
- 使用OpenAI的`whisper`将音频转换为文本
- 可选择使用`ollama`和DeepSeek模型对转录文本进行摘要
- 支持仅转录和转录+总结两种模式
- 将转录文本和摘要分别保存到文件

## 安装

### 1. 环境要求

- Python 3.9+
- FFmpeg（用于音频处理）

### 2. 安装FFmpeg

#### macOS:
```bash
brew install ffmpeg
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Windows:
从[FFmpeg官网](https://ffmpeg.org/download.html)下载并添加到系统PATH。

### 3. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

### 4. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 5. 安装Ollama和DeepSeek模型

1. 从[Ollama官网](https://ollama.com/)安装Ollama
2. 下载DeepSeek模型：
```bash
ollama pull deepseek-r1:1.5b
```

## 使用方法

### 命令行方式

1. 处理YouTube视频：
```bash
# 转录+总结
python main.py "https://www.youtube.com/watch?v=..."

# 仅转录
python main.py "https://www.youtube.com/watch?v=..." --transcribe
```

2. 处理本地音频文件：
```bash
# 转录+总结
python main.py --file "path/to/audio.mp3"

# 仅转录
python main.py --file "path/to/audio.mp3" --transcribe
```

### 交互式方式

运行程序并按提示操作：
```bash
python main.py
```

程序会提示：
1. 选择输入模式（YouTube或本地文件）
2. 选择任务类型（转录+总结或仅转录）
3. 输入URL或文件路径

### 输出文件

- 转录文本保存在 `transcript.txt`
- 总结文本保存在 `output.txt`

## 自定义选项

- 修改Whisper模型：更改`stt_model`参数（默认：`small`）
- 使用不同的Ollama模型：在`summarize()`中设置`model_name`
- 更改输出文件：指定`output_file`参数

## Example Output

```bash
Summary saved to output.txt
```

##


