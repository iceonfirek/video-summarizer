import warnings
import whisper
import ollama
import yt_dlp
import sys
import os


# 在文件开头添加警告过滤
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


class VideoSummarizer:
    def __init__(self, url_or_path: str, is_local_file: bool = False):
        self.url_or_path = url_or_path
        self.is_local_file = is_local_file

    def __download_sound(self, file_name="sound") -> str:
        if self.is_local_file:
            if not os.path.exists(self.url_or_path):
                print(f"错误：文件不存在：{self.url_or_path}")
                sys.exit(1)
            return self.url_or_path

        temp_path = "./temp/"
        os.makedirs(temp_path, exist_ok=True)
        options = {
            "format": "bestaudio/best",
            "outtmpl": f"{temp_path}{file_name}.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "cookiesfrombrowser": ("safari",),
        }

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(self.url_or_path, download=True)
                output_file = ydl.prepare_filename(info).replace(info["ext"], "mp3")
                return output_file
        except Exception as e:
            print(f"下载错误: {e}")
            print("\n请确保：")
            print("1. 已经在浏览器中登录了YouTube")
            print("2. 使用的是支持的浏览器（Chrome/Firefox/Safari/Edge）")
            print("3. URL是有效的YouTube视频链接")
            sys.exit(1)

    def __convert_to_text(self, sound_path: str, model_name="small") -> str:
        model = whisper.load_model(model_name, device="cpu")
        
        # 使用支持的参数进行转录
        result = model.transcribe(
            sound_path,
            initial_prompt="请使用自然段落分隔文本。在主题改变时开始新的段落。",
            language="en"  # 可以根据需要更改语言
        )
        
        # 处理转录文本，添加段落分隔
        text = result["text"]
        
        # 基于句号和主题变化添加段落
        sentences = text.split(". ")
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            # 确保句子结尾有句号
            if not sentence.endswith("."):
                sentence += "."
                
            current_paragraph.append(sentence)
            
            # 当积累了足够的句子或遇到主题转换标记时，创建新段落
            if len(current_paragraph) >= 3 or any(marker in sentence.lower() for marker in [
                "chapter", "section", "part",  # 章节标记
                "however", "nevertheless", "moreover",  # 转折词
                "firstly", "secondly", "finally",  # 序列词
                "in conclusion", "to summarize",  # 总结标记
            ]):
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
        
        # 添加最后一个段落
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        # 使用双换行符连接段落
        formatted_text = "\n\n".join(paragraphs)
        
        return formatted_text

    def __ai_summarizer(self, text: str, model_name="deepseek-r1:1.5b") -> str:
        input_text = f"""你是一个擅长总结文字的助手，善于保留重要信息并以清晰的格式呈现。请帮我总结以下文字。要求：
1. 保持重要信息完整
2. 结构清晰，便于阅读
3. 不要过于简短
4. 可以用要点或分段的形式展示

文字内容：
{text}"""

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "",
                    },
                    {
                        "role": "user",
                        "content": input_text,
                    },
                ],
            )

            return response["message"]["content"]
        except Exception as e:
            print(f"AI总结错误: {e}")
            sys.exit(1)

    def summarize(
        self,
        sound_file_name="sound",
        stt_model="small",
        model_name="deepseek-r1:1.5b",
        output_file="output.txt",
    ) -> str:
        sound_path = self.__download_sound(sound_file_name)
        original_text = self.__convert_to_text(sound_path, stt_model)
        summarized_text = self.__ai_summarizer(original_text, model_name)

        try:
            with open(output_file, "w") as file:
                file.write(summarized_text)
                print(f"Summary saved to {output_file}")
        except Exception as e:
            print(f"Saving Error: {e}")

        return summarized_text

    def transcribe(
        self,
        sound_file_name="sound",
        stt_model="small",
        output_file="transcript.txt",
        max_chars_per_file=100000  # 每个文件的最大字符数
    ) -> str:
        sound_path = self.__download_sound(sound_file_name)
        transcribed_text = self.__convert_to_text(sound_path, stt_model)
        
        # 如果文本超过最大长度，分割成多个文件
        if len(transcribed_text) > max_chars_per_file:
            parts = []
            current_pos = 0
            part_num = 1
            
            while current_pos < len(transcribed_text):
                # 找到合适的分割点（段落结束）
                end_pos = current_pos + max_chars_per_file
                if end_pos < len(transcribed_text):
                    # 向后查找段落结束
                    end_pos = transcribed_text.rfind("\n\n", current_pos, end_pos)
                    if end_pos == -1:  # 如果找不到段落结束，就找句号
                        end_pos = transcribed_text.rfind(". ", current_pos, current_pos + max_chars_per_file) + 1
                else:
                    end_pos = len(transcribed_text)
                
                # 提取当前部分的文本
                part_text = transcribed_text[current_pos:end_pos].strip()
                
                # 保存到文件
                part_file = f"{output_file.rsplit('.', 1)[0]}_part{part_num}.txt"
                try:
                    with open(part_file, "w", encoding="utf-8") as file:
                        file.write(part_text)
                    print(f"转录文本已保存到 {part_file}")
                    parts.append(part_file)
                except Exception as e:
                    print(f"保存错误: {e}")
                
                current_pos = end_pos
                part_num += 1
            
            print(f"\n转录文本已分割成 {len(parts)} 个文件")
            return transcribed_text
        
        # 如果文本没有超过最大长度，保存到单个文件
        try:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(transcribed_text)
            print(f"转录文本已保存到 {output_file}")
        except Exception as e:
            print(f"保存错误: {e}")

        return transcribed_text


def get_input() -> tuple[str, bool, str]:
    if len(sys.argv) > 1:
        if sys.argv[1] == "--file":
            if len(sys.argv) < 3:
                print("错误：使用 --file 选项时需要提供文件路径")
                sys.exit(1)
            mode = "--transcribe" in sys.argv
            return sys.argv[2], True, "transcribe" if mode else "summarize"
        mode = "--transcribe" in sys.argv
        return sys.argv[1], False, "transcribe" if mode else "summarize"
    
    mode = input("选择模式 (1: YouTube链接, 2: 本地音频文件): ")
    if mode not in ["1", "2"]:
        print("错误：无效的模式选择")
        sys.exit(1)
        
    task = input("选择任务 (1: 转录+总结, 2: 仅转录): ")
    if task not in ["1", "2"]:
        print("错误：无效的任务选择")
        sys.exit(1)

    if mode == "1":
        return input("请输入YouTube URL: "), False, "summarize" if task == "1" else "transcribe"
    else:
        return input("请输入音频文件路径: "), True, "summarize" if task == "1" else "transcribe"


if __name__ == "__main__":
    input_path, is_local_file, task = get_input()
    summarizer = VideoSummarizer(input_path, is_local_file)
    
    if task == "transcribe":
        summarizer.transcribe()
    else:
        summarizer.summarize()
