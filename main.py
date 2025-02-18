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
        result = model.transcribe(sound_path)
        return result["text"]

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
    ) -> str:
        sound_path = self.__download_sound(sound_file_name)
        transcribed_text = self.__convert_to_text(sound_path, stt_model)

        try:
            with open(output_file, "w") as file:
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
