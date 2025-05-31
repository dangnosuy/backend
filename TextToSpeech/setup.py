import subprocess
import os
import sys
from huggingface_hub import snapshot_download

def run_command(command):
    """Chạy lệnh và hiển thị kết quả."""
    print(f"> Đang chạy: {command}")
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"❌ Lỗi khi chạy: {command}")
        sys.exit(result.returncode)

print(" > Bỏ qua bước đổi múi giờ trên Windows.")  # Vì bước này chỉ áp dụng trên Linux

# 1. Clone repository TTS
print(" > Clone repo TTS...")
if os.path.exists("TTS"):
    subprocess.run("rmdir /s /q TTS", shell=True)
run_command("git clone --branch add-vietnamese-xtts https://github.com/thinhlpg/TTS.git")

# 2. Cài đặt thư viện
print(" > Cài đặt các thư viện cần thiết...")
libs = [
    "deepspeed",
    "vinorm==2.0.7",
    "cutlet",
    "unidic==1.1.0",
    "underthesea",
    "gradio==4.35",
    "deepfilternet==0.5.6"
]
for lib in libs:
    run_command(f"pip install {lib}")

# 3. Cài đặt repo TTS local (editable mode)
run_command("pip install -e TTS")

# 4. Tải dữ liệu từ unidic
print(" > Tải dữ liệu unidic...")
run_command("python -m unidic download")

# 5. Tải mô hình từ Hugging Face
print(" > Tải mô hình từ HuggingFace...")
snapshot_download(
    repo_id="thinhlpg/viXTTS",
    repo_type="model",
    local_dir="model"
)

print("\n✅ Hoàn tất cài đặt! Bạn có thể chạy tiếp bước chuyển văn bản thành giọng nói.")
