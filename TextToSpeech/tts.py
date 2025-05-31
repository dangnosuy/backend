# The inference code is adopted from https://github.com/coqui-ai/TTS/blob/dev/TTS/demos/xtts_ft_demo/xtts_demo.py
# @title 2. 🤗 **Sử dụng**
# @markdown 👈 Nhấn để chạy.
# @markdown Nếu gặp lỗi thì cũng nhấn nút này nhé!

# @markdown Lần đầu chạy sẽ hơi lâu, bạn chờ tí nhé!

# @markdown Kết quả sẽ được lưu vào `/content/output`
# @markdown Chọn ngôn ngữ:
#language = input("Nhập loại ngôn ngữ(Tiếng Việt, Tiếng Anh, ...): ") # @param ["Tiếng Việt", "Tiếng Anh","Tiếng Tây Ban Nha", "Tiếng Pháp","Tiếng Đức","Tiếng Ý", "Tiếng Bồ Đào Nha", "Tiếng Ba Lan", "Tiếng Thổ Nhĩ Kỳ", "Tiếng Nga", "Tiếng Hà Lan", "Tiếng Séc", "Tiếng Ả Rập", "Tiếng Trung (giản thể)", "Tiếng Nhật", "Tiếng Hungary", "Tiếng Hàn", "Tiếng Hindi"]
# @markdown Văn bản để đọc. Độ dài tối thiểu mỗi câu nên từ 10 từ để đặt kết quả tốt nhất.
#input_text =input("Nhập văn bản: ") # @param {type:"string"}
# @markdown Chọn giọng mẫu:
#reference_audio = "model/st.mp3" # @param [ "model/user_sample.wav",  "model/vi_sample.wav",  "model/samples/nam-calm.wav",  "model/samples/nam-cham.wav",  "model/samples/nam-nhanh.wav",  "model/samples/nam-truyen-cam.wav",  "model/samples/nu-calm.wav",  "model/samples/nu-cham.wav",  "model/samples/nu-luu-loat.wav",  "model/samples/nu-nhan-nha.wav",  "model/samples/nu-nhe-nhang.wav"]
import argparse
parser = argparse.ArgumentParser(description="Text-to-Speech với XTTS")
parser.add_argument('-language', type=str, required=True, help="Ngôn ngữ (Tiếng Việt, Tiếng Anh, ...)")
parser.add_argument('-input', type=str, required=True, help="Văn bản cần đọc")
parser.add_argument('-reference', type=str, required=True, default="model/vi_sample.wav", help="Đường dẫn đến file âm thanh tham chiếu")
args = parser.parse_args()

language = args.language
input_text = args.input
reference_audio = args.reference
# @markdown Tự động chuẩn hóa chữ (VD: 20/11 -> hai mươi tháng mười một)
normalize_text = True # @param {type:"boolean"} #######
# @markdown In chi tiết xử lý
verbose = True # @param {type:"boolean"}
# @markdown Lưu từng câu thành file riêng lẻ.
output_chunks = False # @param {type:"boolean"}

from IPython.display import clear_output
def cry_and_quit():
    clear_output()
    print("> Lỗi rồi huhu 😭😭, bạn hãy nhấn chạy lại phần này nhé!")
    quit()

import os
import string
import unicodedata
from datetime import datetime
from pprint import pprint

import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize
from unidecode import unidecode

try:
    from vinorm import TTSnorm
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except:
    cry_and_quit()

# Load model
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    # The problem is likely caused by using deepspeed for inference in this context.
    # Disabling deepspeed for inference by setting use_deepspeed=False.
    XTTS_MODEL.load_checkpoint(config,
                               checkpoint_path=xtts_checkpoint,
                               vocab_path=xtts_vocab,
                               use_deepspeed=False) # Changed to False
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return XTTS_MODEL


# ... (rest of your code remains the same) ...


def get_file_name(text, max_char=50):
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S%f")
    filename = f"{current_datetime[:-3]}_{filename}"
    return filename


def calculate_keep_len(text, lang):
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = (
        text.count(".")
        + text.count("!")
        + text.count("?")
        + text.count(",")
    )

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1


def normalize_vietnamese_text(text):
    try:
        text = (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
        )
        return text
    except Exception as e:
        print(f"Lỗi khi chuẩn hóa văn bản: {e}")
        return text  # Trả về văn bản gốc nếu lỗi


def run_tts(XTTS_MODEL, lang, tts_text, speaker_audio_file,
            normalize_text= True,
            verbose=False,
            output_chunks=False):
    """
    Run text-to-speech (TTS) synthesis using the provided XTTS_MODEL.

    Args:
        XTTS_MODEL: A pre-trained TTS model.
        lang (str): The language of the input text.
        tts_text (str): The text to be synthesized into speech.
        speaker_audio_file (str): Path to the audio file of the speaker to condition the synthesis on.
        normalize_text (bool, optional): Whether to normalize the input text. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Defaults to False.
        output_chunks (bool, optional): Whether to save synthesized speech chunks separately. Defaults to False.

    Returns:
        str: Path to the synthesized audio file.
    """

    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Đi lên thư mục cha (backend) rồi sang frontend/mp3
    output_dir = os.path.join(base_dir, "..", "..", "frontend", "mp3")
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục mp3 nếu chưa tồn tại

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )

    if normalize_text and lang == "vi":
        # Bug on google colab
        try:
            tts_text = normalize_vietnamese_text(tts_text)
        except:
            cry_and_quit()

    if lang in ["ja", "zh-cn"]:
        tts_texts = tts_text.split("。")
    else:
        tts_texts = sent_tokenize(tts_text)

    if verbose:
        print("Text for TTS:")
        pprint(tts_texts)

    wav_chunks = []
    for text in tqdm(tts_texts):
        if text.strip() == "":
            continue

        wav_chunk = XTTS_MODEL.inference(
            text=text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
        )

        # Quick hack for short sentences
        keep_len = calculate_keep_len(text, lang)
        wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len])
        file_name = get_file_name(text)
        if output_chunks:
            out_path = os.path.join(output_dir, f"{file_name}.wav")
            torchaudio.save(out_path, wav_chunk["wav"].unsqueeze(0), 24000)
            if verbose:
                print(f"Saved chunk to mp3/{file_name}")

        wav_chunks.append(wav_chunk["wav"])

    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    out_path = os.path.join(output_dir, f"{file_name}.wav")
    torchaudio.save(out_path, out_wav, 24000)

    if verbose:
        print(f"Saved final file to mp3/{file_name}.wav")

    return f"mp3/{file_name}"


language_code_map = {
    "Tiếng Việt": "vi",
    "Tiếng Anh": "en",
    "Tiếng Tây Ban Nha": "es",
    "Tiếng Pháp": "fr",
    "Tiếng Đức": "de",
    "Tiếng Ý": "it",
    "Tiếng Bồ Đào Nha": "pt",
    "Tiếng Ba Lan": "pl",
    "Tiếng Thổ Nhĩ Kỳ": "tr",
    "Tiếng Nga": "ru",
    "Tiếng Hà Lan": "nl",
    "Tiếng Séc": "cs",
    "Tiếng Ả Rập": "ar",
    "Tiếng Trung (giản thể)": "zh-cn",
    "Tiếng Nhật": "ja",
    "Tiếng Hungary": "hu",
    "Tiếng Hàn": "ko",
    "Tiếng Hindi": "hi"
}

print("> Đang nạp mô hình...")
try:
    if not vixtts_model:
        vixtts_model = load_model(xtts_checkpoint="model/model.pth",
                                xtts_config="model/config.json",
                                xtts_vocab="model/vocab.json")
except:
    vixtts_model = load_model(xtts_checkpoint="model/model.pth",
                                xtts_config="model/config.json",
                                xtts_vocab="model/vocab.json")
clear_output()
print("> Đã nạp mô hình")

if not os.path.exists(reference_audio):
    print("⚠️⚠️⚠️Bạn chưa tải file âm thanh lên. Hãy chọn giọng khác, hoặc tải file của bạn lên ở bên dưới.⚠️⚠️⚠️")
    audio_file="/content/model/vi_sample.wav"
else:
    audio_file = run_tts(vixtts_model,
            lang=language_code_map[language],
            tts_text=input_text,
            speaker_audio_file=reference_audio,
            normalize_text=normalize_text,
            verbose=verbose,
            output_chunks=output_chunks,)

#from IPython.display import Audio
#Audio(audio_file)