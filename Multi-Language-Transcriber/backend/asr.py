# import os, json, uuid, tempfile
# from typing import List, Dict, Any, Optional
# import ffmpeg
# from faster_whisper import WhisperModel
# from pydub import AudioSegment
# import langid

# # ---------- Paths ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---------- Utilities ----------
# def extract_audio(input_path: str, out_wav: str, sr: int = 16000):
#     (
#         ffmpeg
#         .input(input_path)
#         .output(out_wav, ac=1, ar=sr, format='wav', loglevel="error")
#         .overwrite_output()
#         .run()
#     )
#     return out_wav

# def secs_to_srt_time(t: float) -> str:
#     ms = int((t - int(t)) * 1000)
#     h = int(t // 3600); t -= h * 3600
#     m = int(t // 60);   t -= m * 60
#     s = int(t)
#     return f"{h:02}:{m:02}:{s:02},{ms:03}"

# def write_txt(text: str, path: str):
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(text.strip() + "\n")

# def write_srt(segments: List[Dict[str, Any]], path: str):
#     with open(path, "w", encoding="utf-8") as f:
#         for i, seg in enumerate(segments, 1):
#             f.write(f"{i}\n")
#             f.write(f"{secs_to_srt_time(seg['start'])} --> {secs_to_srt_time(seg['end'])}\n")
#             f.write(seg['text'].strip() + "\n\n")

# def write_vtt(segments: List[Dict[str, Any]], path: str):
#     with open(path, "w", encoding="utf-8") as f:
#         f.write("WEBVTT\n\n")
#         for seg in segments:
#             start = secs_to_srt_time(seg['start']).replace(",", ".")
#             end   = secs_to_srt_time(seg['end']).replace(",", ".")
#             f.write(f"{start} --> {end}\n")
#             f.write(seg['text'].strip() + "\n\n")

# def write_json(payload: Dict[str, Any], path: str):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)

# # ---------- Translation (optional) ----------
# # Uses facebook/m2m100_418M for many-to-many translation.
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# class M2MTranslator:
#     def __init__(self, model_name: str = "facebook/m2m100_418M", device: Optional[str] = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(BASE_DIR, "models"))
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=os.path.join(BASE_DIR, "models")).to(self.device)

#     def translate(self, text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 1024) -> str:
#         # src_lang / tgt_lang are ISO 639-1 codes like 'en', 'hi', 'fr', etc.
#         # m2m expects language codes set on tokenizer.
#         self.tokenizer.src_lang = src_lang
#         encoded = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
#         generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang), max_new_tokens=max_new_tokens)
#         return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# # ---------- ASR Core ----------
# class ASRService:
#     def __init__(self, whisper_model_size: str = "small", compute_type: str = "auto"):
#         # compute_type: "auto" | "int8" | "float16" | "float32"
#         self.model = WhisperModel(whisper_model_size, device="auto", compute_type=compute_type)

#     def transcribe(
#         self,
#         media_path: str,
#         vad_filter: bool = True,
#         beam_size: int = 5,
#         temperature: float = 0.0,
#         language: Optional[str] = None,      # e.g., "en", "hi" (if you want to force)
#         task: str = "transcribe"             # "transcribe" or "translate" (translate -> English)
#     ) -> Dict[str, Any]:
#         # Convert to 16k mono WAV for best performance
#         with tempfile.TemporaryDirectory() as td:
#             wav_path = os.path.join(td, "audio.wav")
#             extract_audio(media_path, wav_path, sr=16000)

#             segments_gen, info = self.model.transcribe(
#                 wav_path,
#                 vad_filter=vad_filter,
#                 beam_size=beam_size,
#                 temperature=temperature,
#                 language=language,  # None -> auto-detect
#                 task=task
#             )

#             segments = []
#             full_text_chunks = []
#             for seg in segments_gen:
#                 item = {
#                     "id": seg.id,
#                     "start": seg.start,
#                     "end": seg.end,
#                     "text": seg.text.strip()
#                 }
#                 segments.append(item)
#                 full_text_chunks.append(item["text"])

#             detected_language = info.language  # ISO 639-1 code
#             return {
#                 "language": detected_language,
#                 "duration": info.duration,
#                 "segments": segments,
#                 "text": " ".join(full_text_chunks).strip()
#             }

# def save_all_outputs(
#     payload: Dict[str, Any],
#     session_id: str,
#     stem: str,
#     out_dir: str,
#     make_txt=True, make_srt=True, make_vtt=True, make_json=True
# ) -> Dict[str, str]:
#     os.makedirs(out_dir, exist_ok=True)
#     paths = {}
#     if make_txt:
#         p = os.path.join(out_dir, f"{stem}.txt")
#         write_txt(payload["text"], p); paths["txt"] = p
#     if make_srt:
#         p = os.path.join(out_dir, f"{stem}.srt")
#         write_srt(payload["segments"], p); paths["srt"] = p
#     if make_vtt:
#         p = os.path.join(out_dir, f"{stem}.vtt")
#         write_vtt(payload["segments"], p); paths["vtt"] = p
#     if make_json:
#         p = os.path.join(out_dir, f"{stem}.json")
#         write_json(payload, p); paths["json"] = p
#     return paths

# def guess_lang_code_from_text(text: str) -> str:
#     # Fallback language detector (if Whisper language is None)
#     code, _ = langid.classify(text[:1000] if text else "")
#     return code or "en"



# asr.py
import os, json, tempfile
from typing import List, Dict, Any, Optional

import ffmpeg
from faster_whisper import WhisperModel
import langid

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Utilities ----------
def extract_audio(input_path: str, out_wav: str, sr: int = 16000):
    """
    Convert any supported audio/video file to mono 16 kHz WAV using ffmpeg.
    """
    (
        ffmpeg
        .input(input_path)
        .output(out_wav, ac=1, ar=sr, format='wav', loglevel="error")
        .overwrite_output()
        .run()
    )
    return out_wav

def secs_to_srt_time(t: float) -> str:
    ms = int((t - int(t)) * 1000)
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_txt(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def write_srt(segments: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{secs_to_srt_time(seg['start'])} --> {secs_to_srt_time(seg['end'])}\n")
            f.write(seg['text'].strip() + "\n\n")

def write_vtt(segments: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = secs_to_srt_time(seg['start']).replace(",", ".")
            end   = secs_to_srt_time(seg['end']).replace(",", ".")
            f.write(f"{start} --> {end}\n")
            f.write(seg['text'].strip() + "\n\n")

def write_json(payload: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ---------- Translation (optional) ----------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class M2MTranslator:
    """
    Many-to-many text translator using facebook/m2m100_418M.
    Use ISO-639-1 codes for src_lang/tgt_lang (e.g., 'en', 'hi', 'fr').
    """
    def __init__(self, model_name: str = "facebook/m2m100_418M", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = os.path.join(BASE_DIR, "models")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)

    def translate(self, text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 1024) -> str:
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
            max_new_tokens=max_new_tokens
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# ---------- ASR Core (with safe compute_type fallback) ----------
class ASRService:
    """
    Whisper (faster-whisper) service.
    Attempts requested compute_type first; if unsupported (e.g., float16 on CPU),
    falls back to int8, then float32 to avoid 500 errors.
    """
    def __init__(self, whisper_model_size: str = "small", compute_type: str = "auto"):
        self.model = None
        last_err: Optional[Exception] = None

        def try_init(ct: str) -> bool:
            nonlocal last_err
            try:
                self.model = WhisperModel(whisper_model_size, device="auto", compute_type=ct)
                return True
            except ValueError as e:
                last_err = e
                return False

        attempts: List[str] = []
        if compute_type and compute_type != "auto":
            attempts.append(compute_type)
        # Always add safe fallbacks
        for ct in ["int8", "float32"]:
            if ct not in attempts:
                attempts.append(ct)

        for ct in attempts:
            if try_init(ct):
                break

        if self.model is None:
            # Last resort (shouldn't hit)
            if not try_init("float32"):
                raise RuntimeError(f"Failed to initialize WhisperModel. Last error: {last_err}")

    def transcribe(
        self,
        media_path: str,
        vad_filter: bool = True,
        beam_size: int = 5,
        temperature: float = 0.0,
        language: Optional[str] = None,
        task: str = "transcribe"  # "transcribe" or "translate" (to English)
    ) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "audio.wav")
            extract_audio(media_path, wav_path, sr=16000)

            segments_gen, info = self.model.transcribe(
                wav_path,
                vad_filter=vad_filter,
                beam_size=beam_size,
                temperature=temperature,
                language=language,  # None -> auto-detect
                task=task
            )

            segments: List[Dict[str, Any]] = []
            full_text_chunks: List[str] = []
            for seg in segments_gen:
                item = {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip()
                }
                segments.append(item)
                full_text_chunks.append(item["text"])

            return {
                "language": info.language,    # ISO 639-1 code
                "duration": info.duration,
                "segments": segments,
                "text": " ".join(full_text_chunks).strip()
            }

def save_all_outputs(
    payload: Dict[str, Any],
    session_id: str,
    stem: str,
    out_dir: str,
    make_txt=True, make_srt=True, make_vtt=True, make_json=True
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    if make_txt:
        p = os.path.join(out_dir, f"{stem}.txt")
        write_txt(payload["text"], p); paths["txt"] = p
    if make_srt:
        p = os.path.join(out_dir, f"{stem}.srt")
        write_srt(payload["segments"], p); paths["srt"] = p
    if make_vtt:
        p = os.path.join(out_dir, f"{stem}.vtt")
        write_vtt(payload["segments"], p); paths["vtt"] = p
    if make_json:
        p = os.path.join(out_dir, f"{stem}.json")
        write_json(payload, p); paths["json"] = p
    return paths

def guess_lang_code_from_text(text: str) -> str:
    code, _ = langid.classify(text[:1000] if text else "")
    return code or "en"

