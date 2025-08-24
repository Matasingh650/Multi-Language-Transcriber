import os, uuid, shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from asr import ASRService, save_all_outputs, OUTPUT_DIR, M2MTranslator

app = FastAPI(title="Multilingual Transcriber", version="1.0.0")

# ---------- CORS for local frontend ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve outputs for direct download
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Lazy singletons
_asr = None
_translator = None

def get_asr(model_size: str, compute_type: str):
    global _asr
    # Recreate only if size/compute_type changed; for simplicity we just new it each time
    return ASRService(whisper_model_size=model_size, compute_type=compute_type)

def get_translator():
    global _translator
    if _translator is None:
        _translator = M2MTranslator()
    return _translator

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/languages")
def languages():
    # Common target languages (ISO 639-1). M2M supports many more.
    return {
        "codes": [
            "en","hi","bn","ta","te","mr","gu","kn","ml","pa","ur",
            "fr","de","es","it","pt","ru","ja","ko","zh","ar","tr","vi","id","th"
        ]
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Form("small"),         # tiny, base, small, medium, large-v3
    compute_type: str = Form("auto"),        # auto, int8, float16, float32
    task: str = Form("transcribe"),          # transcribe | translate (translate -> EN)
    translate_to: Optional[str] = Form(None),# e.g., "en", "hi", "fr"  (if provided -> translate final text)
    make_txt: bool = Form(True),
    make_srt: bool = Form(True),
    make_vtt: bool = Form(True),
    make_json: bool = Form(True),
):
    # Save upload to temp
    session_id = str(uuid.uuid4())[:8]
    tmp_dir = os.path.join(OUTPUT_DIR, f"session_{session_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    orig_name = file.filename or "upload"
    input_path = os.path.join(tmp_dir, orig_name)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run ASR
    asr = get_asr(model_size, compute_type)
    result = asr.transcribe(input_path, task=task)  # auto language detection

    # Optional translation (many-to-many)
    translated_text = None
    if translate_to:
        try:
            from_lang = result["language"] or "en"
            if from_lang == translate_to:
                translated_text = result["text"]
            else:
                translator = get_translator()
                translated_text = translator.translate(result["text"], src_lang=from_lang, tgt_lang=translate_to)
        except Exception as e:
            translated_text = None

    # Save files
    stem = os.path.splitext(os.path.basename(orig_name))[0]
    # If translation requested, save both original and translated variants
    saved = {}

    # Original language outputs
    saved["original"] = save_all_outputs(
        result, session_id, stem, tmp_dir,
        make_txt=make_txt, make_srt=make_srt, make_vtt=make_vtt, make_json=make_json
    )

    # Translated outputs
    if translate_to and translated_text:
        translated_payload = {
            "language": translate_to,
            "duration": result["duration"],
            "segments": result["segments"],  # timestamps unchanged
            "text": translated_text
        }
        saved["translated"] = save_all_outputs(
            translated_payload, session_id, f"{stem}_{translate_to}", tmp_dir,
            make_txt=make_txt, make_srt=make_srt, make_vtt=make_vtt, make_json=make_json
        )

    # Build downloadable URLs (FastAPI static mount)
    def to_url(p: str) -> str:
        rel = os.path.relpath(p, OUTPUT_DIR).replace("\\", "/")
        return f"/outputs/{rel}"

    downloads = {}
    for variant, paths in saved.items():
        downloads[variant] = {k: to_url(v) for k, v in paths.items()}

    return JSONResponse({
        "session": session_id,
        "detected_language": result["language"],
        "duration_sec": result["duration"],
        "text_preview": (result["text"][:500] + ("..." if len(result["text"]) > 500 else "")),
        "downloads": downloads
    })




