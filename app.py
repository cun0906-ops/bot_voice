import io
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tts_engine import VoiceCloner

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
LEGACY_VOICE_FILE = BASE_DIR / "voicesvan_thanh_tri.wav"

app = FastAPI(title="Voice Cloning Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.cloner = None


def get_cloner():
    if app.state.cloner is None:
        app.state.cloner = VoiceCloner()
    return app.state.cloner


def resolve_voice_path(file_name: str) -> str:
    preferred = BASE_DIR / "voices" / file_name
    if preferred.exists():
        return str(preferred)

    legacy = BASE_DIR / file_name
    if legacy.exists():
        return str(legacy)

    if file_name == "van_thanh_tri.wav" and LEGACY_VOICE_FILE.exists():
        return str(LEGACY_VOICE_FILE)

    raise FileNotFoundError(f"Không tìm thấy file giọng mẫu: {file_name}")


# Danh mục các giọng mẫu lưu sẵn trong hệ thống
VOICE_PRESETS = {
    "van_thanh_tri": {
        "audio_path": resolve_voice_path("van_thanh_tri_short.wav"),
        "ref_text": "Dạ, em là Vân ở bên công ty An Đại Phát ở Thanh Trì, Hà Nội chuyên mạ kẽm điện phân và nhúng nóng ạ."
    },
    # Sau này muốn thêm giọng nam/giọng khác chỉ cần nạp thêm vào đây
}


class SynthesisRequest(BaseModel):
    text: str
    voice_id: str = "van_thanh_tri"
    speed: float = 1.0


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": app.state.cloner.device if app.state.cloner is not None else "cpu (ONNX Runtime)",
        "model_loaded": app.state.cloner is not None,
        "voices": list(VOICE_PRESETS.keys()),
    }


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html", media_type="text/html")


@app.get("/voices")
def list_voices():
    """Lấy danh sách các giọng có thể chọn"""
    return {"voices": list(VOICE_PRESETS.keys())}


@app.post("/synthesize")
def synthesize_speech(req: SynthesisRequest):
    if req.voice_id not in VOICE_PRESETS:
        raise HTTPException(status_code=400, detail="Không tìm thấy voice_id yêu cầu")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Nội dung văn bản không được để trống")

    try:
        speed = float(req.speed)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="speed phải là số thực hợp lệ") from exc
    if speed != 1.0:
        raise HTTPException(
            status_code=422,
            detail="Model tiếng Việt hiện chỉ hỗ trợ tốc độ tự nhiên 1.0.",
        )
    preset = VOICE_PRESETS[req.voice_id]
    output_filename = OUTPUT_DIR / f"{uuid.uuid4()}.wav"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cloner = get_cloner()
    result_path = cloner.generate(
        gen_text=req.text,
        ref_audio_path=preset["audio_path"],
        ref_text=preset["ref_text"],
        output_path=str(output_filename),
        speed=speed,
    )

    return FileResponse(result_path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)