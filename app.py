import io
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tts_engine import VoiceCloner

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
VOICES_DIR = BASE_DIR / "voices"

# Dọn dẹp file dính tên cũ nếu còn tồn tại
LEGACY_STRAY_FILE = BASE_DIR / "voicesvan_thanh_tri.wav"
if LEGACY_STRAY_FILE.exists():
    try:
        LEGACY_STRAY_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def cleanup_old_outputs(max_age_seconds: int = 3600):
    """Tự động dọn dẹp các file audio trong outputs/ cũ hơn 1 tiếng để giải phóng đĩa."""
    try:
        now = time.time()
        if OUTPUT_DIR.exists():
            for audio_file in OUTPUT_DIR.glob("*.wav"):
                if audio_file.is_file() and (now - audio_file.stat().st_mtime > max_age_seconds):
                    audio_file.unlink(missing_ok=True)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo model và warmup
    print("🚀 Đang khởi động Voice Studio Engine...")
    try:
        get_cloner()
        print("✅ Model VieNeu TTS đã nạp thành công.")
    except Exception as exc:
        print(f"⚠️ Model chưa nạp sẵn (sẽ nạp khi có request): {exc}")
    yield
    print("🛑 Đang tắt Voice Studio Engine...")


app = FastAPI(title="Voice Studio - Vietnamese Voice Cloning Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.cloner = None
_CACHED_VOICES: Optional[Dict[str, Dict[str, Any]]] = None


def get_cloner() -> VoiceCloner:
    if app.state.cloner is None:
        app.state.cloner = VoiceCloner(threads=4)
    return app.state.cloner


def get_all_voices() -> Dict[str, Dict[str, Any]]:
    """Tổng hợp danh sách tất cả các giọng (Clone + Built-in Presets). Có cache để phản hồi siêu tốc."""
    global _CACHED_VOICES
    if _CACHED_VOICES is not None and len(_CACHED_VOICES) > 1:
        return _CACHED_VOICES

    catalog: Dict[str, Dict[str, Any]] = {}

    # 1. Các giọng Clone riêng từ file mẫu trong voices/
    sample_short = VOICES_DIR / "van_thanh_tri_short.wav"
    sample_full = VOICES_DIR / "van_thanh_tri.wav"
    primary_audio = sample_short if sample_short.exists() else sample_full

    if primary_audio.exists():
        catalog["van_thanh_tri"] = {
            "id": "van_thanh_tri",
            "name": "Vân Thanh Trì",
            "description": "Nữ · Bắc · Giọng doanh nghiệp mạ kẽm An Đại Phát",
            "gender": "female",
            "region": "Bắc",
            "category": "Doanh nghiệp (Clone)",
            "is_clone": True,
            "audio_path": str(primary_audio),
            "ref_text": "Dạ, em là Vân ở bên công ty An Đại Phát ở Thanh Trì, Hà Nội chuyên mạ kẽm điện phân và nhúng nóng ạ.",
        }

    # 2. Tích hợp tất cả các giọng mẫu có sẵn của VieNeu v3 Turbo
    try:
        cloner = get_cloner()
        presets = cloner.get_preset_voices()
        for voice_name, info in presets.items():
            gender = info.get("gender", "")
            region = info.get("region", "")
            desc = info.get("description", "")

            # Phân loại category hiển thị
            if region == "Bắc" and gender == "male":
                category = "Giọng Nam - Miền Bắc"
            elif region == "Bắc" and gender == "female":
                category = "Giọng Nữ - Miền Bắc"
            elif region == "Nam" and gender == "female":
                category = "Giọng Nữ - Miền Nam"
            elif region == "Nam" and gender == "male":
                category = "Giọng Nam - Miền Nam"
            elif region == "Trung":
                category = "Giọng Miền Trung"
            else:
                category = "Giọng Đọc Phổ Thông"

            catalog[voice_name] = {
                "id": voice_name,
                "name": voice_name,
                "description": desc or f"{gender.title()} · {region}",
                "gender": gender,
                "region": region,
                "category": category,
                "is_clone": False,
                "audio_path": None,
                "ref_text": "",
            }
        _CACHED_VOICES = catalog
    except Exception as e:
        print(f"⚠️ Chưa thể đọc danh sách preset voices: {e}")

    return catalog



class SynthesisRequest(BaseModel):
    text: str
    voice_id: str = "van_thanh_tri"
    speed: float = 1.0
    temperature: Any = "default"
    pause_style: str = "default"


@app.get("/health")
def health_check():
    all_voices = get_all_voices()
    return {
        "status": "ok",
        "device": app.state.cloner.device if app.state.cloner is not None else "cpu (ONNX Runtime)",
        "model_loaded": app.state.cloner is not None,
        "voice_count": len(all_voices),
        "voices": list(all_voices.keys()),
    }


@app.get("/")
def home():
    response = FileResponse(BASE_DIR / "static" / "index.html", media_type="text/html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/voices")
@app.get("/api/voices")
@app.get("/api/info")
def list_voices():
    """Lấy danh sách các giọng có thể chọn kèm thông tin phân loại (tương thích mọi phiên bản frontend)."""
    all_voices = get_all_voices()
    voice_list = list(all_voices.values())
    base_voices = [v for v in voice_list if not v.get("is_clone")]
    clone_voices = [v for v in voice_list if v.get("is_clone")]
    return {
        "status": "ok",
        "device": app.state.cloner.device if app.state.cloner is not None else "cpu (ONNX Runtime)",
        "model_loaded": app.state.cloner is not None,
        "voices": voice_list,
        "base_voices": base_voices,
        "clone_voices": clone_voices,
        "voice_ids": [v["id"] for v in voice_list],
    }


@app.get("/voices/{voice_id}/preview")
@app.get("/api/voices/{voice_id}/preview")
def preview_voice(voice_id: str):
    """Cung cấp audio nghe thử giọng đọc (có cơ chế cache tức thì)."""
    all_voices = get_all_voices()
    if voice_id not in all_voices:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy giọng đọc '{voice_id}'.",
        )

    voice_info = all_voices[voice_id]

    # 1. Nếu là giọng clone riêng và có file audio mẫu tham chiếu sẵn
    if voice_info.get("is_clone") and voice_info.get("audio_path"):
        audio_file = Path(voice_info["audio_path"])
        if audio_file.exists():
            return FileResponse(str(audio_file), media_type="audio/wav")

    # 2. Thư mục lưu cache các bản nghe thử
    previews_dir = VOICES_DIR / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in voice_id)
    cached_preview = previews_dir / f"{safe_name}.wav"
    if cached_preview.exists() and cached_preview.stat().st_size > 1024:
        return FileResponse(str(cached_preview), media_type="audio/wav")

    # 3. Tìm trong vieneu assets/samples
    try:
        import vieneu
        vieneu_samples = Path(vieneu.__file__).parent / "assets" / "samples"
        if vieneu_samples.exists():
            for sample_wav in vieneu_samples.glob("*.wav"):
                if voice_id.lower() in sample_wav.stem.lower():
                    return FileResponse(str(sample_wav), media_type="audio/wav")
    except Exception:
        pass

    # 4. Tự động sinh audio mẫu ngắn gọn (~1.5s - 2s)
    try:
        cloner = get_cloner()
        sample_text = f"Xin chào, tôi là giọng đọc {voice_info['name']} trên hệ thống Voice Studio."
        cloner.generate(
            gen_text=sample_text,
            voice_id=voice_id,
            output_path=str(cached_preview),
            speed=1.05,
            temperature=0.8,
            pause_style="natural",
        )
        if cached_preview.exists():
            return FileResponse(str(cached_preview), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tạo audio nghe thử: {str(exc)}",
        ) from exc

    raise HTTPException(status_code=404, detail="Không thể tạo nghe thử cho giọng này.")


@app.post("/synthesize")
@app.post("/api/synthesize")
def synthesize_speech(req: SynthesisRequest, background_tasks: BackgroundTasks):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Nội dung văn bản không được để trống.")

    all_voices = get_all_voices()
    if req.voice_id not in all_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy giọng đọc '{req.voice_id}'. Vui lòng chọn giọng có sẵn.",
        )

    try:
        speed = float(req.speed)
        speed = max(0.65, min(1.75, speed))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="speed phải là số thực hợp lệ.") from exc

    if req.temperature in [None, "default", ""]:
        temperature = "default"
    else:
        try:
            temperature = float(req.temperature)
            temperature = max(0.3, min(1.2, temperature))
        except (TypeError, ValueError):
            temperature = "default"

    pause_style = str(req.pause_style).lower() if req.pause_style else "default"
    valid_pause_styles = ["default", "natural", "relaxed", "expressive", "compact", "dramatic", "presentation", "rapid"]
    if pause_style not in valid_pause_styles:
        pause_style = "default"

    voice_info = all_voices[req.voice_id]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = OUTPUT_DIR / f"{uuid.uuid4()}.wav"

    try:
        cloner = get_cloner()
        result_path = cloner.generate(
            gen_text=req.text,
            voice_id=req.voice_id if not voice_info["is_clone"] else None,
            ref_audio_path=voice_info["audio_path"] if voice_info["is_clone"] else None,
            ref_text=voice_info.get("ref_text", ""),
            output_path=str(output_filename),
            speed=speed,
            temperature=temperature,
            pause_style=pause_style,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình tạo âm thanh: {str(exc)}",
        ) from exc

    # Thêm tác vụ nền dọn dẹp file audio cũ
    background_tasks.add_task(cleanup_old_outputs)

    return FileResponse(result_path, media_type="audio/wav")


def extract_preview_text(full_text: str, max_words: int = 35) -> str:
    """Trích xuất 1-2 câu đầu tiên (~10 giây đọc, tối đa 35 từ) từ văn bản người dùng để nghe thử nhanh."""
    clean = full_text.strip()
    if not clean:
        return ""
    parts = re.split(r"(?<=[.!?…\n])\s+", clean)
    collected = []
    word_count = 0
    for p in parts:
        p_strip = p.strip()
        if not p_strip:
            continue
        p_words = len(p_strip.split())
        if word_count == 0 or (word_count + p_words <= max_words):
            collected.append(p_strip)
            word_count += p_words
        else:
            break
    if not collected:
        words = clean.split()
        return " ".join(words[:max_words])
    return " ".join(collected)


@app.post("/synthesize-preview")
@app.post("/api/synthesize-preview")
def synthesize_preview_snippet(req: SynthesisRequest, background_tasks: BackgroundTasks):
    """Tạo audio nghe thử ~10 giây đoạn đầu của văn bản người dùng."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản trước khi nghe thử.")

    preview_text = extract_preview_text(req.text, max_words=35)
    if not preview_text:
        preview_text = req.text[:140]

    all_voices = get_all_voices()
    if req.voice_id not in all_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy giọng đọc '{req.voice_id}'.",
        )

    try:
        speed = float(req.speed)
        speed = max(0.65, min(1.75, speed))
    except (TypeError, ValueError):
        speed = 1.0

    if req.temperature in [None, "default", ""]:
        temperature = "default"
    else:
        try:
            temperature = float(req.temperature)
            temperature = max(0.3, min(1.2, temperature))
        except (TypeError, ValueError):
            temperature = "default"

    pause_style = str(req.pause_style).lower() if req.pause_style else "default"
    valid_pause_styles = ["default", "natural", "relaxed", "expressive", "compact", "dramatic", "presentation", "rapid"]
    if pause_style not in valid_pause_styles:
        pause_style = "default"

    voice_info = all_voices[req.voice_id]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = OUTPUT_DIR / f"snippet_{uuid.uuid4().hex[:8]}.wav"

    try:
        cloner = get_cloner()
        result_path = cloner.generate(
            gen_text=preview_text,
            voice_id=req.voice_id if not voice_info["is_clone"] else None,
            ref_audio_path=voice_info["audio_path"] if voice_info["is_clone"] else None,
            ref_text=voice_info.get("ref_text", ""),
            output_path=str(output_filename),
            speed=speed,
            temperature=temperature,
            pause_style=pause_style,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi tạo audio nghe thử: {str(exc)}",
        ) from exc

    background_tasks.add_task(cleanup_old_outputs)
    return FileResponse(result_path, media_type="audio/wav")


if __name__ == "__main__":
    import socket
    import sys
    import threading
    import time
    import webbrowser
    import uvicorn

    def is_port_in_use(port_num: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port_num)) == 0

    target_port = None
    if len(sys.argv) > 1:
        try:
            target_port = int(sys.argv[1])
        except ValueError:
            target_port = None

    if target_port is None:
        candidate_ports = [8000, 8080, 8001, 8002, 5000]
        target_port = 8000
        for p in candidate_ports:
            if not is_port_in_use(p):
                target_port = p
                break

    url = f"http://localhost:{target_port}"
    print(f"\n========================================================")
    print(f"🎙️ VOICE STUDIO SERVER SẴN SÀNG")
    print(f"👉 Địa chỉ: {url}")
    print(f"========================================================\n")

    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=target_port)
