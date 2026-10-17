import io
import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tts_engine import VoiceCloner
from multilingual_engine import (
    is_international_voice,
    get_international_languages,
    get_international_voices,
    generate_multilingual_speech,
    CURATED_INTERNATIONAL_VOICES,
    VOICE_MAP as INT_VOICE_MAP,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
VOICES_DIR = BASE_DIR / "voices"
CUSTOM_DIR = VOICES_DIR / "custom"
HISTORY_FILE = OUTPUT_DIR / "history.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_outputs(max_age_seconds: int = 7200):
    """Tự động dọn dẹp các file audio trong outputs/ cũ hơn 2 tiếng để giải phóng đĩa."""
    try:
        now = time.time()
        if OUTPUT_DIR.exists():
            for audio_file in OUTPUT_DIR.glob("*.wav"):
                if audio_file.is_file() and (now - audio_file.stat().st_mtime > max_age_seconds):
                    audio_file.unlink(missing_ok=True)
    except Exception:
        pass


def record_history(file_path: Path, voice_id: str, voice_name: str, text: str, speed: float, pitch: float, srt_url: Optional[str] = None):
    """Lưu trữ lịch sử các bản audio đã tạo kèm phụ đề .srt."""
    try:
        import soundfile as sf
        info = sf.info(str(file_path))
        duration = round(info.duration, 1)
    except Exception:
        duration = 0.0

    entry = {
        "id": file_path.stem,
        "filename": file_path.name,
        "url": f"/outputs/{file_path.name}",
        "srt_url": srt_url,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "text_snippet": (text[:80] + "...") if len(text) > 80 else text,
        "full_text": text,
        "duration": duration,
        "size_kb": round(file_path.stat().st_size / 1024, 1),
        "created_at": time.strftime("%H:%M %d/%m/%Y"),
        "speed": speed,
        "pitch": pitch,
    }

    try:
        data = []
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        data.insert(0, entry)
        data = data[:50]  # Giữ tối đa 50 bản gần nhất
        HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Không thể cập nhật history.json: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Đang khởi động Voice Studio Engine...")
    try:
        get_cloner()
        print("✅ Model VieNeu TTS đã nạp thành công.")
    except Exception as exc:
        print(f"⚠️ Model chưa nạp sẵn (sẽ nạp khi có request): {exc}")
    yield
    print("🛑 Đang tắt Voice Studio Engine...")


app = FastAPI(title="Voice Studio Pro - Vietnamese Voice Cloning Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ thư mục outputs cho phép tải và phát audio tĩnh
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.state.cloner = None
_CACHED_VOICES: Optional[Dict[str, Dict[str, Any]]] = None


def get_cloner() -> VoiceCloner:
    if app.state.cloner is None:
        app.state.cloner = VoiceCloner(threads=4)
    return app.state.cloner


def get_all_voices() -> Dict[str, Dict[str, Any]]:
    """Tổng hợp danh sách tất cả các giọng (Clone + Built-in Presets). Có cache để phản hồi tức thì."""
    global _CACHED_VOICES
    if _CACHED_VOICES is not None and len(_CACHED_VOICES) > 1:
        return _CACHED_VOICES

    catalog: Dict[str, Dict[str, Any]] = {}

    # 1. Các giọng clone từ file voices/custom_voices.json
    custom_json = VOICES_DIR / "custom_voices.json"
    if custom_json.exists():
        try:
            custom_list = json.loads(custom_json.read_text(encoding="utf-8"))
            for v in custom_list:
                audio_p = Path(v.get("audio_path", ""))
                if not audio_p.is_absolute():
                    audio_p = BASE_DIR / audio_p
                if audio_p.exists():
                    v_entry = dict(v)
                    v_entry["audio_path"] = str(audio_p)
                    v_entry["category"] = v.get("category") or "Giọng Clone Của Bạn"
                    catalog[v["id"]] = v_entry
        except Exception as e:
            print(f"⚠️ Lỗi đọc custom_voices.json: {e}")

    # Fallback cho Vân Thanh Trì nếu chưa có trong custom_voices.json
    if "van_thanh_tri" not in catalog:
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
                "is_builtin": True,
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
                "is_builtin": True,
            }
    except Exception as e:
        print(f"⚠️ Chưa thể nạp danh sách preset voices: {e}")

    # 3. Đăng ký các giọng đọc quốc tế chuẩn bản xứ (Anh, Trung, Nhật, Hàn, Pháp, Đức...)
    for iv in CURATED_INTERNATIONAL_VOICES:
        catalog[iv["id"]] = {
            "id": iv["id"],
            "name": iv["name"],
            "description": iv["description"],
            "gender": iv["gender"],
            "region": iv["lang_name"],
            "category": f"🌐 {iv['lang_name']}",
            "is_clone": False,
            "is_international": True,
            "lang": iv["lang"],
            "audio_path": None,
            "ref_text": "",
            "is_builtin": True,
        }

    _CACHED_VOICES = catalog
    return catalog


class SynthesisRequest(BaseModel):
    text: str
    voice_id: str = "van_thanh_tri"
    speed: float = 1.0
    pitch: float = 0.0
    stability: float = 50.0
    similarity: float = 75.0
    speaker_boost: bool = True
    temperature: Any = "default"
    pause_style: str = "default"
    comma_pause: float = 0.3
    sentence_pause: float = 0.5
    segment_pause: float = 0.3
    strip_special_chars: bool = False
    max_chars_per_line: int = 200
    normalize_text: bool = True
    generate_srt: bool = True
    custom_dict: Optional[Dict[str, str]] = None


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
    """Lấy danh sách các giọng có thể chọn kèm thông tin phân loại."""
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
async def preview_voice(voice_id: str):
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

    # 3. Nếu là giọng quốc tế
    if is_international_voice(voice_id):
        sample_text = f"Hello! This is a voice preview of {voice_info['name']} on Voice Studio Pro."
        if voice_id.startswith("zh-"):
            sample_text = f"您好！这是 {voice_info['name']} 在系统上的声音试听样本。"
        elif voice_id.startswith("ja-"):
            sample_text = f"こんにちは！これは {voice_info['name']} の音声プレビューです。"
        elif voice_id.startswith("ko-"):
            sample_text = f"안녕하세요! 이것은 {voice_info['name']} 목소리 샘플입니다."
        elif voice_id.startswith("fr-"):
            sample_text = f"Bonjour! Ceci est un aperçu de la voix {voice_info['name']}."
        elif voice_id.startswith("de-"):
            sample_text = f"Hallo! Dies ist eine Hörprobe von {voice_info['name']}."
        elif voice_id.startswith("es-"):
            sample_text = f"¡Hola! Esta es una muestra de la voz de {voice_info['name']}."
        elif voice_id.startswith("ru-"):
            sample_text = f"Здравствуйте! Это образец голоса {voice_info['name']}."
        elif voice_id.startswith("th-"):
            sample_text = f"สวัสดีค่ะ นี่คือตัวอย่างเสียงของ {voice_info['name']}"

        try:
            await generate_multilingual_speech(
                text=sample_text,
                voice_id=voice_id,
                output_path=str(cached_preview),
                speed=1.0,
                pitch=0.0,
                speaker_boost=True,
                generate_srt=False,
            )
            if cached_preview.exists():
                return FileResponse(str(cached_preview), media_type="audio/wav")
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi khi tạo audio nghe thử quốc tế: {str(exc)}",
            ) from exc

    # 4. Tự động sinh audio mẫu tiếng Việt (~1.5s - 2s)
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


@app.get("/api/multilingual/languages")
def list_multilingual_languages():
    """Lấy danh sách các ngôn ngữ quốc tế hỗ trợ."""
    return {"status": "ok", "languages": get_international_languages()}


@app.get("/api/multilingual/voices")
def list_multilingual_voices(lang: Optional[str] = None):
    """Lấy danh sách các giọng quốc tế theo ngôn ngữ."""
    return {"status": "ok", "voices": get_international_voices(lang)}


@app.post("/api/voices/clone")
async def clone_new_voice(
    audio_file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    gender: str = Form("female"),
    region: str = Form("Bắc"),
    ref_text: str = Form(""),
):
    """Nhân bản một giọng nói mới từ file âm thanh người dùng tải lên."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập tên cho giọng đọc.")

    clean_name = name.strip()
    safe_slug = re.sub(r"[^\w\-]+", "_", clean_name.lower()).strip("_") or "voice"
    voice_id = f"custom_{safe_slug}_{uuid.uuid4().hex[:6]}"
    safe_filename = "".join(c if c.isalnum() or c in (".", "-", "_") else "_" for c in audio_file.filename)
    temp_path = CUSTOM_DIR / f"upload_{uuid.uuid4().hex[:6]}_{safe_filename}"
    final_wav = CUSTOM_DIR / f"{voice_id}.wav"

    try:
        content = await audio_file.read()
        if len(content) < 1024:
            raise HTTPException(status_code=400, detail="File âm thanh quá nhỏ hoặc rỗng.")
        temp_path.write_bytes(content)

        # Chuyển đổi và chuẩn hóa thành file WAV 48kHz mono 16-bit
        import soundfile as sf
        import librosa

        try:
            wav, sr = librosa.load(str(temp_path), sr=48000, mono=True)
        except Exception as load_err:
            # Fallback 1: Thử đọc trực tiếp bằng soundfile
            try:
                wav, sr = sf.read(str(temp_path), dtype="float32")
                if getattr(wav, "ndim", 1) > 1:
                    wav = wav.mean(axis=1)
                if sr != 48000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=48000)
            except Exception:
                # Fallback 2: Thử đọc bằng pydub
                try:
                    from pydub import AudioSegment
                    seg = AudioSegment.from_file(str(temp_path))
                    seg = seg.set_frame_rate(48000).set_channels(1)
                    buf = io.BytesIO()
                    seg.export(buf, format="wav")
                    buf.seek(0)
                    wav, sr = sf.read(buf, dtype="float32")
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Định dạng âm thanh chưa được hỗ trợ trên máy chủ ({str(load_err)}). Bạn hãy cập nhật trang web để trình duyệt tự động chuyển file sang chuẩn WAV.",
                    )

        max_amp = float(np.max(np.abs(wav)))
        if max_amp > 1e-4:
            wav = wav * (0.95 / max_amp)
        sf.write(str(final_wav), wav, 48000, subtype="PCM_16")

        # Đăng ký với cloner
        cloner = get_cloner()
        cloner.register_custom_voice(
            voice_id=voice_id,
            audio_path=str(final_wav),
            description=description.strip() or f"{gender.title()} · {region}",
            gender=gender,
        )

        # Lưu vào custom_voices.json
        custom_json = VOICES_DIR / "custom_voices.json"
        existing = []
        if custom_json.exists():
            try:
                existing = json.loads(custom_json.read_text(encoding="utf-8"))
            except Exception:
                existing = []

        new_voice_entry = {
            "id": voice_id,
            "name": clean_name,
            "description": description.strip() or f"{gender.title()} · {region}",
            "gender": gender,
            "region": region,
            "category": "Giọng Clone Của Bạn",
            "is_clone": True,
            "audio_path": str(final_wav),
            "ref_text": ref_text.strip(),
            "is_builtin": False,
            "created_at": time.strftime("%H:%M %d/%m/%Y"),
        }
        existing.append(new_voice_entry)
        custom_json.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

        # Xóa cache danh mục giọng để load lại
        global _CACHED_VOICES
        _CACHED_VOICES = None

        return {
            "status": "ok",
            "message": f"Nhân bản thành công giọng '{clean_name}'!",
            "voice": new_voice_entry,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file âm thanh: {str(exc)}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.delete("/api/voices/{voice_id}")
def delete_custom_voice(voice_id: str):
    """Xóa một giọng clone do người dùng tạo."""
    custom_json = VOICES_DIR / "custom_voices.json"
    if not custom_json.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng cần xóa.")

    try:
        voices = json.loads(custom_json.read_text(encoding="utf-8"))
    except Exception:
        voices = []

    target = next((v for v in voices if v["id"] == voice_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng này.")

    if target.get("is_builtin"):
        raise HTTPException(status_code=400, detail="Không thể xóa giọng mẫu mặc định của hệ thống.")

    # Xóa file audio
    if target.get("audio_path"):
        audio_p = Path(target["audio_path"])
        if audio_p.exists():
            audio_p.unlink(missing_ok=True)

    # Hủy đăng ký trong engine
    try:
        cloner = get_cloner()
        cloner.remove_custom_voice(voice_id)
    except Exception:
        pass

    # Cập nhật JSON
    updated = [v for v in voices if v["id"] != voice_id]
    custom_json.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    global _CACHED_VOICES
    _CACHED_VOICES = None

    return {"status": "ok", "deleted": voice_id}


@app.get("/api/history")
def get_audio_history():
    """Lấy danh sách các audio đã tạo gần đây."""
    if not HISTORY_FILE.exists():
        return {"status": "ok", "history": []}
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        valid = [item for item in data if (OUTPUT_DIR / item["filename"]).exists()]
        return {"status": "ok", "history": valid}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "history": []}


@app.delete("/api/history/{filename}")
def delete_audio_history_item(filename: str):
    """Xóa một audio trong lịch sử và file tương ứng trên đĩa."""
    target = OUTPUT_DIR / filename
    if target.exists():
        target.unlink(missing_ok=True)

    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            data = [item for item in data if item.get("filename") != filename]
            HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return {"status": "ok", "deleted": filename}


@app.delete("/api/history")
def clear_all_audio_history():
    """Xóa toàn bộ lịch sử audio."""
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.wav"):
            f.unlink(missing_ok=True)
    if HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")
    return {"status": "ok", "message": "Đã dọn dẹp toàn bộ lịch sử audio."}


@app.post("/synthesize")
@app.post("/api/synthesize")
async def synthesize_speech(req: SynthesisRequest, background_tasks: BackgroundTasks):
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

    try:
        pitch = float(req.pitch)
        pitch = max(-4.0, min(4.0, pitch))
    except (TypeError, ValueError):
        pitch = 0.0

    normalize_text = bool(req.normalize_text)

    voice_info = all_voices[req.voice_id]
    output_filename = OUTPUT_DIR / f"{uuid.uuid4()}.wav"
    output_srt_path = output_filename.with_suffix(".srt")

    if is_international_voice(req.voice_id):
        try:
            result = await generate_multilingual_speech(
                text=req.text,
                voice_id=req.voice_id,
                output_path=str(output_filename),
                output_srt_path=str(output_srt_path),
                speed=speed,
                pitch=pitch,
                speaker_boost=req.speaker_boost,
                generate_srt=req.generate_srt,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi tạo âm thanh quốc tế: {str(exc)}",
            ) from exc
    else:
        try:
            cloner = get_cloner()
            result = cloner.generate(
                gen_text=req.text,
                voice_id=req.voice_id if not voice_info["is_clone"] else None,
                ref_audio_path=voice_info["audio_path"] if voice_info["is_clone"] else None,
                ref_text=voice_info.get("ref_text", ""),
                output_path=str(output_filename),
                output_srt_path=str(output_srt_path),
                speed=speed,
                pitch=pitch,
                stability=req.stability,
                similarity=req.similarity,
                speaker_boost=req.speaker_boost,
                temperature=temperature,
                pause_style=pause_style,
                comma_pause=req.comma_pause,
                sentence_pause=req.sentence_pause,
                segment_pause=req.segment_pause,
                strip_special_chars=req.strip_special_chars,
                max_chars_per_line=req.max_chars_per_line,
                normalize_text=normalize_text,
                generate_srt=req.generate_srt,
                custom_dict=req.custom_dict,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi trong quá trình tạo âm thanh: {str(exc)}",
            ) from exc

    audio_path = result["audio_path"] if isinstance(result, dict) else str(result)
    srt_path = result.get("srt_path") if isinstance(result, dict) else None
    duration = result.get("duration", 0.0) if isinstance(result, dict) else 0.0
    cues = result.get("cues", []) if isinstance(result, dict) else []

    audio_url = f"/outputs/{Path(audio_path).name}"
    srt_url = f"/outputs/{Path(srt_path).name}" if srt_path and Path(srt_path).exists() else None

    # Lưu vào lịch sử và dọn dẹp file cũ
    record_history(
        file_path=Path(audio_path),
        voice_id=req.voice_id,
        voice_name=voice_info["name"],
        text=req.text,
        speed=speed,
        pitch=pitch,
        srt_url=srt_url,
    )
    background_tasks.add_task(cleanup_old_outputs)

    return JSONResponse({
        "status": "ok",
        "audio_url": audio_url,
        "srt_url": srt_url,
        "duration": duration,
        "cues": cues,
    })


@app.post("/api/parse-script")
async def parse_script_file(file: UploadFile = File(...)):
    """Trích xuất nội dung văn bản từ file kịch bản .txt hoặc .srt tải lên."""
    content = await file.read()
    text = ""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("cp1258")
        except Exception:
            text = content.decode("latin-1", errors="replace")

    if file.filename.lower().endswith(".srt"):
        blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
        clean_lines = []
        for b in blocks:
            lines = b.strip().split("\n")
            content_lines = [l.strip() for l in lines if l.strip() and not re.match(r"^\d+$", l.strip()) and "-->" not in l]
            if content_lines:
                clean_lines.append(" ".join(content_lines))
        clean_text = "\n".join(clean_lines)
    else:
        clean_text = text.strip()

    return {"status": "ok", "text": clean_text, "filename": file.filename}


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
async def synthesize_preview_snippet(req: SynthesisRequest, background_tasks: BackgroundTasks):
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

    try:
        pitch = float(req.pitch)
        pitch = max(-4.0, min(4.0, pitch))
    except (TypeError, ValueError):
        pitch = 0.0

    normalize_text = bool(req.normalize_text)

    voice_info = all_voices[req.voice_id]
    output_filename = OUTPUT_DIR / f"snippet_{uuid.uuid4().hex[:8]}.wav"

    if is_international_voice(req.voice_id):
        try:
            result = await generate_multilingual_speech(
                text=preview_text,
                voice_id=req.voice_id,
                output_path=str(output_filename),
                speed=speed,
                pitch=pitch,
                speaker_boost=req.speaker_boost,
                generate_srt=False,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi tạo audio nghe thử quốc tế: {str(exc)}",
            ) from exc
    else:
        try:
            cloner = get_cloner()
            result = cloner.generate(
                gen_text=preview_text,
                voice_id=req.voice_id if not voice_info["is_clone"] else None,
                ref_audio_path=voice_info["audio_path"] if voice_info["is_clone"] else None,
                ref_text=voice_info.get("ref_text", ""),
                output_path=str(output_filename),
                speed=speed,
                pitch=pitch,
                stability=req.stability,
                similarity=req.similarity,
                speaker_boost=req.speaker_boost,
                temperature=temperature,
                pause_style=pause_style,
                comma_pause=req.comma_pause,
                sentence_pause=req.sentence_pause,
                segment_pause=req.segment_pause,
                strip_special_chars=req.strip_special_chars,
                normalize_text=normalize_text,
                generate_srt=False,
                custom_dict=req.custom_dict,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi tạo audio nghe thử: {str(exc)}",
            ) from exc

    audio_path = result["audio_path"] if isinstance(result, dict) else str(result)
    background_tasks.add_task(cleanup_old_outputs)
    return FileResponse(audio_path, media_type="audio/wav")


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
    print(f"🎙️ VOICE STUDIO PRO SERVER SẴN SÀNG")
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