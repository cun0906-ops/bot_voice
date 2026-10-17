import asyncio
import io
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure edge-tts is installed
try:
    import edge_tts  # type: ignore
except ImportError:
    import subprocess
    try:
        print("📦 Đang tự động cài đặt thư viện edge-tts cho giọng đọc quốc tế...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "edge-tts"])
        import edge_tts  # type: ignore
        print("✅ Đã cài đặt edge-tts thành công!")
    except Exception as exc:
        print(f"⚠️ Chưa thể cài đặt edge-tts tự động: {exc}")
        edge_tts = None


CURATED_INTERNATIONAL_VOICES = [
    # Tiếng Anh (Mỹ)
    {"id": "en-US-JennyNeural", "name": "Jenny (Nữ)", "lang": "en-US", "lang_name": "🇺🇸 Tiếng Anh (Mỹ)", "gender": "female", "description": "Nữ · Trẻ trung, tự nhiên, truyền cảm"},
    {"id": "en-US-GuyNeural", "name": "Guy (Nam)", "lang": "en-US", "lang_name": "🇺🇸 Tiếng Anh (Mỹ)", "gender": "male", "description": "Nam · Trầm ấm, dõng dạc, phát thanh viên"},
    {"id": "en-US-AriaNeural", "name": "Aria (Nữ)", "lang": "en-US", "lang_name": "🇺🇸 Tiếng Anh (Mỹ)", "gender": "female", "description": "Nữ · Tự tin, thời sự, chuyên nghiệp"},
    {"id": "en-US-ChristopherNeural", "name": "Christopher (Nam)", "lang": "en-US", "lang_name": "🇺🇸 Tiếng Anh (Mỹ)", "gender": "male", "description": "Nam · Đĩnh đạc, bản tin, giảng dạy"},
    {"id": "en-US-EricNeural", "name": "Eric (Nam)", "lang": "en-US", "lang_name": "🇺🇸 Tiếng Anh (Mỹ)", "gender": "male", "description": "Nam · Trẻ trung, năng động, quảng cáo"},
    {"id": "en-US-AnaNeural", "name": "Ana (Bé gái)", "lang": "en-US", "lang_name": "🇺🇸 Tiếng Anh (Mỹ)", "gender": "female", "description": "Nữ · Giọng trẻ em dễ thương, phim hoạt hình"},

    # Tiếng Anh (Anh)
    {"id": "en-GB-SoniaNeural", "name": "Sonia (Nữ)", "lang": "en-GB", "lang_name": "🇬🇧 Tiếng Anh (Anh)", "gender": "female", "description": "Nữ · Giọng London quý phái, chuẩn BBC"},
    {"id": "en-GB-RyanNeural", "name": "Ryan (Nam)", "lang": "en-GB", "lang_name": "🇬🇧 Tiếng Anh (Anh)", "gender": "male", "description": "Nam · Giọng Anh truyền thống, lịch lãm"},
    {"id": "en-GB-LibbyNeural", "name": "Libby (Nữ)", "lang": "en-GB", "lang_name": "🇬🇧 Tiếng Anh (Anh)", "gender": "female", "description": "Nữ · Nhẹ nhàng, tự sự, audiobook"},

    # Tiếng Trung (Mandarin)
    {"id": "zh-CN-XiaoxiaoNeural", "name": "Hiểu Hiểu - Xiaoxiao (Nữ)", "lang": "zh-CN", "lang_name": "🇨🇳 Tiếng Trung (Phổ thông)", "gender": "female", "description": "Nữ · Giọng Bắc Kinh chuẩn, ấm áp, dịu dàng"},
    {"id": "zh-CN-YunxiNeural", "name": "Vân Hy - Yunxi (Nam)", "lang": "zh-CN", "lang_name": "🇨🇳 Tiếng Trung (Phổ thông)", "gender": "male", "description": "Nam · Giọng trẻ trung, sôi động, review, podcast"},
    {"id": "zh-CN-YunjianNeural", "name": "Vân Kiện - Yunjian (Nam)", "lang": "zh-CN", "lang_name": "🇨🇳 Tiếng Trung (Phổ thông)", "gender": "male", "description": "Nam · Kịch tính, giọng đọc phim kiếm hiệp, truyện kiếm hiệp"},
    {"id": "zh-CN-XiaoyiNeural", "name": "Hiểu Nghệ - Xiaoyi (Nữ)", "lang": "zh-CN", "lang_name": "🇨🇳 Tiếng Trung (Phổ thông)", "gender": "female", "description": "Nữ · Thanh lịch, đọc tin tức, thương mại"},

    # Tiếng Nhật (Japanese)
    {"id": "ja-JP-NanamiNeural", "name": "Nanami (Nữ)", "lang": "ja-JP", "lang_name": "🇯🇵 Tiếng Nhật", "gender": "female", "description": "Nữ · Trong trẻo, anime, podcast dễ thương"},
    {"id": "ja-JP-KeitaNeural", "name": "Keita (Nam)", "lang": "ja-JP", "lang_name": "🇯🇵 Tiếng Nhật", "gender": "male", "description": "Nam · Chuẩn đài NHK, điềm đạm, tin tức"},
    {"id": "ja-JP-AoiNeural", "name": "Aoi (Nữ)", "lang": "ja-JP", "lang_name": "🇯🇵 Tiếng Nhật", "gender": "female", "description": "Nữ · Dịu dàng, kể chuyện, thuyết minh"},

    # Tiếng Hàn (Korean)
    {"id": "ko-KR-SunHiNeural", "name": "Sun-Hi (Nữ)", "lang": "ko-KR", "lang_name": "🇰🇷 Tiếng Hàn", "gender": "female", "description": "Nữ · Ngọt ngào, chuẩn giọng phim truyền hình Hàn Quốc"},
    {"id": "ko-KR-InJoonNeural", "name": "In-Joon (Nam)", "lang": "ko-KR", "lang_name": "🇰🇷 Tiếng Hàn", "gender": "male", "description": "Nam · Ấm áp, giọng MC, phóng sự"},
    {"id": "ko-KR-HyunsuNeural", "name": "Hyunsu (Nam)", "lang": "ko-KR", "lang_name": "🇰🇷 Tiếng Hàn", "gender": "male", "description": "Nam · Trẻ trung, tự nhiên"},

    # Tiếng Pháp (French)
    {"id": "fr-FR-DeniseNeural", "name": "Denise (Nữ)", "lang": "fr-FR", "lang_name": "🇫🇷 Tiếng Pháp", "gender": "female", "description": "Nữ · Chuẩn Paris, thanh lịch, lãng mạn"},
    {"id": "fr-FR-HenriNeural", "name": "Henri (Nam)", "lang": "fr-FR", "lang_name": "🇫🇷 Tiếng Pháp", "gender": "male", "description": "Nam · Trầm, sang trọng, thuyết minh"},

    # Tiếng Đức (German)
    {"id": "de-DE-KatjaNeural", "name": "Katja (Nữ)", "lang": "de-DE", "lang_name": "🇩🇪 Tiếng Đức", "gender": "female", "description": "Nữ · Chuẩn Berlin, rõ ràng, dứt khoát"},
    {"id": "de-DE-KillianNeural", "name": "Killian (Nam)", "lang": "de-DE", "lang_name": "🇩🇪 Tiếng Đức", "gender": "male", "description": "Nam · Chuyên nghiệp, tin tức"},

    # Tây Ban Nha (Spanish)
    {"id": "es-ES-ElviraNeural", "name": "Elvira (Nữ)", "lang": "es-ES", "lang_name": "🇪🇸 Tây Ban Nha", "gender": "female", "description": "Nữ · Chuẩn Madrid, giàu năng lượng"},
    {"id": "es-ES-AlvaroNeural", "name": "Alvaro (Nam)", "lang": "es-ES", "lang_name": "🇪🇸 Tây Ban Nha", "gender": "male", "description": "Nam · Trầm ấm, tự nhiên"},

    # Tiếng Nga (Russian)
    {"id": "ru-RU-SvetlanaNeural", "name": "Svetlana (Nữ)", "lang": "ru-RU", "lang_name": "🇷🇺 Tiếng Nga", "gender": "female", "description": "Nữ · Chuẩn Moscow, truyền cảm"},
    {"id": "ru-RU-DmitryNeural", "name": "Dmitry (Nam)", "lang": "ru-RU", "lang_name": "🇷🇺 Tiếng Nga", "gender": "male", "description": "Nam · Vang, dõng dạc"},

    # Tiếng Thái (Thai)
    {"id": "th-TH-PremwadeeNeural", "name": "Premwadee (Nữ)", "lang": "th-TH", "lang_name": "🇹🇭 Tiếng Thái", "gender": "female", "description": "Nữ · Ngọt ngào, nhẹ nhàng"},
    {"id": "th-TH-NiwatNeural", "name": "Niwat (Nam)", "lang": "th-TH", "lang_name": "🇹🇭 Tiếng Thái", "gender": "male", "description": "Nam · Tự nhiên, tin tức"},
]

VOICE_MAP: Dict[str, Dict[str, Any]] = {v["id"]: v for v in CURATED_INTERNATIONAL_VOICES}


def get_international_languages() -> List[Dict[str, str]]:
    """Lấy danh sách các ngôn ngữ quốc tế khả dụng."""
    seen = {}
    result = []
    for v in CURATED_INTERNATIONAL_VOICES:
        code = v["lang"]
        if code not in seen:
            seen[code] = True
            result.append({"code": code, "name": v["lang_name"]})
    return result


def get_international_voices(lang_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lấy danh sách các giọng quốc tế (có thể lọc theo mã ngôn ngữ)."""
    if not lang_code:
        return CURATED_INTERNATIONAL_VOICES
    return [v for v in CURATED_INTERNATIONAL_VOICES if v["lang"] == lang_code]


def is_international_voice(voice_id: str) -> bool:
    """Kiểm tra xem voice_id có thuộc về hệ thống giọng quốc tế hay không."""
    return voice_id in VOICE_MAP or bool(re.match(r"^[a-z]{2,3}-[A-Z]{2}-", voice_id))


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


async def generate_multilingual_speech(
    text: str,
    voice_id: str = "en-US-JennyNeural",
    output_path: str = "output.wav",
    output_srt_path: Optional[str] = None,
    speed: float = 1.0,
    pitch: float = 0.0,
    speaker_boost: bool = True,
    generate_srt: bool = True,
) -> Dict[str, Any]:
    """Tạo âm thanh đa ngôn ngữ với giọng đọc bản xứ chuẩn quốc tế qua Edge-TTS."""
    if not edge_tts:
        raise RuntimeError("Thư viện edge-tts chưa được cài đặt. Vui lòng kiểm tra lại môi trường.")

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Nội dung văn bản không được để trống.")

    # Tính toán rate và pitch cho SSML Edge-TTS
    # speed: 1.0 -> "+0%"; 1.2 -> "+20%"; 0.8 -> "-20%"
    speed_percent = int(round((speed - 1.0) * 100))
    rate_str = f"{'+' if speed_percent >= 0 else ''}{speed_percent}%"

    # pitch: 0.0 -> "+0Hz"; +2.0 -> "+20Hz"; -2.0 -> "-20Hz"
    pitch_val = int(round(pitch * 10))
    pitch_str = f"{'+' if pitch_val >= 0 else ''}{pitch_val}Hz"

    communicate = edge_tts.Communicate(
        text=clean_text,
        voice=voice_id,
        rate=rate_str,
        pitch=pitch_str,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_mp3 = output_file.with_suffix(".temp.mp3")

    cues: List[Dict[str, Any]] = []
    cue_idx = 1
    current_cue_words: List[str] = []
    cue_start_sec = 0.0
    cue_end_sec = 0.0

    # Lắng nghe stream âm thanh và word boundaries để tạo phụ đề chính xác
    with open(temp_mp3, "wb") as f_mp3:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f_mp3.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # offset và duration tính bằng 100 nano-giây (1 tick = 1e-7 s)
                offset_sec = chunk["offset"] / 10_000_000.0
                duration_sec = chunk["duration"] / 10_000_000.0
                word = chunk["text"]

                if not current_cue_words:
                    cue_start_sec = offset_sec
                current_cue_words.append(word)
                cue_end_sec = offset_sec + duration_sec

                # Ngắt phụ đề nếu gặp dấu câu hoặc gom đủ 8-12 từ
                is_punc = bool(re.search(r"[.!?…\n;]$", word))
                if is_punc or len(current_cue_words) >= 10:
                    cues.append({
                        "id": cue_idx,
                        "start": round(cue_start_sec, 3),
                        "end": round(cue_end_sec, 3),
                        "text": " ".join(current_cue_words),
                    })
                    cue_idx += 1
                    current_cue_words = []

    if current_cue_words:
        cues.append({
            "id": cue_idx,
            "start": round(cue_start_sec, 3),
            "end": round(cue_end_sec, 3),
            "text": " ".join(current_cue_words),
        })

    # Chuyển đổi MP3 sang file WAV 48kHz High-Res PCM 16-bit chuẩn chỉnh
    import soundfile as sf
    import librosa
    import numpy as np

    try:
        wav, sr = librosa.load(str(temp_mp3), sr=48000, mono=True)
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(temp_mp3))
        seg = seg.set_frame_rate(48000).set_channels(1)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)
        wav, sr = sf.read(buf, dtype="float32")

    # Speaker Boost & Khử rung
    if len(wav) > 0:
        wav = wav - np.mean(wav)
        if speaker_boost:
            try:
                from scipy.signal import butter, sosfilt
                sos = butter(2, 65, btype="highpass", fs=48000, output="sos")
                wav = sosfilt(sos, wav).astype(np.float32)
            except Exception:
                pass

        target_peak = 0.95 if speaker_boost else 0.90
        max_amp = float(np.max(np.abs(wav)))
        if max_amp > 1e-4:
            wav = wav * (target_peak / max_amp)

    sf.write(str(output_file), wav, 48000, subtype="PCM_16")

    # Xóa file MP3 tạm
    if temp_mp3.exists():
        temp_mp3.unlink(missing_ok=True)

    total_duration = round(len(wav) / 48000.0, 2)

    # Lưu file .SRT
    srt_path_saved = None
    if generate_srt:
        if output_srt_path is None:
            output_srt_path = str(output_file.with_suffix(".srt"))

        if not cues:
            # Fallback nếu không bắt được WordBoundary
            cues = [{
                "id": 1,
                "start": 0.0,
                "end": total_duration,
                "text": clean_text,
            }]

        srt_lines = []
        for c in cues:
            srt_lines.append(f"{c['id']}")
            srt_lines.append(f"{format_srt_time(c['start'])} --> {format_srt_time(c['end'])}")
            srt_lines.append(c["text"])
            srt_lines.append("")

        srt_content = "\n".join(srt_lines)
        srt_file = Path(output_srt_path)
        srt_file.parent.mkdir(parents=True, exist_ok=True)
        srt_file.write_text(srt_content, encoding="utf-8")
        srt_path_saved = str(srt_file)

    return {
        "audio_path": str(output_file),
        "srt_path": srt_path_saved,
        "duration": total_duration,
        "cues": cues,
    }
