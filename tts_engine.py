import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

MODEL_CACHE_DIR = Path(__file__).resolve().parent / ".model_vieneu_cache"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

from vieneu import Vieneu
import vieneu_utils.core_utils as core_utils

# Các cấu hình khoảng nghỉ tự nhiên theo ngữ cảnh
PAUSE_PROFILES = {
    "natural": {"para": 0.70, "sentence": 0.50, "minor": 0.30},       # Mặc định: Tự nhiên, đàm thoại chuẩn
    "relaxed": {"para": 0.95, "sentence": 0.75, "minor": 0.45},       # Thư thả, sâu lắng (sách nói, podcast)
    "expressive": {"para": 0.85, "sentence": 0.60, "minor": 0.35},    # Diễn cảm, rõ ràng từng vế (kể chuyện)
    "compact": {"para": 0.45, "sentence": 0.30, "minor": 0.18},       # Nhanh gọn, liền mạch (tin tức, video ngắn)
    "dramatic": {"para": 1.25, "sentence": 0.95, "minor": 0.55},      # Lắng đọng, kịch tính (phim ảnh, tự sự tâm trạng)
    "presentation": {"para": 0.75, "sentence": 0.55, "minor": 0.38},  # Dõng dạc, rành mạch (thuyết trình, bài giảng)
    "rapid": {"para": 0.35, "sentence": 0.22, "minor": 0.12},         # Nhịp dồn, siêu tốc (quảng cáo xả hàng, tin nóng)
}

PAUSE_REGEX = re.compile(r"\[(?:ngh[iỉ]|pause):([0-9.]+)s?\]", re.IGNORECASE)


def preprocess_natural_text(text: str) -> str:
    """Chuẩn hóa văn bản, hỗ trợ thẻ lấy hơi, nhịp thở và khoảng ngắt nghỉ tự nhiên."""
    if not text:
        return ""

    # 1. Chuyển đổi các thẻ lấy hơi / thở tự nhiên sang [thở dài] (emotion token 2 của VieNeu sinh âm thanh hít thở thật)
    text = re.sub(
        r"\[(thở|lay hoi|lấy hơi|hít thở|hit tho|tho|breathe|hơi)\]",
        "[thở dài]",
        text,
        flags=re.IGNORECASE,
    )

    # 2. Chuyển đổi các thẻ nghỉ không kèm số giây sang dấu ngắt nhịp tự nhiên
    text = re.sub(r"\[(nghỉ ngắn|nghi ngan|short pause)\]", " ... ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(nghỉ dài|nghi dai|long pause)\]", " .\n\n", text, flags=re.IGNORECASE)

    # 3. Chuẩn hóa gạch ngang và dấu ba chấm để AI tạo khoảng ngập ngừng tự nhiên
    text = re.sub(r"\s*--\s*", " — ", text)
    text = re.sub(r"\.{4,}", "...", text)

    return text.strip()


class VoiceCloner:
    def __init__(self, threads: int = 4):
        self.device = "cpu (ONNX Runtime)"
        self._tts = Vieneu(mode="v3turbo", backend="onnx", threads=threads)
        self._registered_references = {}
        self._generation_lock = threading.Lock()

    def get_preset_voices(self) -> Dict[str, Dict[str, Any]]:
        """Lấy danh mục các giọng mẫu có sẵn của VieNeu v3 Turbo kèm metadata."""
        voices = {}
        for name, data in self._tts._preset_voices.items():
            voices[name] = {
                "name": name,
                "description": data.get("description", ""),
                "gender": data.get("gender", ""),
                "region": data.get("region", ""),
                "style": data.get("style", "tu_nhien"),
                "is_clone": False,
            }
        return voices

    def _infer_segment(
        self,
        segment_text: str,
        voice_name: str,
        temperature: float,
        dynamic_max_frames: int,
    ) -> np.ndarray:
        """Sinh audio cho một phân đoạn văn bản với ranh giới câu ngắt nghỉ tự nhiên (max_chars=130)."""
        clean_text = preprocess_natural_text(segment_text)
        if not clean_text:
            return np.array([], dtype=np.float32)

        audio = self._tts.infer(
            clean_text,
            voice=voice_name,
            temperature=temperature,
            top_k=25,
            top_p=0.95,
            repetition_penalty=1.2,
            max_new_frames=dynamic_max_frames,
            max_chars=130,  # Chia nhỏ câu để luôn có khoảng nghỉ thở tự nhiên giữa các câu
        )
        return audio if audio is not None else np.array([], dtype=np.float32)

    def generate(
        self,
        gen_text: str,
        voice_id: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: str = "",
        output_path: str = "output.wav",
        speed: float = 1.0,
        temperature: float = 0.8,
        pause_style: str = "natural",
    ) -> str:
        if not gen_text or not gen_text.strip():
            raise ValueError("Nội dung văn bản không được để trống.")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with self._generation_lock:
            # Cập nhật cấu hình ngắt nghỉ tự nhiên theo profile được chọn
            profile = PAUSE_PROFILES.get(pause_style, PAUSE_PROFILES["natural"])
            core_utils.V3_GAP_SILENCE.update(profile)

            # 1. Trường hợp sử dụng file âm thanh tham chiếu (Custom Voice Clone)
            if ref_audio_path:
                reference_path = Path(ref_audio_path).resolve()
                if not reference_path.is_file():
                    raise FileNotFoundError(f"Không tìm thấy audio tham chiếu: {reference_path}")

                voice_name = f"clone-{reference_path.stem}"
                if self._registered_references.get(voice_name) != reference_path:
                    self._tts.add_voice(
                        voice_name,
                        reference_path,
                        denoise=True,
                        use_ref_codes=True,
                        save=False,
                    )
                    self._registered_references[voice_name] = reference_path
            # 2. Trường hợp sử dụng giọng mẫu có sẵn (Preset Voice)
            elif voice_id:
                resolved = self._tts.resolve_voice_name(voice_id) or voice_id
                if resolved not in self._tts._preset_voices:
                    raise ValueError(f"Không tìm thấy giọng mẫu '{voice_id}' trong hệ thống.")
                voice_name = resolved
            else:
                voice_name = self._tts._default_voice or "Hải Đăng"

            # Tính toán frame trần an toàn
            word_count = len(gen_text.strip().split())
            dynamic_max_frames = max(450, min(3500, word_count * 45))

            # Temperature chuẩn 0.75 - 0.85 giúp ngữ điệu trầm bổng, biểu cảm và có hồn
            temp = float(temperature) if temperature is not None else 0.8
            temp = max(0.3, min(1.2, temp))

            # Xử lý các thẻ dừng chủ động [nghỉ:X s] hoặc [pause:X s] nếu có
            if PAUSE_REGEX.search(gen_text):
                parts = PAUSE_REGEX.split(gen_text)
                audio_segments: List[np.ndarray] = []
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        # Phân đoạn chữ
                        if part.strip():
                            seg_audio = self._infer_segment(part, voice_name, temp, dynamic_max_frames)
                            if len(seg_audio) > 0:
                                audio_segments.append(seg_audio)
                    else:
                        # Thời gian nghỉ tính bằng giây
                        try:
                            pause_sec = float(part)
                            pause_samples = int(pause_sec * self._tts.sample_rate)
                            if pause_samples > 0:
                                audio_segments.append(np.zeros(pause_samples, dtype=np.float32))
                        except Exception:
                            pass

                if not audio_segments:
                    raise RuntimeError("Mô hình không tạo được audio từ văn bản.")
                audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
            else:
                # Xử lý toàn văn với ngắt nghỉ tự nhiên
                audio = self._infer_segment(gen_text, voice_name, temp, dynamic_max_frames)
                if audio is None or len(audio) == 0:
                    raise RuntimeError("Mô hình không tạo được audio từ văn bản đã cung cấp.")

            # Hỗ trợ điều chỉnh tốc độ đọc (speed) giữ nguyên cao độ (pitch)
            try:
                target_speed = float(speed)
            except (TypeError, ValueError):
                target_speed = 1.0

            if abs(target_speed - 1.0) >= 0.03:
                clamped_speed = max(0.65, min(1.75, target_speed))
                try:
                    import librosa
                    audio = librosa.effects.time_stretch(
                        audio.astype(np.float32),
                        rate=clamped_speed,
                    )
                except Exception as stretch_err:
                    print(f"⚠️ Không thể co giãn tốc độ: {stretch_err}")

            self._tts.save(audio, output_file)

        return str(output_file)



