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

# Các cấu hình ngắt nghỉ và ranh giới câu (tương thích trực tiếp với silence_p và max_chars của VieNeu v3 Turbo)
PAUSE_STYLE_CONFIG = {
    "default": {"silence_p": 0.15, "max_chars": 256},      # Mặc định: Giữ nguyên vẹn 100% nhịp điệu gốc của model
    "natural": {"silence_p": 0.15, "max_chars": 256},      # Tự nhiên, cân bằng
    "relaxed": {"silence_p": 0.40, "max_chars": 130},      # Thư thả, sâu lắng (sách nói, podcast đêm)
    "expressive": {"silence_p": 0.30, "max_chars": 120},   # Diễn cảm, rõ vế (kể chuyện)
    "presentation": {"silence_p": 0.25, "max_chars": 140}, # Dõng dạc, rành mạch (thuyết trình, bài giảng)
    "dramatic": {"silence_p": 0.55, "max_chars": 110},     # Lắng đọng, kịch tính (phim ảnh, tâm trạng)
    "compact": {"silence_p": 0.08, "max_chars": 256},      # Nhanh, liền mạch (tin tức, video ngắn)
    "rapid": {"silence_p": 0.02, "max_chars": 300},        # Siêu tốc, nhịp dồn (quảng cáo xả hàng, tin nóng)
}

PAUSE_REGEX = re.compile(r"\[(?:ngh[iỉ]|pause):([0-9.]+)s?\]", re.IGNORECASE)


def normalize_vietnamese_text(text: str, custom_dict: Optional[Dict[str, str]] = None) -> str:
    """Tự động chuẩn hóa từ viết tắt, số tiền, ngày tháng, đơn vị đo lường, thương hiệu và từ điển tùy chỉnh."""
    if not text:
        return ""

    # 0. Áp dụng từ điển tùy chỉnh nếu người dùng cung cấp
    if custom_dict and isinstance(custom_dict, dict):
        for k, v in custom_dict.items():
            if k and v:
                pattern = r"\b" + re.escape(str(k).strip()) + r"\b"
                text = re.sub(pattern, str(v).strip(), text, flags=re.IGNORECASE)

    # 1. Số điện thoại & Hotline Việt Nam (đọc từng chữ số rõ ràng)
    def _digits_to_vi(match):
        digits = re.sub(r"\D", "", match.group(0))
        d_map = {
            "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
            "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
        }
        return " ".join(d_map.get(d, d) for d in digits)

    text = re.sub(r"\b(1900|1800)\s*(\d{4})\b", r"tổng đài \1 \2", text)
    text = re.sub(r"\b(0[35789]\d{8})\b", _digits_to_vi, text)
    text = re.sub(r"\b(0[35789]\d{1,2}[-.\s]\d{3}[-.\s]\d{3,4})\b", _digits_to_vi, text)

    # 2. Ngày tháng dạng DD/MM/YYYY hoặc DD/MM
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", r"ngày \1 tháng \2 năm \3", text)
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})\b", r"ngày \1 tháng \2", text)

    # 3. Giờ giấc
    text = re.sub(r"\b(\d{1,2})h(\d{1,2})\b", r"\1 giờ \2 phút", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d{1,2})h\b", r"\1 giờ", text, flags=re.IGNORECASE)

    # 4. Đơn vị tiền tệ & số viết tắt
    text = re.sub(r"(\d+)\s*(?:k|K)\b", r"\1 nghìn", text)
    text = re.sub(r"(\d+)\s*(?:tr|TR|trđ|Trđ)\b", r"\1 triệu", text)
    text = re.sub(r"(\d+)[,\.](\d+)\s*(?:tr|TR|trieu|triệu)\b", r"\1 triệu \2", text)
    text = re.sub(r"(\d+)\s*tỷ\b", r"\1 tỷ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*(?:đ|vnđ|vnd|đồng)\b", r"\1 đồng", text, flags=re.IGNORECASE)
    text = re.sub(r"\$(\d+)", r"\1 đô la", text)
    text = re.sub(r"(\d+)\s*\$", r"\1 đô la", text)
    text = re.sub(r"(\d+)\s*%", r"\1 phần trăm", text)

    # 5. Thế kỷ số La Mã
    text = re.sub(r"\bthế kỷ XXI\b", "thế kỷ hai mươi mốt", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthế kỷ XX\b", "thế kỷ hai mươi", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthế kỷ XIX\b", "thế kỷ mười chín", text, flags=re.IGNORECASE)

    # 6. Đơn vị đo lường
    text = re.sub(r"(\d+)\s*km/h\b", r"\1 ki lô mét trên giờ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*km\b", r"\1 ki lô mét", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*(?:m2|m²)\b", r"\1 mét vuông", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*(?:m3|m³)\b", r"\1 mét khối", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*kg\b", r"\1 ki lô gam", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*cm\b", r"\1 xen ti mét", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*mm\b", r"\1 mi li mét", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*(?:gb|GB)\b", r"\1 ghi ga bai", text)
    text = re.sub(r"(\d+)\s*(?:mb|MB)\b", r"\1 mê ga bai", text)

    # 7. Từ viết tắt xưng danh, nghiệp vụ
    abbreviations = [
        (r"\bSTK\b", "số tài khoản"),
        (r"\bSĐT\b", "số điện thoại"),
        (r"\bđ/c\b", "địa chỉ"),
        (r"\bKTS\b", "kiến trúc sư"),
        (r"\bBS\b", "bác sĩ"),
        (r"\bThS\b", "thạc sĩ"),
        (r"\bTS\b", "tiến sĩ"),
        (r"\bPGS\b", "phó giáo sư"),
        (r"\bGS\b", "giáo sư"),
        (r"\bsp\b", "sản phẩm"),
        (r"\bko\b", "không"),
        (r"\bk\b", "không"),
        (r"\bđc\b", "được"),
        (r"\bvs\b", "với"),
        (r"\bib\b", "in-bóc"),
    ]
    for pattern, rep in abbreviations:
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)

    # 8. Từ mượn công nghệ / thương hiệu thông dụng
    brand_map = [
        (r"\b(?:ChatGPT|chatgpt)\b", "chát giê-pi-ti"),
        (r"\b(?:GPT|gpt)\b", "giê-pi-ti"),
        (r"\b(?:AI|ai)\b", "ây-ai"),
        (r"\b(?:FB|fb|Facebook|facebook)\b", "phây-búc"),
        (r"\b(?:Zalo|zalo)\b", "da-lo"),
        (r"\b(?:TikTok|tiktok)\b", "tíc-tốc"),
        (r"\b(?:YouTube|youtube|yt|YT)\b", "diu-túp"),
        (r"\b(?:Website|website|web)\b", "oép-sai"),
        (r"\b(?:Hotline|hotline)\b", "hót-lai"),
        (r"\b(?:CEO|ceo)\b", "xê-e-ô"),
        (r"\b(?:VIP|vip)\b", "víp"),
        (r"\b(?:Email|email|mail)\b", "ê-meo"),
        (r"\b(?:Wifi|wifi)\b", "oai-phai"),
        (r"\b(?:Livestream|livestream)\b", "lai-chim"),
        (r"\b(?:Admin|admin)\b", "át-min"),
        (r"\b(?:Freeship|freeship)\b", "phi-síp"),
        (r"\b(?:Ship|ship)\b", "síp"),
        (r"\b(?:Sale|sale)\b", "seo"),
        (r"\b(?:Review|review)\b", "ri-viu"),
        (r"\b(?:Video|video)\b", "vi-de-ô"),
        (r"\b(?:Link|link)\b", "linh"),
        (r"\b(?:App|app)\b", "áp"),
    ]
    for pattern, replacement in brand_map:
        text = re.sub(pattern, replacement, text)

    return text


def preprocess_natural_text(
    text: str,
    auto_normalize: bool = True,
    custom_dict: Optional[Dict[str, str]] = None,
) -> str:
    """Chuẩn hóa văn bản, hỗ trợ thẻ lấy hơi, nhịp thở và khoảng ngắt nghỉ tự nhiên."""
    if not text:
        return ""

    if auto_normalize:
        text = normalize_vietnamese_text(text, custom_dict=custom_dict)

    # 1. Chuyển đổi các thẻ lấy hơi / thở tự nhiên sang [thở dài] (emotion token 2 của VieNeu sinh âm thanh hít thở thật)
    text = re.sub(
        r"\[(thở|lay hoi|lấy hơi|hít thở|hit tho|tho|breathe|hơi|thở phào)\]",
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


def split_text_into_chunks(text: str, max_words: int = 35) -> List[str]:
    """Chia văn bản thành các câu hoặc nhóm câu nhỏ gọn (~20-35 từ) để mô hình không bị quá tải trên bài dài."""
    clean = text.strip()
    if not clean:
        return []

    # Chia theo dấu kết thúc câu hoặc xuống dòng
    raw_sentences = re.split(r"(?<=[.!?…\n])\s+", clean)
    chunks = []
    current_chunk = []
    current_words = 0

    for sent in raw_sentences:
        s = sent.strip()
        if not s:
            continue
        words_in_s = len(s.split())
        if current_words > 0 and (current_words + words_in_s > max_words):
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_words = words_in_s
        else:
            current_chunk.append(s)
            current_words += words_in_s

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [clean]


def wsola_time_stretch(audio: np.ndarray, rate: float, sr: int = 48000) -> np.ndarray:
    """Co giãn thời gian bằng WSOLA trong miền thời gian (Waveform Similarity Overlap-Add).
    Khắc phục triệt để hiện tượng ồm ồm, méo tiếng hoặc vang hẫng của Phase Vocoder.
    Giữ nguyên 100% âm sắc, formants và độ tròn vành rõ chữ của giọng AI.
    """
    if abs(rate - 1.0) < 0.02 or len(audio) == 0:
        return audio

    rate = float(max(0.65, min(1.75, rate)))
    win_size = int(0.025 * sr)
    if win_size % 2 != 0:
        win_size += 1
    syn_hop = win_size // 2
    ana_hop = int(round(syn_hop * rate))
    delta = int(0.007 * sr)

    window = np.hanning(win_size).astype(np.float32)
    total_samples = len(audio)

    if total_samples < win_size + delta * 2:
        return audio

    est_out_len = int(total_samples / rate) + win_size * 2
    out_audio = np.zeros(est_out_len, dtype=np.float32)
    norm_window = np.zeros(est_out_len, dtype=np.float32)

    # Khởi tạo frame đầu
    out_audio[0:win_size] += audio[0:win_size] * window
    norm_window[0:win_size] += window

    out_pos = syn_hop
    in_pos = ana_hop
    prev_target_pos = syn_hop

    from scipy.signal import correlate

    while (in_pos + win_size + delta < total_samples) and (out_pos + win_size < est_out_len):
        search_start = max(0, in_pos - delta)
        search_end = min(total_samples, in_pos + delta + win_size)

        if search_end - search_start < win_size:
            chosen_seg = audio[in_pos : in_pos + win_size]
            prev_target_pos = in_pos + syn_hop
        else:
            search_region = audio[search_start:search_end]
            ref_seg = audio[prev_target_pos : prev_target_pos + win_size]

            if len(ref_seg) == win_size and len(search_region) >= win_size:
                corr = correlate(search_region, ref_seg, mode="valid")
                best_idx = int(np.argmax(corr))
                chosen_start = search_start + best_idx
                chosen_seg = audio[chosen_start : chosen_start + win_size]
                prev_target_pos = chosen_start + syn_hop
            else:
                chosen_seg = audio[in_pos : in_pos + win_size]
                prev_target_pos = in_pos + syn_hop

        out_audio[out_pos : out_pos + win_size] += chosen_seg * window
        norm_window[out_pos : out_pos + win_size] += window

        out_pos += syn_hop
        in_pos += ana_hop

    valid_len = out_pos + win_size
    mask = norm_window[:valid_len] > 1e-4
    out_audio[:valid_len][mask] /= norm_window[:valid_len][mask]

    return out_audio[:valid_len]


def clean_pitch_shift(audio: np.ndarray, n_steps: float, sr: int = 48000) -> np.ndarray:
    """Dịch cao độ giọng nói mà KHÔNG gây méo tiếng hay ồm ồm như phase vocoder."""
    if abs(n_steps) < 0.25 or len(audio) == 0:
        return audio

    n_steps = float(max(-3.5, min(3.5, n_steps)))
    factor = 2.0 ** (n_steps / 12.0)

    try:
        import librosa
        target_sr = int(round(sr / factor))
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        shifted = wsola_time_stretch(resampled, rate=factor, sr=sr)
    except Exception as e:
        print(f"⚠️ Không thể dịch cao độ giọng: {e}")
        shifted = audio

    # Khử dải ù trầm nếu hạ tông (loại bỏ tiếng ồm ở 80Hz - 250Hz)
    if n_steps < 0 and len(shifted) > 0:
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(2, 80, btype="highpass", fs=sr, output="sos")
            shifted = sosfilt(sos, shifted).astype(np.float32)
        except Exception:
            pass

    return shifted


class VoiceCloner:
    def __init__(self, threads: int = 4):
        self.device = "cpu (ONNX Runtime)"
        self._tts = Vieneu(mode="v3turbo", backend="onnx", threads=threads)
        self.sample_rate = getattr(self._tts, "sample_rate", 48000)
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

    def register_custom_voice(
        self,
        voice_id: str,
        audio_path: str,
        description: str = "",
        gender: str = "",
    ) -> str:
        """Đăng ký giọng clone mới từ file âm thanh người dùng tải lên."""
        audio_file = Path(audio_path).resolve()
        if not audio_file.is_file():
            raise FileNotFoundError(f"Không tìm thấy file audio: {audio_file}")

        voice_name = f"clone-{voice_id}"
        with self._generation_lock:
            self._tts.add_voice(
                name=voice_name,
                ref_audio=audio_file,
                denoise=True,
                use_ref_codes=True,
                description=description,
                gender=gender,
                save=False,
            )
            self._registered_references[voice_name] = audio_file
        return voice_name

    def remove_custom_voice(self, voice_id: str) -> None:
        """Hủy đăng ký giọng clone."""
        voice_name = f"clone-{voice_id}"
        with self._generation_lock:
            if hasattr(self._tts, "remove_voice"):
                self._tts.remove_voice(voice_name)
            self._registered_references.pop(voice_name, None)

    def _infer_segment(
        self,
        segment_text: str,
        voice_name: str,
        temperature: float,
        dynamic_max_frames: int,
        silence_p: float = 0.15,
        max_chars: int = 256,
        auto_normalize: bool = True,
        custom_dict: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Sinh audio cho một phân đoạn văn bản."""
        clean_text = preprocess_natural_text(
            segment_text, auto_normalize=auto_normalize, custom_dict=custom_dict
        )
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
            max_chars=max_chars,
            silence_p=silence_p,
        )
        return audio if audio is not None else np.array([], dtype=np.float32)

    def _infer_block_smart(
        self,
        block_text: str,
        voice_name: str,
        temperature: float,
        silence_p: float = 0.15,
        max_chars: int = 256,
        auto_normalize: bool = True,
        custom_dict: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Xử lý khối chữ: nếu ngắn thì infer trực tiếp, nếu dài thì chia câu thông minh để audio không bao giờ bị cắt ngắn."""
        words = block_text.strip().split()
        if len(words) <= 40:
            frame_cap = max(350, min(2200, len(words) * 50))
            return self._infer_segment(
                block_text, voice_name, temperature, frame_cap,
                silence_p=silence_p, max_chars=max_chars,
                auto_normalize=auto_normalize, custom_dict=custom_dict,
            )

        chunks = split_text_into_chunks(block_text, max_words=32)
        sub_audios: List[np.ndarray] = []
        silence_samples = int(0.18 * self.sample_rate)
        gap_zeros = np.zeros(silence_samples, dtype=np.float32)

        for chunk in chunks:
            c_words = len(chunk.split())
            frame_cap = max(350, min(1800, c_words * 50))
            audio_part = self._infer_segment(
                chunk, voice_name, temperature, frame_cap,
                silence_p=silence_p, max_chars=max_chars,
                auto_normalize=auto_normalize, custom_dict=custom_dict,
            )
            if len(audio_part) > 0:
                if sub_audios:
                    sub_audios.append(gap_zeros)
                sub_audios.append(audio_part)

        if not sub_audios:
            return np.array([], dtype=np.float32)
        return np.concatenate(sub_audios)

    def generate(
        self,
        gen_text: str,
        voice_id: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: str = "",
        output_path: str = "output.wav",
        speed: float = 1.0,
        pitch: float = 0.0,
        temperature: Any = "default",
        pause_style: str = "default",
        normalize_text: bool = True,
        custom_dict: Optional[Dict[str, str]] = None,
    ) -> str:
        if not gen_text or not gen_text.strip():
            raise ValueError("Nội dung văn bản không được để trống.")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        pause_config = PAUSE_STYLE_CONFIG.get(pause_style, PAUSE_STYLE_CONFIG["default"])
        silence_p = pause_config["silence_p"]
        max_chars = pause_config["max_chars"]

        with self._generation_lock:
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
            # 2. Trường hợp sử dụng giọng mẫu có sẵn (Preset Voice hoặc đã đăng ký)
            elif voice_id:
                resolved = self._tts.resolve_voice_name(voice_id) or voice_id
                if resolved not in self._tts._preset_voices:
                    # Kiểm tra xem có phải tên clone không
                    clone_candidate = f"clone-{voice_id}"
                    if clone_candidate in self._tts._preset_voices:
                        resolved = clone_candidate
                    else:
                        raise ValueError(f"Không tìm thấy giọng đọc '{voice_id}' trong hệ thống.")
                voice_name = resolved
            else:
                voice_name = self._tts._default_voice or "Hải Đăng"

            # Độ biểu cảm: nếu là mặc định ("default" hoặc None) thì giữ nguyên 0.8 của model gốc
            if temperature is None or temperature == "default":
                temp = 0.8
            else:
                try:
                    temp = float(temperature)
                    temp = max(0.3, min(1.2, temp))
                except (TypeError, ValueError):
                    temp = 0.8

            # Xử lý các thẻ dừng chủ động [nghỉ:X s] hoặc [pause:X s] nếu có
            if PAUSE_REGEX.search(gen_text):
                parts = PAUSE_REGEX.split(gen_text)
                audio_segments: List[np.ndarray] = []
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        # Phân đoạn chữ
                        if part.strip():
                            seg_audio = self._infer_block_smart(
                                part, voice_name, temp,
                                silence_p=silence_p, max_chars=max_chars,
                                auto_normalize=normalize_text,
                                custom_dict=custom_dict,
                            )
                            if len(seg_audio) > 0:
                                audio_segments.append(seg_audio)
                    else:
                        # Thời gian nghỉ tính bằng giây
                        try:
                            pause_sec = float(part)
                            pause_samples = int(pause_sec * self.sample_rate)
                            if pause_samples > 0:
                                audio_segments.append(np.zeros(pause_samples, dtype=np.float32))
                        except Exception:
                            pass

                if not audio_segments:
                    raise RuntimeError("Mô hình không tạo được audio từ văn bản.")
                audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
            else:
                # Xử lý toàn văn với khối chữ thông minh
                audio = self._infer_block_smart(
                    gen_text, voice_name, temp,
                    silence_p=silence_p, max_chars=max_chars,
                    auto_normalize=normalize_text,
                    custom_dict=custom_dict,
                )
                if audio is None or len(audio) == 0:
                    raise RuntimeError("Mô hình không tạo được audio từ văn bản đã cung cấp.")

            # 1. Điều chỉnh tốc độ đọc (speed) bằng WSOLA miền thời gian (100% tròn tiếng, không bị ồm)
            try:
                target_speed = float(speed)
            except (TypeError, ValueError):
                target_speed = 1.0

            if abs(target_speed - 1.0) >= 0.02:
                try:
                    audio = wsola_time_stretch(audio, rate=target_speed, sr=self.sample_rate)
                except Exception as stretch_err:
                    print(f"⚠️ Không thể co giãn tốc độ: {stretch_err}")

            # 2. Điều chỉnh cao độ (pitch) bằng Resample + WSOLA (không dùng phase vocoder)
            try:
                pitch_steps = float(pitch)
            except (TypeError, ValueError):
                pitch_steps = 0.0

            if abs(pitch_steps) >= 0.25:
                try:
                    audio = clean_pitch_shift(audio, n_steps=pitch_steps, sr=self.sample_rate)
                except Exception as pitch_err:
                    print(f"⚠️ Không thể dịch cao độ: {pitch_err}")

            # 3. Khử tiếng ồm (de-boom), lọc DC offset & chuẩn hóa âm lượng chuẩn phát thanh
            if len(audio) > 0:
                audio = audio - np.mean(audio)
                try:
                    from scipy.signal import butter, sosfilt
                    # Highpass 65Hz khử sub-bass rumble, tiếng rung micro
                    sos = butter(2, 65, btype="highpass", fs=self.sample_rate, output="sos")
                    audio = sosfilt(sos, audio).astype(np.float32)
                except Exception:
                    pass

                max_amp = float(np.max(np.abs(audio)))
                if max_amp > 1e-4:
                    target_peak = 0.92
                    audio = audio * (target_peak / max_amp)

            self._tts.save(audio, output_file)

        return str(output_file)
