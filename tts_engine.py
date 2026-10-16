import os
import threading
from pathlib import Path

MODEL_CACHE_DIR = Path(__file__).resolve().parent / ".model_vieneu_cache"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

from vieneu import Vieneu


class VoiceCloner:
    def __init__(self, device=None):
        self.device = "cpu (ONNX Runtime)"
        self._tts = Vieneu(mode="v3turbo", backend="onnx", threads=4)
        self._registered_references = {}
        self._generation_lock = threading.Lock()

    def generate(
        self,
        gen_text: str,
        ref_audio_path: str,
        ref_text: str = "",
        output_path: str = "output.wav",
        speed: float = 1.0,
    ) -> str:
        if not gen_text or not gen_text.strip():
            raise ValueError("gen_text không được rỗng")
        if speed != 1.0:
            raise ValueError("Model VieNeu hiện chỉ hỗ trợ tốc độ mặc định 1.0.")

        reference_path = Path(ref_audio_path).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy audio tham chiếu: {reference_path}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        voice_name = f"clone-{reference_path.stem}"

        with self._generation_lock:
            if self._registered_references.get(voice_name) != reference_path:
                self._tts.add_voice(
                    voice_name,
                    reference_path,
                    denoise=True,
                    use_ref_codes=True,
                    save=False,
                )
                self._registered_references[voice_name] = reference_path

            audio = self._tts.infer(
                gen_text.strip(),
                voice=voice_name,
                temperature=0.4,
                top_k=25,
                top_p=0.95,
                max_new_frames=400,
            )
            if audio is None or len(audio) == 0:
                raise RuntimeError("Mô hình không tạo được audio.")
            self._tts.save(audio, output_file)

        return str(output_file)
