from __future__ import annotations

class ASR:
    def __init__(self, engine: str, model_path: str, beam_size: int, lang_detect: bool, use_gpu: bool, mock: bool):
        self.engine = engine
        self.model_path = model_path
        self.beam_size = beam_size
        self.lang_detect = lang_detect
        self.use_gpu = use_gpu
        self.mock = mock
        self.model = None
        if not mock:
            if engine == "faster-whisper":
                from faster_whisper import WhisperModel
                device = "cuda" if use_gpu else "cpu"
                self.model = WhisperModel(model_path, device=device, compute_type="float16" if use_gpu else "int8")
            elif engine == "vosk":
                import vosk  # type: ignore
                self.model = vosk.Model(model_path)
            else:
                raise ValueError(f"Unknown ASR engine: {engine}")

    def transcribe_segment(self, audio_f32_mono, sr: int) -> tuple[str, str]:
        if self.mock:
            return ("ru", "это тестовая фраза")
        if self.engine == "faster-whisper":
            segments, info = self.model.transcribe(audio=audio_f32_mono, beam_size=self.beam_size, language=None if self.lang_detect else None)
            text = " ".join(s.text.strip() for s in segments)
            lang = info.language or "auto"
            return (lang, text.strip())
        elif self.engine == "vosk":
            raise NotImplementedError("Vosk segment mode not implemented in this skeleton")
        else:
            raise RuntimeError("ASR not initialized")
