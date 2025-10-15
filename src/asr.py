from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)


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
                if not use_gpu:
                    # У некоторых сборок ctranslate2 при импорте пытается загрузить CUDA-библиотеки,
                    # даже если планируем работать на CPU. Явно запрещаем использование CUDA,
                    # чтобы избежать ошибок вида «Could not locate cudnn_ops64_9.dll» на Windows.
                    os.environ.setdefault("CT2_USE_CUDA", "0")

                from faster_whisper import WhisperModel

                device = "cuda" if use_gpu else "cpu"
                compute_type = "float16" if use_gpu else "int8"

                try:
                    self.model = WhisperModel(
                        model_path,
                        device=device,
                        compute_type=compute_type,
                    )
                except (RuntimeError, ValueError) as err:
                    if use_gpu and "float16" in compute_type:
                        # Для некоторых видеокарт (например, GTX 10xx) может потребоваться float32.
                        self.model = WhisperModel(
                            model_path,
                            device=device,
                            compute_type="float32",
                        )
                    elif "cudnn" in str(err).lower() or "cuda" in str(err).lower():
                        logger.warning(
                            "Не удалось инициализировать Faster-Whisper на GPU: %s. Переключаемся на CPU.",
                            err,
                        )
                        os.environ["CT2_USE_CUDA"] = "0"
                        device = "cpu"
                        compute_type = "int8"
                        self.use_gpu = False
                        self.model = WhisperModel(
                            model_path,
                            device=device,
                            compute_type=compute_type,
                        )
                    else:
                        raise err
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
