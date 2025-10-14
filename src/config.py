from __future__ import annotations
from pydantic import BaseModel
from typing import Literal
import yaml

class AppCfg(BaseModel):
    mode: Literal["mic", "wav"] = "mic"
    input_wav: str = ""
    mock: bool = True
    sample_rate: int = 16000
    chunk_ms: int = 500
    src_lang: Literal["ru", "en", "auto"] = "auto"
    dir: Literal["ru-en", "en-ru", "auto"] = "auto"

class ResourcesCfg(BaseModel):
    use_gpu: Literal["auto", True, False] = "auto"
    threads: int = 4

class VadCfg(BaseModel):
    enabled: bool = True
    threshold: float = 0.5
    min_speech_ms: int = 400
    min_silence_ms: int = 300

class AsrCfg(BaseModel):
    engine: Literal["faster-whisper", "vosk"] = "faster-whisper"
    model_path: str = "models/whisper-small-ct2"
    beam_size: int = 5
    lang_detect: bool = True

class SegmenterCfg(BaseModel):
    strategy: Literal["punct_pause", "fixed"] = "punct_pause"
    max_words: int = 20

class MtCfg(BaseModel):
    engine: Literal["nllb-ct2", "marian", "argos"] = "nllb-ct2"
    model_path: str = "models/nllb-ru-en-ct2"       # для RU→EN
    model_path_back: str | None = None              # для EN→RU (Marian/NLLB)
    max_src_tokens: int = 64
    segmenter: SegmenterCfg = SegmenterCfg()

class TtsVoicesCfg(BaseModel):
    ru: str = "aidar_v3_16khz"
    en: str = "lj_16khz"

class TtsPlaybackCfg(BaseModel):
    device: str | int | None = None
    volume: float = 0.9

class PiperCfg(BaseModel):
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8

class TtsCfg(BaseModel):
    engine: Literal["silero", "piper"] = "silero"   # <-- добавили "piper"
    voices: TtsVoicesCfg = TtsVoicesCfg()
    playback: TtsPlaybackCfg = TtsPlaybackCfg()
    # Параметры Piper (опционально в конфиге tts.piper: {...})
    piper: PiperCfg | None = None


class LoggingCfg(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir: str = "logs"
    save_tts_wav: bool = False

class SafetyCfg(BaseModel):
    prevent_feedback_loop: Literal["mute_mic", "ducking", "none"] = "ducking"
    ducking_db: int = -12

class Cfg(BaseModel):
    app: AppCfg = AppCfg()
    resources: ResourcesCfg = ResourcesCfg()
    vad: VadCfg = VadCfg()
    asr: AsrCfg = AsrCfg()
    mt: MtCfg = MtCfg()
    tts: TtsCfg = TtsCfg()
    logging: LoggingCfg = LoggingCfg()
    safety: SafetyCfg = SafetyCfg()

    @staticmethod
    def load(path: str) -> "Cfg":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Cfg(**data)
