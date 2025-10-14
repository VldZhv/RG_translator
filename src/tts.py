from __future__ import annotations
import os
import sys
import subprocess
import tempfile
import numpy as np
import soundfile as sf


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Простой линейный ресемплинг до нужной частоты воспроизведения."""
    if sr_in == sr_out or x.size == 0:
        return x.astype("float32")
    if x.ndim == 2:
        x = x.mean(axis=1)  # моно
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_out = np.linspace(0.0, 1.0, num=max(n_out, 1), endpoint=False, dtype=np.float64)
    y = np.interp(t_out, t_in, x.astype(np.float64))
    return y.astype("float32")


def _is_ru_model(path: str) -> bool:
    """Грубая проверка, что для RU действительно выбран RU-голос (по имени файла)."""
    name = os.path.basename(path).lower()
    return ("ru_" in name) or ("_ru-" in name) or name.startswith("ru_")


class TTS:
    def __init__(self, engine: str, voice_ru: str, voice_en: str,
                 use_gpu: bool, mock: bool, sr: int = 16000):
        self.engine = engine
        self.voice_ru = voice_ru
        self.voice_en = voice_en
        self.mock = mock
        self.target_sr = sr

        # Параметры Piper из ENV (pipeline прокидывает их из конфига)
        self.length_scale = float(os.environ.get("PIPER_LENGTH", "1.0"))
        self.noise_scale = float(os.environ.get("PIPER_NOISE", "0.667"))
        self.noise_w = float(os.environ.get("PIPER_NOISE_W", "0.8"))

        # Опционально: id спикера (для мультирежимных моделей Piper)
        self.spk_ru = os.environ.get("PIPER_SPK_RU")  # напр. "0"
        self.spk_en = os.environ.get("PIPER_SPK_EN")

    def _normalize_model_path(self, p: str) -> str:
        # Если дали ...onnx.json — используем соседний .onnx
        if p.endswith(".onnx.json"):
            p = p[:-5]
        return p

    def _piper_say(self, text: str, model_path: str, speaker_id: str | None) -> np.ndarray:
        """Надёжный вызов Piper: подаём текст через --input_file, читаем WAV, ресемплим."""
        model_path = self._normalize_model_path(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Piper model not found: {model_path}")

        with tempfile.TemporaryDirectory() as td:
            txt_path = os.path.join(td, "in.txt")
            out_wav = os.path.join(td, "out.wav")

            # Пишем входной текст (одна строка = одно высказывание)
            with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write((text or "").strip() + "\n")

            cmd = [
                sys.executable, "-m", "piper",
                "--model", model_path,
                "--input_file", txt_path,     # ← подаём через файл, без --text и без stdin
                "--output_file", out_wav,
                "--length_scale", str(self.length_scale),
                "--noise_scale", str(self.noise_scale),
                "--noise_w", str(self.noise_w),
            ]
            if speaker_id:
                cmd += ["--speaker", str(speaker_id)]

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True, text=True,
                creationflags=creationflags,
            )

            # Лог ошибок Piper — в файл
            if proc.returncode != 0:
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open(os.path.join("logs", "piper_stderr.txt"), "a", encoding="utf-8") as f:
                        f.write((proc.stderr or proc.stdout or "") + "\n")
                finally:
                    pass
                raise RuntimeError("Piper synthesis failed")

            wav, sr_in = sf.read(out_wav, dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)

            wav = _resample_linear(wav, sr_in, self.target_sr)

            # Защита от «битых» результатов
            if not np.isfinite(wav).all() or len(wav) < int(self.target_sr * 0.05):
                raise RuntimeError("Piper produced invalid/too short audio")

            return wav.astype("float32")

    def synth(self, text: str, lang: str) -> np.ndarray:
        if self.mock:
            return np.zeros(int(self.target_sr * 0.2), dtype="float32")

        if self.engine != "piper":
            # (если когда-то добавим другие TTS — тут будет роутинг)
            return np.zeros(int(self.target_sr * 0.2), dtype="float32")

        t = (text or "").strip()
        if not t:
            return np.zeros(int(self.target_sr * 0.05), dtype="float32")

        if lang == "ru":
            model = self.voice_ru
            spk = self.spk_ru
            # Подстраховка: предупредим в лог, если выбран не-ru голос для ru-текста
            if not _is_ru_model(model):
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open(os.path.join("logs", "piper_stderr.txt"), "a", encoding="utf-8") as f:
                        f.write(f"Warning: RU text with non-RU model: {model}\n")
                finally:
                    pass
        else:
            model = self.voice_en
            spk = self.spk_en

        return self._piper_say(t, model, spk)
