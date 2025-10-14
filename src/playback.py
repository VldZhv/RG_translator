from __future__ import annotations
import sounddevice as sd
import numpy as np

def _resolve_device(device: str | int | None):
    # None / "" / "default" -> системное устройство по умолчанию
    if device is None:
        return None
    if isinstance(device, str):
        s = device.strip().lower()
        if s == "" or s == "default":
            return None
        # числовая строка -> индекс
        if s.isdigit():
            return int(s)
        # поиск по подстроке имени среди выходных устройств
        try:
            devs = sd.query_devices()
            outs = [(i, d) for i, d in enumerate(devs) if d.get("max_output_channels", 0) > 0]
            for i, d in outs:
                if s in d["name"].lower():
                    return i
        except Exception:
            pass
        # не нашли — пусть будет дефолт
        return None
    if isinstance(device, int):
        return device
    return None

class Player:
    def __init__(self, device: str | int | None = None, volume: float = 1.0, sr: int = 16000):
        self.device = _resolve_device(device)
        self.volume = volume
        self.sr = sr

    def play(self, wav: np.ndarray):
        try:
            sd.play(wav * self.volume, self.sr, device=self.device)
            sd.wait()
        except ValueError:
            # Если указали несуществующее устройство — пробуем системное по умолчанию
            sd.play(wav * self.volume, self.sr, device=None)
            sd.wait()
