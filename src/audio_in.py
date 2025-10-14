from __future__ import annotations
import sounddevice as sd
import numpy as np
import soundfile as sf
from typing import Iterable

class MicStream:
    def __init__(self, samplerate: int = 16000, block_ms: int = 500, device: str | None = None):
        self.sr = samplerate
        self.block = int(self.sr * block_ms / 1000)
        self.device = device

    def stream(self) -> Iterable[np.ndarray]:
        with sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', device=self.device,
                            blocksize=self.block) as st:
            while True:
                data, _ = st.read(self.block)
                yield data.reshape(-1)

class WavStream:
    def __init__(self, path: str, samplerate: int = 16000, block_ms: int = 500):
        self.path = path
        self.sr = samplerate
        self.block = int(self.sr * block_ms / 1000)

    def stream(self) -> Iterable[np.ndarray]:
        data, sr = sf.read(self.path, dtype='float32', always_2d=False)
        if sr != self.sr:
            raise RuntimeError(f"Expected {self.sr} Hz, got {sr}. Resample externally for now.")
        if data.ndim == 2:
            data = data.mean(axis=1)
        for i in range(0, len(data), self.block):
            yield data[i:i+self.block]
