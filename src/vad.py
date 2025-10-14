from __future__ import annotations
import numpy as np

class SimpleEnergyVAD:
    def __init__(self, threshold: float = 0.01, min_speech_ms: int = 300, min_silence_ms: int = 250, sr: int = 16000):
        self.th = threshold
        self.min_speech = int(sr * min_speech_ms / 1000)
        self.min_sil = int(sr * min_silence_ms / 1000)
        self.speech_buf: list[np.ndarray] = []
        self.silence = 0
        self.in_speech = False
        self.sr = sr

    def push(self, block: np.ndarray):
        energy = float((block**2).mean())
        if energy > self.th:
            self.silence = 0
            self.speech_buf.append(block)
            if not self.in_speech:
                if sum(len(x) for x in self.speech_buf) >= self.min_speech:
                    self.in_speech = True
            return None
        else:
            if self.in_speech:
                self.silence += len(block)
                self.speech_buf.append(block)
                if self.silence >= self.min_sil:
                    seg = np.concatenate(self.speech_buf, axis=0)
                    self.speech_buf = []
                    self.silence = 0
                    self.in_speech = False
                    return seg
            else:
                self.speech_buf = []
            return None
