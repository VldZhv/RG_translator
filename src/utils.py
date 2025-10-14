from __future__ import annotations
from dataclasses import dataclass
import time
import uuid

@dataclass
class Fragment:
    fragment_id: str
    t_start: float
    t_end: float
    src_lang: str
    asr_text: str
    mt_dir: str
    mt_text: str | None = None

class Stopwatch:
    def __init__(self):
        self.t0 = time.perf_counter()
    def ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

def new_fragment(t_start: float, t_end: float, src_lang: str, text: str, mt_dir: str) -> Fragment:
    return Fragment(fragment_id=str(uuid.uuid4()), t_start=t_start, t_end=t_end,
                    src_lang=src_lang, asr_text=text, mt_dir=mt_dir)
import re

_UUID_RE = re.compile(r"\b[0-9a-f]{8}\b-[0-9a-f\-]{27,}\b", re.I)

def clean_for_tts(text: str) -> str:
    if not text:
        return ""
    t = text

    # вырезаем служебные метки и UUID/ID
    t = re.sub(r"\b(id|src|trg)\s*[:=]\s*", " ", t, flags=re.I)
    t = _UUID_RE.sub(" ", t)

    # убираем технические скобки/теги вида SRC[en]:
    t = re.sub(r"\b(SRC|TRG)\s*\[[^\]]*\]\s*:\s*", " ", t, flags=re.I)

    # убираем одиночные латинские/цифровые токены, «мусор» во фразе
    t = re.sub(r"\b[0-9a-f]{1,3}\b", " ", t, flags=re.I)

    # сжимаем повторные пробелы, подчищаем
    t = re.sub(r"\s+", " ", t).strip()

    return t

def is_meaningful(text: str, min_len: int = 2) -> bool:
    if not text:
        return False
    # не считаем осмысленным набор 1–2 символов, чистые цифры и т.п.
    t = text.strip()
    if len(t) < min_len:
        return False
    if re.fullmatch(r"[\d\W_]+", t):
        return False
    return True

