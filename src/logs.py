from __future__ import annotations
import os
import io
import threading
from datetime import datetime

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # до миллисекунд

class LogWriter:
    """
    Потокобезопасная запись логов ASR, MT и сводного "dialog".
    На каждый запуск создаёт новый сет файлов с меткой времени.
    """
    def __init__(self, log_dir: str, session_prefix: str | None = None):
        self.log_dir = log_dir or "logs"
        os.makedirs(self.log_dir, exist_ok=True)

        if session_prefix is None:
            session_prefix = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        self._lock = threading.Lock()

        self.asr_path    = os.path.join(self.log_dir, f"{session_prefix}_asr.txt")
        self.mt_path     = os.path.join(self.log_dir, f"{session_prefix}_mt.txt")
        self.dialog_path = os.path.join(self.log_dir, f"{session_prefix}_dialog.txt")

        # «ленивая» инициализация файлов — создадим пустые сразу (удобно глазами)
        for p in (self.asr_path, self.mt_path, self.dialog_path):
            open(p, "a", encoding="utf-8").close()

    def log_asr(self, fragment_id: str, t_start: float, t_end: float,
                src_lang: str, text: str):
        line = f"[{_ts()}] id={fragment_id} t=({t_start:.2f}..{t_end:.2f}) src_lang={src_lang} ASR: {text}\n"
        self._append(self.asr_path, line)

    def log_mt(self, fragment_id: str, direction: str, src_text: str, hyp_text: str):
        line = f"[{_ts()}] id={fragment_id} dir={direction} MT: {src_text}  =>  {hyp_text}\n"
        self._append(self.mt_path, line)

    def log_dialog(self, fragment_id: str, src_lang: str, direction: str,
                   asr_text: str, mt_text: str):
        """
        Компактный «диалог»: одна пара строк SRC/TRG на фрагмент.
        """
        trg_lang = "en" if direction == "ru-en" else "ru"
        block = (
            f"[{_ts()}] id={fragment_id} dir={direction}\n"
            f"SRC[{src_lang}]: {asr_text}\n"
            f"TRG[{trg_lang}]: {mt_text}\n\n"
        )
        self._append(self.dialog_path, block)

    # --- внутреннее ---

    def _append(self, path: str, text: str):
        with self._lock:
            with io.open(path, "a", encoding="utf-8") as f:
                f.write(text)
                f.flush()
