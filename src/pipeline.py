from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional

from .utils import (
    new_fragment,
    Fragment,
    Stopwatch,
    clean_for_tts,   # санитайзер текста перед TTS
    is_meaningful,   # фильтр пустых/мусорных строк
)
from .asr import ASR
from .mt import MT
from .tts import TTS
from .playback import Player
from .vad import SimpleEnergyVAD
from .logs import LogWriter


logger = logging.getLogger(__name__)


def _resolve_gpu_flag(resources_cfg) -> bool:
    """Определяет, использовать ли GPU для вычислений."""
    requested = getattr(resources_cfg, "use_gpu", "auto")
    if requested is True:
        logger.info("GPU режим принудительно включён конфигом")
        return True
    if requested is False:
        logger.info("GPU режим принудительно отключён конфигом")
        return False

    # Режим auto: проверяем наличие CUDA-устройств через ctranslate2
    try:
        import ctranslate2  # type: ignore

        has_cuda = ctranslate2.get_cuda_device_count() > 0
        if has_cuda:
            logger.info("Обнаружена доступная GPU, включаем ускорение CUDA")
            return True
    except Exception as exc:  # pragma: no cover - диагностика окружения
        logger.debug("Не удалось определить наличие GPU через ctranslate2: %s", exc)

    logger.info("GPU не обнаружена — используем режим CPU")
    return False


class Pipeline:
    """
    Реалтайм-конвейер:
      Audio -> VAD -> ASR -> MT -> TTS -> Playback

    Возможности:
      - Логи в текст: ASR, MT и сводный "dialog".
      - Проброс параметров Piper (если tts.engine == "piper") в ENV до инициализации TTS.
      - Две модели MT (ru→en и en→ru) через model_path / model_path_back.
      - Фильтрация пустых/числовых фрагментов, санитайзер перед TTS.
      - Лог входа в TTS: logs/tts_input_{ru|en}.txt
    """

    def __init__(self, cfg, audio_src):
        self.cfg = cfg
        self.audio_src = audio_src

        # Очередь: ASR -> MT/TTS
        self.q_asr2mt: queue.Queue[Fragment] = queue.Queue(maxsize=32)
        self.stop = threading.Event()

        # Логи: отдельный сет файлов на каждую сессию
        self.logger = LogWriter(log_dir=self.cfg.logging.dir)

        # Плеер
        self.player = Player(
            device=cfg.tts.playback.device,
            volume=cfg.tts.playback.volume,
            sr=cfg.app.sample_rate,
        )

        # Флаг GPU (auto-детект через ctranslate2)
        use_gpu = _resolve_gpu_flag(cfg.resources)

        # ASR
        self.asr = ASR(
            cfg.asr.engine,
            cfg.asr.model_path,
            cfg.asr.beam_size,
            cfg.asr.lang_detect,
            use_gpu,
            cfg.app.mock,
        )

        # MT (две модели: прямое и обратное направления)
        self.mt = MT(
            cfg.mt.engine,
            cfg.mt.model_path,
            use_gpu,
            cfg.app.mock,
            model_path_back=getattr(cfg.mt, "model_path_back", None),
        )

        # Проброс параметров Piper (если движок piper)
        if cfg.tts.engine == "piper" and getattr(cfg.tts, "piper", None):
            os.environ["PIPER_LENGTH"] = str(cfg.tts.piper.length_scale)
            os.environ["PIPER_NOISE"] = str(cfg.tts.piper.noise_scale)
            os.environ["PIPER_NOISE_W"] = str(cfg.tts.piper.noise_w)

        # TTS
        self.tts = TTS(
            cfg.tts.engine,
            cfg.tts.voices.ru,
            cfg.tts.voices.en,
            use_gpu,
            cfg.app.mock,
            sr=cfg.app.sample_rate,
        )

        # VAD (простая энергия; при желании заменить на Silero-VAD)
        self.vad = SimpleEnergyVAD(
            threshold=0.0008,  # можно вынести в конфиг при необходимости
            min_speech_ms=cfg.vad.min_speech_ms,
            min_silence_ms=cfg.vad.min_silence_ms,
            sr=cfg.app.sample_rate,
        )

    # --------------------------- Публичный API ---------------------------

    def run(self):
        """Запустить конвейер (блокирующе)."""
        t0 = time.perf_counter()

        def asr_loop():
            sr = self.cfg.app.sample_rate
            for block in self.audio_src.stream():
                if self.stop.is_set():
                    break

                seg = self.vad.push(block)
                if seg is None:
                    continue

                lang, text = self.asr.transcribe_segment(seg, sr)

                # Отбрасываем пустые/мусорные распознавания (шум, «тишина», служебное)
                if not is_meaningful(text, min_len=3):
                    continue

                mt_dir = self._dir_from_lang(lang)
                t_start = time.perf_counter() - t0

                frag = new_fragment(
                    t_start=t_start,
                    t_end=t_start + len(seg) / sr,
                    src_lang=lang or "auto",
                    text=text,
                    mt_dir=mt_dir,
                )

                # Лог ASR-сегмента
                self.logger.log_asr(
                    fragment_id=frag.fragment_id,
                    t_start=frag.t_start,
                    t_end=frag.t_end,
                    src_lang=frag.src_lang,
                    text=frag.asr_text,
                )

                # Неблокирующая попытка положить фрагмент
                try:
                    self.q_asr2mt.put(frag, timeout=0.2)
                except queue.Full:
                    # Если очередь заполнена — пропускаем, чтобы не накапливать задержку
                    pass

            # Источник исчерпан или остановка — сигналим воркеру
            self.stop.set()

        def worker_loop():
            while not self.stop.is_set() or not self.q_asr2mt.empty():
                try:
                    frag = self.q_asr2mt.get(timeout=0.2)
                except queue.Empty:
                    continue

                _ = Stopwatch()  # при желании использовать для измерений

                # Перевод — только если есть осмысленный текст
                src_txt = (frag.asr_text or "").strip()
                if not is_meaningful(src_txt, min_len=3):
                    continue

                hyp = self.mt.translate(src_txt, frag.mt_dir)

                # Подстраховка: если MT вернул пустое/шум — пропускаем
                if not is_meaningful(hyp, min_len=2):
                    continue

                frag.mt_text = hyp

                # Логи MT + сводный «диалог»
                self.logger.log_mt(
                    fragment_id=frag.fragment_id,
                    direction=frag.mt_dir,
                    src_text=src_txt,
                    hyp_text=hyp,
                )
                self.logger.log_dialog(
                    fragment_id=frag.fragment_id,
                    src_lang=frag.src_lang,
                    direction=frag.mt_dir,
                    asr_text=src_txt,
                    mt_text=hyp,
                )

                # Санитизируем текст перед синтезом (уберём id/SRC/TRG/UUID и т.п.)
                tts_text = clean_for_tts(hyp)
                if not is_meaningful(tts_text, min_len=2):
                    continue

                # ---- Жёсткая фильтрация числового мусора ----
                def _digits_ratio(s: str) -> float:
                    n = len(s.strip())
                    if n == 0:
                        return 1.0
                    d = sum(ch.isdigit() for ch in s)
                    sym = sum(ch in "-_:.;,/#$%&*+=()[]{}" for ch in s)
                    return (d + sym) / n

                if _digits_ratio(tts_text) > 0.6:
                    continue

                # ---- Лог входа TTS (ru/en) ----
                try:
                    os.makedirs("logs", exist_ok=True)
                    lang_tag = "en" if frag.mt_dir == "ru-en" else "ru"
                    with open(os.path.join("logs", f"tts_input_{lang_tag}.txt"), "a", encoding="utf-8") as f:
                        f.write(tts_text + "\n")
                except Exception:
                    pass

                # Синтез
                out_lang = "en" if frag.mt_dir == "ru-en" else "ru"
                wav = self.tts.synth(tts_text, out_lang)

                # Воспроизведение
                self.player.play(wav)

        t_asr = threading.Thread(target=asr_loop, daemon=True, name="asr_loop")
        t_work = threading.Thread(target=worker_loop, daemon=True, name="worker_loop")
        t_asr.start()
        t_work.start()

        try:
            while t_asr.is_alive() and t_work.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop.set()
        finally:
            t_asr.join(timeout=1.0)
            t_work.join(timeout=1.0)

    # --------------------------- Вспомогательные ---------------------------

    def _dir_from_lang(self, lang: Optional[str]) -> str:
        """
        Выбрать направление перевода:
          - если в конфиге задано явно (ru-en / en-ru), берём его;
          - иначе отталкиваемся от языка ASR (ru* -> ru-en, иначе en-ru).
        """
        if self.cfg.app.dir != "auto":
            return self.cfg.app.dir
        if (lang or "").lower().startswith("ru"):
            return "ru-en"
        return "en-ru"
