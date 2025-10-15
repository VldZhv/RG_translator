from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def _ensure_cuda(module, *, component: str, enabled: bool) -> tuple[object, bool]:
    """Пытается перевести модуль на CUDA, при неудаче возвращает CPU-версию."""

    if not enabled or module is None:
        return module, enabled

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - завязано на окружение
        logger.warning(
            "Не удалось импортировать torch для GPU режима (%s). Оставляем CPU: %s",
            component,
            exc,
        )
        return module, False

    if not torch.cuda.is_available():  # pragma: no cover - зависит от окружения
        logger.warning(
            "CUDA недоступна для компонента %s — переключаемся на CPU", component
        )
        return module, False

    try:
        module = module.to("cuda")
    except (AssertionError, RuntimeError) as exc:  # pragma: no cover - зависит от сборки torch
        logger.warning(
            "Не удалось перевести компонент %s на CUDA, работаем на CPU: %s",
            component,
            exc,
        )
        return module, False

    return module, enabled

class MT:
    def __init__(self, engine: str, model_path: str, use_gpu: bool, mock: bool, model_path_back: str | None = None):
        self.engine = engine
        self.model_path = model_path
        self.model_path_back = model_path_back
        self.use_gpu = use_gpu
        self.mock = mock

        self.model = None
        self.tokenizer = None
        self.model_back = None
        self.tokenizer_back = None

        if not mock:
            if engine == "marian":
                from transformers import MarianMTModel, MarianTokenizer  # type: ignore
                # RU→EN
                self.tokenizer = MarianTokenizer.from_pretrained(model_path)
                self.model = MarianMTModel.from_pretrained(model_path)
                self.model, self.use_gpu = _ensure_cuda(
                    self.model,
                    component="основная модель MarianMT",
                    enabled=self.use_gpu,
                )
                # EN→RU (опционально)
                if model_path_back:
                    self.tokenizer_back = MarianTokenizer.from_pretrained(model_path_back)
                    self.model_back = MarianMTModel.from_pretrained(model_path_back)
                    self.model_back, self.use_gpu = _ensure_cuda(
                        self.model_back,
                        component="обратная модель MarianMT",
                        enabled=self.use_gpu,
                    )
            elif engine == "nllb-ct2":
                import ctranslate2  # type: ignore
                # Для NLLB обычно нужен токенайзер и специальные языковые теги — TODO
                self.model = ctranslate2.Translator(model_path, device="cuda" if use_gpu else "cpu")
                if model_path_back:
                    self.model_back = ctranslate2.Translator(model_path_back, device="cuda" if use_gpu else "cpu")
            elif engine == "argos":
                import argostranslate.package  # type: ignore
            else:
                raise ValueError(f"Unknown MT engine: {engine}")

    def translate(self, text: str, direction: str) -> str:
        if self.mock:
            return "this is a test phrase" if direction == "ru-en" else "это тестовая фраза"

        if self.engine == "marian":
            if direction == "ru-en":
                tok = self.tokenizer(text, return_tensors="pt", padding=True)
                if hasattr(self.model, "device") and str(self.model.device).startswith("cuda"):
                    tok = {k: v.to("cuda") for k, v in tok.items()}
                out = self.model.generate(**tok, num_beams=4, max_length=256)
                return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            else:  # en-ru
                if not (self.model_back and self.tokenizer_back):
                    # если второй модели нет — временно переводим той же (хуже качеством)
                    tok = self.tokenizer(text, return_tensors="pt", padding=True)
                    if hasattr(self.model, "device") and str(self.model.device).startswith("cuda"):
                        tok = {k: v.to("cuda") for k, v in tok.items()}
                    out = self.model.generate(**tok, num_beams=4, max_length=256)
                    return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                tok = self.tokenizer_back(text, return_tensors="pt", padding=True)
                if hasattr(self.model_back, "device") and str(self.model_back.device).startswith("cuda"):
                    tok = {k: v.to("cuda") for k, v in tok.items()}
                out = self.model_back.generate(**tok, num_beams=4, max_length=256)
                return self.tokenizer_back.batch_decode(out, skip_special_tokens=True)[0]

        elif self.engine == "nllb-ct2":
            # TODO: полноценная обвязка с токенайзером и языковыми тегами
            return text

        elif self.engine == "argos":
            return text

        raise RuntimeError("MT not initialized")
