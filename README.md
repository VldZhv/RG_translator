# translator-rt (RU↔EN, Windows, Offline)

## Quick start (mock)
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
python -m src.app --config configs/default.yaml --mode mic
```
Говори в микрофон — услышишь короткий синус (mock TTS), увидишь, что пайплайн живой.

## Real run
1. Установи PyTorch (CPU или CUDA) по инструкции с pytorch.org.
2. Запусти `tools/fetch_models.ps1` и следуй подсказкам, скачай модели:
   - Whisper (ct2): `models/whisper-*-ct2`
   - MT (NLLB ct2 **или** Marian): `models/nllb-ru-en-ct2` или `models/marian-ru-en`
3. В `configs/gpu_quality.yaml` установи `app.mock: false` и путь к моделям.
4. Запуск:
```bash
python -m src.app --config configs/gpu_quality.yaml --mode mic
```

## WAV mode
```bash
python -m src.app --config configs/cpu_fast.yaml --mode wav --input D:/audio/sample_ru.wav
```

## Notes
- Для VAD сейчас стоит простая заглушка (энергия). Можно заменить на Silero VAD.
- В `mt.py` метод для NLLB ct2 помечен как TODO — обвязать токенизацию и перевод. Marian готов.
- Логи и метрики см. в `logs/` (добавь при необходимости).
