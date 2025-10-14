param(
  [string]$ModelsDir = "models"
)

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
Write-Host "Models dir: $ModelsDir"

Write-Host "`n[ASR] Download a CTranslate2 Whisper model (e.g., small):"
Write-Host "  - https://github.com/guillaumekln/faster-whisper#available-models"
Write-Host "  Place into: $ModelsDir/whisper-small-ct2"

Write-Host "`n[MT] Convert NLLB to CTranslate2 or use Marian (OPUS-MT):"
Write-Host "  - NLLB ct2 guide: https://opennmt.net/CTranslate2/"
Write-Host "  - Marian RU-EN: https://huggingface.co/Helsinki-NLP/opus-mt-ru-en"
Write-Host "  Place into: $ModelsDir/nllb-ru-en-ct2 or $ModelsDir/marian-ru-en"

Write-Host "`n[TTS] Silero TTS loads via torch hub automatically (no local files needed)."
