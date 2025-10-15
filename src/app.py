from __future__ import annotations
import os as _os
_os.environ.setdefault("ORT_DISABLE_CUDA", "1")
_os.environ.setdefault("ORT_DISABLE_TENSORRT", "1")
_os.environ.setdefault("ORT_DML_ENABLE", "0")

import argparse
from .config import Cfg
from .pipeline import Pipeline
from .audio_in import MicStream, WavStream

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', choices=['mic','wav'])
    ap.add_argument('--input', help='wav path for wav mode')
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    if args.mode:
        cfg.app.mode = args.mode
    if args.input:
        cfg.app.input_wav = args.input

    if cfg.app.mode == 'mic':
        audio_src = MicStream(samplerate=cfg.app.sample_rate, block_ms=cfg.app.chunk_ms)
    else:
        if not cfg.app.input_wav:
            raise SystemExit('Provide --input <file.wav> for wav mode')
        audio_src = WavStream(path=cfg.app.input_wav, samplerate=cfg.app.sample_rate, block_ms=cfg.app.chunk_ms)

    pipe = Pipeline(cfg, audio_src)
    pipe.run()

if __name__ == '__main__':
    main()
