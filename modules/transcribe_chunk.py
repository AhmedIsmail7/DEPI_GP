# modules/transcribe_chunk.py
import argparse
import sys
import json
from faster_whisper import WhisperModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    try:
        # Load whisper model in a fresh process with isolated memory
        model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=1)
        segments, info = model.transcribe(
            args.audio,
            beam_size=5,
            language=args.language if (args.language and args.language != "None") else None
        )
        segments = list(segments)
        text = " ".join([seg.text for seg in segments]).strip()
        
        result = {
            "text": text,
            "language": info.language if info else None,
            "language_probability": info.language_probability if info else 0.0
        }
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
