import os
from typing import List, Dict


class Transcriber:
    """
    Thin wrapper around an ASR backend (e.g., OpenAI Whisper) that returns
    timestamped transcript segments.

    If the ASR library is not installed, this gracefully degrades and returns
    empty results so the rest of the pipeline still works.
    """

    def __init__(self, model_name: str = "small"):
        self._backend = None
        self._model_name = model_name

        try:
            import whisper  # type: ignore

            self._backend = whisper.load_model(model_name)
            print(f"Loaded Whisper ASR model '{model_name}'.")
        except ImportError:
            print(
                "Warning: 'whisper' is not installed. "
                "Audio transcripts will be skipped (Phase 2 STT disabled)."
            )
        except Exception as e:
            print(f"Failed to initialize ASR backend: {e}")

    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Run ASR over the full video and return a list of segments:

        [ { 'start': float, 'end': float, 'text': str }, ... ]
        """
        if self._backend is None:
            return []

        if not os.path.exists(video_path):
            print(f"ASR: video path does not exist: {video_path}")
            return []

        try:
            result = self._backend.transcribe(video_path, verbose=False)
        except Exception as e:
            print(f"ASR transcription failed: {e}")
            return []

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": str(seg.get("text", "")).strip(),
                }
            )

        print(f"ASR produced {len(segments)} transcript segments.")
        return segments

    @staticmethod
    def get_text_for_window(
        segments: List[Dict], start_time: float, end_time: float
    ) -> str:
        """
        Concatenate transcript pieces that overlap a given [start_time, end_time] window.
        """
        if not segments:
            return ""

        parts: List[str] = []
        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            if seg_end < start_time or seg_start > end_time:
                continue
            text = str(seg.get("text", "")).strip()
            if text:
                parts.append(text)

        return " ".join(parts).strip()

