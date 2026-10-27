"""Speech-to-Text Recognition and WER Evaluation Module.

Features:
- Converts and decodes any audio format (.m4a, .wav, .mp3, etc.) automatically.
- End-to-end Speech Recognition (ASR) to text.
- Word Error Rate (WER) and Character Error Rate (CER) computation with full alignment breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import wave

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import speech_recognition as sr
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

try:
    import imageio_ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False


def convert_to_wav(audio_path: str | Path, target_sr: int = 16000) -> Path:
    """Ensure audio is 16kHz Mono 16-bit PCM WAV for optimal ASR accuracy."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Tệp không tồn tại: {audio_path}")

    # Check if already a 16kHz mono WAV
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == target_sr and wf.getsampwidth() == 2:
                    return audio_path
        except Exception:
            pass

    out_wav = audio_path.with_suffix(".asr_temp.wav")
    if not FFMPEG_AVAILABLE:
        raise RuntimeError("Cần imageio-ffmpeg để chuyển đổi định dạng âm thanh.")

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        exe, "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", str(target_sr),
        str(out_wav), "-y"
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"Lỗi chuyển đổi audio sang WAV: {res.stderr}")
    return out_wav


class SpeechToTextModel:
    """Speech Recognition Model wrapper supporting multi-lingual ASR."""

    def __init__(self, default_language: str = "vi-VN"):
        self.recognizer = sr.Recognizer()
        self.default_language = default_language
        # Adjust for ambient noise and sensitivity
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> str:
        """Transcribe an audio file to text."""
        lang = language or self.default_language
        wav_path = convert_to_wav(audio_path)

        with sr.AudioFile(str(wav_path)) as source:
            # Record entire audio
            audio_data = self.recognizer.record(source)

        try:
            text = self.recognizer.recognize_google(audio_data, language=lang)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            raise RuntimeError(f"Lỗi kết nối tới dịch vụ nhận diện giọng nói: {e}")


def normalize_text(text: str) -> str:
    """Normalize text: lowercases, removes punctuation, and strips whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer_custom(reference: str, hypothesis: str) -> dict:
    """Pure Python Levenshtein distance at word-level for exact S, D, I counts."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        if m == 0:
            return {"wer": 0.0, "s": 0, "d": 0, "i": 0, "c": 0, "n": 0, "accuracy": 100.0}
        return {"wer": 1.0, "s": 0, "d": 0, "i": m, "c": 0, "n": 0, "accuracy": 0.0}

    # DP Matrix: d[i][j] = (cost, s, d, i_ops, c)
    dp = [[(0, 0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0, 0)  # Deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j, 0)  # Insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost, s, d, ins, c = dp[i - 1][j - 1]
                dp[i][j] = (cost, s, d, ins, c + 1)
            else:
                sub_cost, s1, d1, i1, c1 = dp[i - 1][j - 1]
                del_cost, s2, d2, i2, c2 = dp[i - 1][j]
                ins_cost, s3, d3, i3, c3 = dp[i][j - 1]

                candidates = [
                    (sub_cost + 1, s1 + 1, d1, i1, c1),      # Substitution
                    (del_cost + 1, s2, d2 + 1, i2, c2),      # Deletion
                    (ins_cost + 1, s3, d3, i3 + 1, c3),      # Insertion
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])

    total_cost, s, d, ins, c = dp[n][m]
    wer = (s + d + ins) / n
    accuracy = max(0.0, 1.0 - wer) * 100.0

    return {
        "wer": round(float(wer), 4),
        "wer_percent": round(float(wer * 100.0), 2),
        "substitutions": s,
        "deletions": d,
        "insertions": ins,
        "correct": c,
        "reference_word_count": n,
        "hypothesis_word_count": m,
        "accuracy_percent": round(accuracy, 2),
    }


def evaluate_wer(reference: str, hypothesis: str) -> dict:
    """Calculate WER using both jiwer (if installed) and custom Levenshtein."""
    custom_res = compute_wer_custom(reference, hypothesis)

    if JIWER_AVAILABLE and reference.strip():
        try:
            jiwer_out = jiwer.process_words(normalize_text(reference), normalize_text(hypothesis))
            custom_res["jiwer_wer"] = round(float(jiwer_out.wer), 4)
            custom_res["jiwer_alignment"] = jiwer.visualize_alignment(jiwer_out)
        except Exception:
            pass

    return custom_res


def run_pipeline(audio_path: str | Path, reference_text: str | None = None, language: str = "vi-VN"):
    """Run full Speech-to-Text inference and evaluate WER against reference text."""
    audio_path = Path(audio_path)
    print("=" * 70)
    print(f"HỆ THỐNG NHẬN DIỆN GIỌNG NÓI & ĐÁNH GIÁ ĐỘ LỖI TỪ (WER)")
    print("=" * 70)
    print(f"1. Tệp âm thanh đầu vào: {audio_path.resolve()}")
    print(f"2. Ngôn ngữ nhận diện: {language}")

    # Model inference
    model = SpeechToTextModel(default_language=language)
    print("\n[Đang nhận diện giọng nói...]")
    hypothesis_text = model.transcribe(audio_path, language=language)

    print("\n" + "-" * 70)
    print("KẾT QUẢ ĐẦU RA (SPEECH-TO-TEXT HYPOTHESIS):")
    print("-" * 70)
    print(f"Text nhận diện được: \"{hypothesis_text}\"")

    # If reference is not provided, test against the exact ground truth spoken in Hi_new.m4a
    if reference_text is None:
        # Default ground-truth spoken in Hi_new.m4a
        default_ground_truth = (
            "Hello xin chào tất cả các bạn Hello xin chào tất cả các bạn "
            "Xin chào tất cả các bạn đến với trường Đại học Công nghệ Đại học Quốc gia Hà Nội"
        )
        reference_text = default_ground_truth
        print(f"\n[Ghi chú: Đang sử dụng Ground-Truth thực tế của bản ghi âm để tính WER]")

    print("\n" + "-" * 70)
    print("KẾT QUẢ ĐỐI CHIẾU & TÍNH TOÁN ĐỘ LỖI TỪ (WER):")
    print("-" * 70)
    print(f"Văn bản gốc (Reference) : \"{reference_text}\"")
    print(f"Văn bản nhận diện (Hypothesis): \"{hypothesis_text}\"")

    metrics = evaluate_wer(reference_text, hypothesis_text)
    print("\nCHI TIẾT CHỈ SỐ LỖI:")
    print(f"  * Word Error Rate (WER) : {metrics['wer_percent']}% (WER = {metrics['wer']})")
    print(f"  * Độ chính xác (Accuracy): {metrics['accuracy_percent']}%")
    print(f"  * Số từ văn bản gốc (N)  : {metrics['reference_word_count']}")
    print(f"  * Số từ nhận diện (M)   : {metrics['hypothesis_word_count']}")
    print(f"  * Số từ đúng (Correct C): {metrics['correct']}")
    print(f"  * Thay thế (Substitutions S) : {metrics['substitutions']}")
    print(f"  * Bỏ sót   (Deletions D)     : {metrics['deletions']}")
    print(f"  * Thêm từ  (Insertions I)    : {metrics['insertions']}")

    if "jiwer_alignment" in metrics:
        print("\nSƠ ĐỒ SO KHỚP CÁC TỪ (WORD ALIGNMENT):")
        print(metrics["jiwer_alignment"])

    print("=" * 70)
    return {
        "audio_file": str(audio_path),
        "hypothesis": hypothesis_text,
        "reference": reference_text,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Mô hình nhận diện giọng nói (ASR) và tính Word Error Rate (WER)")
    parser.add_argument("--audio", "-a", type=str, default="Hi_new.m4a", help="Đường dẫn file âm thanh đầu vào")
    parser.add_argument("--reference", "-r", type=str, default=None, help="Văn bản gốc (Ground-truth) để tính WER")
    parser.add_argument("--language", "-l", type=str, default="vi-VN", help="Mã ngôn ngữ (vi-VN, en-US, ...)")
    args = parser.parse_args()

    run_pipeline(args.audio, args.reference, args.language)


if __name__ == "__main__":
    main()
