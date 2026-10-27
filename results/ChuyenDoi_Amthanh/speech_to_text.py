import os
import re
import sys
import jiwer
from faster_whisper import WhisperModel

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Đường dẫn file âm thanh và văn bản gốc
audio_path = sys.argv[1] if len(sys.argv) > 1 else "Hi_new2.m4a"
reference_path = sys.argv[2] if len(sys.argv) > 2 else "reference.txt"

print(f"Đang xử lý tệp: {audio_path}")
print("Đang tải mô hình faster-whisper (small, int8)...")
model = WhisperModel("small", device="cpu", compute_type="int8")

# Nhận diện giọng nói
print("Đang nhận diện giọng nói...")
segments, info = model.transcribe(
    audio_path,
    language="vi",
    beam_size=5
)

text = " ".join(segment.text.strip() for segment in segments)

print("\nVĂN BẢN NHẬN DIỆN:")
print(text)

# Lưu văn bản nhận diện
with open("ket_qua.txt", "w", encoding="utf-8") as f:
    f.write(text)

# Chuẩn hóa văn bản
def chuan_hoa(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = " ".join(s.split())
    return s

# Tính WER nếu có văn bản gốc
if os.path.exists(reference_path):
    with open(reference_path, "r", encoding="utf-8") as f:
        reference = f.read().strip()

    ref = chuan_hoa(reference)
    hyp = chuan_hoa(text)

    if not ref:
        print("\nVăn bản gốc rỗng, không thể tính WER.")
    else:
        result = jiwer.process_words(ref, hyp)

        print("\nVĂN BẢN GỐC (REFERENCE):")
        print(reference)

        print("\nKẾT QUẢ ĐÁNH GIÁ ĐỘ LỖI TỪ (WER):")
        print(f"  * Word Error Rate (WER) : {result.wer * 100:.2f}%")
        print(f"  * Số từ đúng (Hits)     : {result.hits}")
        print(f"  * Số từ thay thế (S)    : {result.substitutions}")
        print(f"  * Số từ bị thiếu (D)    : {result.deletions}")
        print(f"  * Số từ bị thêm (I)     : {result.insertions}")
        print(f"  * Tổng số từ gốc (N)    : {len(ref.split())}")

        print("\nSƠ ĐỒ SO KHỚP TỪ (ALIGNMENT):")
        print(jiwer.visualize_alignment(result))

        with open("danh_gia.txt", "w", encoding="utf-8") as f:
            f.write(f"Văn bản gốc: {reference}\n")
            f.write(f"Văn bản nhận diện: {text}\n")
            f.write(f"S = {result.substitutions}\n")
            f.write(f"D = {result.deletions}\n")
            f.write(f"I = {result.insertions}\n")
            f.write(f"WER = {result.wer * 100:.2f}%\n")
else:
    print(f"\nChưa có {reference_path} để tính WER.")
