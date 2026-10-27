"""Audio Processing and Visualization Module.

Supports:
1. Comparative Signal Analysis: Clean vs Noisy (SNR, RMS, PSD, STFT, Wiener & Bandpass Filtering).
2. Real-World Speech Signal Processing: Ambient Noise Profiling, VAD, Pitch (F0) Tracking,
   Spectral Gating Enhancement, and Multi-Domain Visualizations.
3. Multi-format support (.wav, .m4a, .mp3, etc.) via embedded imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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

import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file generation
import matplotlib.pyplot as plt

try:
    import imageio_ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False


# Styling configuration
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Calibri", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# AUDIO I/O UTILITIES
# =====================================================================

def load_audio(path: str | Path) -> tuple[np.ndarray, int, dict]:
    """Load an audio file (.wav, .m4a, .mp3) into normalized float64 [-1.0, 1.0]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tệp âm thanh không tồn tại: {path}")

    # If not a standard WAV, decode using embedded ffmpeg
    if path.suffix.lower() != ".wav":
        wav_path = path.with_suffix(".converted.wav")
        if not wav_path.is_file() or wav_path.stat().st_mtime < path.stat().st_mtime:
            if not FFMPEG_AVAILABLE:
                raise RuntimeError("Cần thư viện 'imageio-ffmpeg' để giải mã định dạng non-WAV (.m4a/.mp3).")
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [exe, "-i", str(path), "-vn", "-ac", "1", "-ar", "48000", str(wav_path), "-y"]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, encoding="utf-8", errors="replace")
            if res.returncode != 0:
                raise RuntimeError(f"Lỗi giải mã audio bằng ffmpeg: {res.stderr}")
        read_target = wav_path
    else:
        read_target = path

    with wave.open(str(read_target), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        raw_int = np.frombuffer(raw_bytes, dtype=np.int16)
        data = raw_int.astype(np.float64) / 32768.0
    elif sampwidth == 4:
        raw_int = np.frombuffer(raw_bytes, dtype=np.int32)
        data = raw_int.astype(np.float64) / 2147483648.0
    elif sampwidth == 1:
        raw_int = np.frombuffer(raw_bytes, dtype=np.uint8)
        data = (raw_int.astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"Độ sâu bit không được hỗ trợ: {sampwidth * 8} bits")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    metadata = {
        "source_file": str(path),
        "channels": 1,
        "sample_width_bytes": 2,
        "sample_rate": framerate,
        "num_samples": len(data),
        "duration_seconds": round(len(data) / framerate, 4),
    }
    return data, framerate, metadata


# Backward compatibility alias
load_wav = load_audio


def save_wav(path: str | Path, data: np.ndarray, sample_rate: int) -> Path:
    """Save normalized float array [-1.0, 1.0] to 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(data, -1.0, 1.0)
    int_pcm = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_pcm.tobytes())

    return path


# =====================================================================
# REAL-WORLD SPEECH PROCESSING & ENHANCEMENT (For Himn.m4a)
# =====================================================================

def compute_speech_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    """Analyze real speech: Short-Time Energy, VAD, Pitch (F0), and Spectrum."""
    total_samples = len(audio)
    duration = total_samples / sample_rate

    # Global metrics
    rms_global = np.sqrt(np.mean(audio ** 2))
    peak_global = np.max(np.abs(audio))

    # Framing: 30ms window, 10ms hop
    frame_len = int(0.03 * sample_rate)
    hop_len = int(0.01 * sample_rate)
    times = []
    rms_frames = []
    zcr_frames = []
    f0_frames = []

    # Frequency bounds for pitch detection (human male/female voice: 70Hz - 400Hz)
    min_lag = int(sample_rate / 400)
    max_lag = int(sample_rate / 70)

    for i in range(0, total_samples - frame_len, hop_len):
        frame = audio[i:i + frame_len]
        t = (i + frame_len // 2) / sample_rate
        times.append(t)

        # RMS Energy
        f_rms = np.sqrt(np.mean(frame ** 2))
        rms_frames.append(f_rms)

        # Zero Crossing Rate (ZCR)
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        zcr_frames.append(zcr)

        # Autocorrelation Pitch (F0)
        if f_rms > 0.04:  # Voiced candidate
            frame_centered = frame - np.mean(frame)
            corr = np.correlate(frame_centered, frame_centered, mode="full")
            corr = corr[len(corr) // 2:]
            if max_lag < len(corr):
                peak_idx = min_lag + np.argmax(corr[min_lag:max_lag])
                if corr[0] > 0 and (corr[peak_idx] / corr[0]) > 0.35:
                    f0 = sample_rate / peak_idx
                    f0_frames.append(f0)
                else:
                    f0_frames.append(np.nan)
            else:
                f0_frames.append(np.nan)
        else:
            f0_frames.append(np.nan)

    rms_arr = np.array(rms_frames)
    zcr_arr = np.array(zcr_frames)
    times_arr = np.array(times)

    # Voice Activity Detection (VAD) via adaptive energy thresholding
    # Sort energy to estimate ambient noise floor (lower 25%) and speech peak (top 15%)
    noise_est = np.percentile(rms_arr, 20)
    speech_thresh = max(0.05, noise_est * 2.8)
    vad_mask = rms_arr > speech_thresh

    # Active speech statistics
    speech_frames = rms_arr[vad_mask]
    noise_frames = rms_arr[~vad_mask]

    rms_speech = float(np.mean(speech_frames)) if len(speech_frames) > 0 else rms_global
    rms_noise = float(np.mean(noise_frames)) if len(noise_frames) > 0 else noise_est
    estimated_snr = 20.0 * np.log10(rms_speech / max(1e-6, rms_noise))

    valid_f0 = [f for f in f0_frames if not np.isnan(f)]
    mean_pitch = float(np.mean(valid_f0)) if valid_f0 else 0.0
    min_pitch = float(np.min(valid_f0)) if valid_f0 else 0.0
    max_pitch = float(np.max(valid_f0)) if valid_f0 else 0.0

    # Spectral Formant Peak estimation in speech segment
    if np.any(vad_mask):
        speech_time_start = times_arr[np.where(vad_mask)[0][0]]
        speech_time_end = times_arr[np.where(vad_mask)[0][-1]]
        s_idx = int(speech_time_start * sample_rate)
        e_idx = int(speech_time_end * sample_rate)
        speech_segment = audio[s_idx:e_idx]
    else:
        speech_segment = audio
        speech_time_start, speech_time_end = 0.0, duration

    f_welch, psd_speech = signal.welch(speech_segment, fs=sample_rate, nperseg=2048)
    top_formants_idx = np.argsort(psd_speech[:int(3000 * len(f_welch) / (sample_rate / 2))])[-6:][::-1]
    formants = sorted([round(float(f_welch[idx]), 1) for idx in top_formants_idx])

    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "total_samples": total_samples,
        "rms_global": round(float(rms_global), 4),
        "peak_global": round(float(peak_global), 4),
        "rms_speech": round(float(rms_speech), 4),
        "rms_noise_floor": round(float(rms_noise), 4),
        "estimated_snr_db": round(float(estimated_snr), 2),
        "mean_pitch_f0_hz": round(mean_pitch, 1),
        "min_pitch_hz": round(min_pitch, 1),
        "max_pitch_hz": round(max_pitch, 1),
        "speech_start_sec": round(float(speech_time_start), 3),
        "speech_end_sec": round(float(speech_time_end), 3),
        "formants_hz": formants,
        "times": times_arr,
        "rms_frames": rms_arr,
        "zcr_frames": zcr_arr,
        "f0_frames": np.array(f0_frames),
        "vad_mask": vad_mask,
    }


def enhance_speech(audio: np.ndarray, sample_rate: int, noise_duration_sec: float = 0.55) -> tuple[np.ndarray, dict]:
    """Denoise and enhance speech audio via High-pass filtering and Spectral Gating."""
    # 1. High-pass filter at 80 Hz to eliminate micro-vibrations and room rumble
    sos_hp = signal.butter(4, 80, btype="highpass", fs=sample_rate, output="sos")
    audio_hp = signal.sosfiltfilt(sos_hp, audio)

    # 2. STFT Spectral Gating Denoising
    nperseg = 1024
    noverlap = 768
    f, t, Zxx = signal.stft(audio_hp, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Learn noise profile from initial ambient period (first ~0.55s)
    noise_frames_count = max(4, int(noise_duration_sec * sample_rate / (nperseg - noverlap)))
    noise_profile = np.mean(mag[:, :noise_frames_count], axis=1, keepdims=True)
    noise_std = np.std(mag[:, :noise_frames_count], axis=1, keepdims=True)

    # Dynamic spectral subtraction with soft floor
    alpha = 2.2  # Oversubtraction factor
    beta = 0.03  # Spectral floor
    threshold = noise_profile + 1.2 * noise_std
    gain = np.maximum(beta, (mag - alpha * threshold) / np.maximum(mag, 1e-12))
    # Smooth gain matrix over time and frequency
    gain = signal.medfilt2d(gain, kernel_size=[3, 3])

    Zxx_enhanced = mag * gain * np.exp(1j * phase)
    _, denoised = signal.istft(Zxx_enhanced, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    denoised = denoised[:len(audio)]

    # 3. Soft-knee gating in non-speech quiet regions
    noise_sample_before = audio[:int(noise_duration_sec * sample_rate)]
    noise_sample_after = denoised[:int(noise_duration_sec * sample_rate)]
    rms_before = np.sqrt(np.mean(noise_sample_before ** 2))
    rms_after = np.sqrt(np.mean(noise_sample_after ** 2))
    noise_attenuation_db = 20.0 * np.log10(rms_before / max(1e-6, rms_after))

    # 4. Peak Normalization to -1.0 dBFS (0.891) for clear playback
    peak_val = np.max(np.abs(denoised))
    if peak_val > 0:
        enhanced_normalized = (denoised / peak_val) * 0.891
    else:
        enhanced_normalized = denoised

    info = {
        "noise_attenuation_db": round(float(noise_attenuation_db), 2),
        "rms_noise_before": round(float(rms_before), 4),
        "rms_noise_after": round(float(rms_after), 4),
        "enhanced_peak": round(float(np.max(np.abs(enhanced_normalized))), 4),
        "enhanced_rms": round(float(np.sqrt(np.mean(enhanced_normalized ** 2))), 4),
    }
    return enhanced_normalized, info


def plot_speech_overview(
    audio: np.ndarray,
    sample_rate: int,
    metrics: dict,
    output_path: Path,
) -> Path:
    """Generate high-resolution 6-panel overview chart for speech audio."""
    t = np.arange(len(audio)) / sample_rate
    times_vad = metrics["times"]
    rms_vad = metrics["rms_frames"]
    vad_mask = metrics["vad_mask"]

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor("#0F172A")  # Deep slate navy

    title_text = (
        f"BÁO CÁO PHÂN TÍCH ÂM THANH GIỌNG NÓI (VOICE & ACOUSTIC ANALYSIS)\n"
        f"File: Himn.m4a | Thời lượng: {metrics['duration_seconds']}s | Fs: {sample_rate} Hz | "
        f"Cao độ F0 trung bình: {metrics['mean_pitch_f0_hz']} Hz | SNR ước lượng: {metrics['estimated_snr_db']} dB"
    )
    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#F8FAFC", y=0.98)

    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.2], hspace=0.35, wspace=0.22,
                          left=0.07, right=0.95, top=0.91, bottom=0.06)

    # 1. Waveform with Short-Time Energy & VAD Segmentation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1E293B")
    ax1.plot(t, audio, color="#38BDF8", alpha=0.6, linewidth=0.7, label="Dạng sóng âm thanh")
    ax1.plot(times_vad, rms_vad * 2.0, color="#FBBF24", linewidth=1.5, label="Đường bao năng lượng RMS (x2)")
    # Highlight speech active region
    if np.any(vad_mask):
        ax1.axvspan(metrics["speech_start_sec"], metrics["speech_end_sec"],
                    color="#34D399", alpha=0.18, label="Vùng giọng nói tích cực (VAD Speech)")
    ax1.set_title("1. Dạng sóng toàn phần & Phân đoạn giọng nói (Waveform & VAD)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax1.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax1.set_ylabel("Biên độ chuẩn hóa", fontsize=9, color="#94A3B8")
    ax1.set_xlim(0, metrics["duration_seconds"])
    ax1.tick_params(colors="#94A3B8", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 2. Zoomed-in Voiced Speech (Vocal cord vibration ~50ms)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1E293B")
    zoom_start = min(metrics["speech_start_sec"] + 0.15, metrics["duration_seconds"] - 0.06)
    zoom_end = zoom_start + 0.05  # 50ms window
    idx_s, idx_e = int(zoom_start * sample_rate), int(zoom_end * sample_rate)
    t_zoom_ms = (t[idx_s:idx_e] - zoom_start) * 1000.0
    ax2.plot(t_zoom_ms, audio[idx_s:idx_e], color="#34D399", linewidth=1.8, label="Dao động thanh quản hữu thanh")
    ax2.set_title(f"2. Cận cảnh chu kỳ giọng nói [50ms: {zoom_start:.2f}s - {zoom_end:.2f}s]", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax2.set_xlabel("Thời gian tương đối (ms)", fontsize=9, color="#94A3B8")
    ax2.set_ylabel("Biên độ", fontsize=9, color="#94A3B8")
    ax2.tick_params(colors="#94A3B8", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 3. FFT Magnitude Spectrum (Speech vs Ambient Noise)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1E293B")
    s_idx = int(metrics["speech_start_sec"] * sample_rate)
    e_idx = int(metrics["speech_end_sec"] * sample_rate)
    speech_part = audio[s_idx:e_idx] if e_idx > s_idx else audio
    noise_part = audio[:int(0.55 * sample_rate)]

    f_s, psd_s = signal.welch(speech_part, fs=sample_rate, nperseg=2048)
    _, psd_n = signal.welch(noise_part, fs=sample_rate, nperseg=2048)
    mask_f = f_s <= 4000
    db_s = 10 * np.log10(np.maximum(psd_s, 1e-12))
    db_n = 10 * np.log10(np.maximum(psd_n, 1e-12))

    ax3.plot(f_s[mask_f], db_s[mask_f], color="#38BDF8", linewidth=1.4, label="Phổ đoạn Tiếng nói (Speech)")
    ax3.plot(f_s[mask_f], db_n[mask_f], color="#F87171", linewidth=1.1, alpha=0.75, linestyle="--", label="Phổ Nhiễu môi trường (Ambient Noise Floor)")
    ax3.set_title("3. Phổ tần số biên độ: Giọng nói vs Nhiễu nền [0 - 4000 Hz]", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax3.set_xlabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax3.set_ylabel("Mật độ phổ biên độ (dB/Hz)", fontsize=9, color="#94A3B8")
    ax3.set_xlim(0, 4000)
    ax3.tick_params(colors="#94A3B8", labelsize=8)
    ax3.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 4. Pitch Contour (F0 Fundamental Frequency Tracking)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1E293B")
    f0_arr = metrics["f0_frames"]
    valid_mask = ~np.isnan(f0_arr)
    if np.any(valid_mask):
        ax4.scatter(times_vad[valid_mask], f0_arr[valid_mask], color="#FBBF24", s=18, label=f"Cao độ F0 (Trung bình: {metrics['mean_pitch_f0_hz']:.1f} Hz)")
        ax4.plot(times_vad[valid_mask], f0_arr[valid_mask], color="#F59E0B", alpha=0.6, linewidth=1.0)
    ax4.set_title("4. Đường cong cao độ giọng nói (Pitch Contour F0 Tracking)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax4.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax4.set_ylabel("Tần số cơ bản F0 (Hz)", fontsize=9, color="#94A3B8")
    ax4.set_xlim(0, metrics["duration_seconds"])
    ax4.set_ylim(60, 250)
    ax4.tick_params(colors="#94A3B8", labelsize=8)
    ax4.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 5. STFT Spectrogram (Full Voice Formant Representation)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor("#0F172A")
    f_stft, t_stft, Sxx = signal.spectrogram(audio, fs=sample_rate, nperseg=1024, noverlap=768)
    pcm5 = ax5.pcolormesh(t_stft, f_stft, 10 * np.log10(np.maximum(Sxx, 1e-10)),
                          shading="gouraud", cmap="inferno", vmin=-90, vmax=-10)
    ax5.set_title("5. Spectrogram thời gian - tần số: Cấu trúc Formant và Hài âm giọng nói", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax5.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax5.set_ylabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax5.set_ylim(0, 5000)
    ax5.tick_params(colors="#94A3B8", labelsize=8)
    cb5 = fig.colorbar(pcm5, ax=ax5, orientation="horizontal", pad=0.18, shrink=0.45)
    cb5.set_label("Năng lượng phổ (dB/Hz)", fontsize=8, color="#94A3B8")
    cb5.ax.tick_params(labelsize=7, colors="#94A3B8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_speech_enhancement(
    original: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    enh_info: dict,
    output_path: Path,
) -> Path:
    """Generate Before vs After visual enhancement comparison figure."""
    min_len = min(len(original), len(enhanced))
    orig = original[:min_len]
    enh = enhanced[:min_len]
    t = np.arange(min_len) / sample_rate

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 11), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    title_text = (
        f"KẾT QUẢ XỬ LÝ KHỬ NHIỄU & NÂNG CAO CHẤT LƯỢNG GIỌNG NÓI (SPEECH ENHANCEMENT)\n"
        f"Mức suy giảm nhiễu nền môi trường: -{enh_info['noise_attenuation_db']} dB | "
        f"Chuẩn hóa đỉnh: {enh_info['enhanced_peak']} (-1.0 dBFS) | Năng lượng RMS: {enh_info['enhanced_rms']}"
    )
    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#F8FAFC", y=0.98)

    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1], hspace=0.35, wspace=0.22,
                          left=0.07, right=0.95, top=0.91, bottom=0.06)

    # 1. Waveform Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1E293B")
    ax1.plot(t, orig, color="#F87171", alpha=0.5, linewidth=0.7, label="Gốc (chứa nhiễu nền ở khoảng lặng)")
    ax1.plot(t, enh, color="#34D399", alpha=0.85, linewidth=0.8, label="Đã nâng cao (làm sạch khoảng lặng & chuẩn hóa)")
    ax1.set_title("1. Dạng sóng so sánh: Gốc vs Đã xử lý lọc", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax1.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax1.set_ylabel("Biên độ", fontsize=9, color="#94A3B8")
    ax1.set_xlim(0, t[-1])
    ax1.tick_params(colors="#94A3B8", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 2. Quiet Segment Zoom-in (Ambient noise comparison)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1E293B")
    q_s, q_e = int(0.1 * sample_rate), int(0.3 * sample_rate)
    t_q_ms = (t[q_s:q_e] - 0.1) * 1000
    ax2.plot(t_q_ms, orig[q_s:q_e], color="#F87171", alpha=0.7, linewidth=1.0, label="Nhiễu nền phòng gốc")
    ax2.plot(t_q_ms, enh[q_s:q_e], color="#34D399", alpha=0.9, linewidth=1.2, label=f"Nhiễu nền sau lọc (-{enh_info['noise_attenuation_db']} dB)")
    ax2.set_title("2. Cận cảnh đoạn khoảng lặng: Mức độ triệt tiêu nhiễu nền phòng", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax2.set_xlabel("Thời gian (ms)", fontsize=9, color="#94A3B8")
    ax2.set_ylabel("Biên độ", fontsize=9, color="#94A3B8")
    ax2.tick_params(colors="#94A3B8", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 3. Ambient Noise Spectrum before & after
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#1E293B")
    noise_orig = orig[:int(0.55 * sample_rate)]
    noise_enh = enh[:int(0.55 * sample_rate)]
    f_w, p_orig = signal.welch(noise_orig, fs=sample_rate, nperseg=2048)
    _, p_enh = signal.welch(noise_enh, fs=sample_rate, nperseg=2048)
    m_f = f_w <= 4000
    ax3.plot(f_w[m_f], 10 * np.log10(np.maximum(p_orig[m_f], 1e-12)), color="#F87171", linewidth=1.2, label="Phổ nhiễu môi trường gốc")
    ax3.plot(f_w[m_f], 10 * np.log10(np.maximum(p_enh[m_f], 1e-12)), color="#34D399", linewidth=1.4, label="Phổ nhiễu sau khi khử phổ Spectral Gating")
    ax3.set_title("3. Mức độ suy giảm phổ nhiễu môi trường (Noise Floor Reduction Spectrum)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax3.set_xlabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax3.set_ylabel("Mật độ phổ (dB/Hz)", fontsize=9, color="#94A3B8")
    ax3.set_xlim(0, 4000)
    ax3.tick_params(colors="#94A3B8", labelsize=8)
    ax3.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 4. Spectrogram Original
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#0F172A")
    f_stft, t_stft, Sxx_orig = signal.spectrogram(orig, fs=sample_rate, nperseg=1024, noverlap=768)
    pcm4 = ax4.pcolormesh(t_stft, f_stft, 10 * np.log10(np.maximum(Sxx_orig, 1e-10)),
                          shading="gouraud", cmap="inferno", vmin=-85, vmax=-15)
    ax4.set_title("4. Spectrogram Trước xử lý (Chứa nhiễu phòng ở khoảng lặng)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax4.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax4.set_ylabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax4.set_ylim(0, 5000)
    ax4.tick_params(colors="#94A3B8", labelsize=8)

    # 5. Spectrogram Enhanced
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#0F172A")
    _, _, Sxx_enh = signal.spectrogram(enh, fs=sample_rate, nperseg=1024, noverlap=768)
    pcm5 = ax5.pcolormesh(t_stft, f_stft, 10 * np.log10(np.maximum(Sxx_enh, 1e-10)),
                          shading="gouraud", cmap="inferno", vmin=-85, vmax=-15)
    ax5.set_title("5. Spectrogram Sau xử lý (Nền tĩnh, giữ nguyên dải Formant)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax5.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax5.set_ylabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax5.set_ylim(0, 5000)
    ax5.tick_params(colors="#94A3B8", labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def process_speech_pipeline(
    input_path: str | Path,
    output_dir: str | Path = "results/himn_analysis",
) -> dict:
    """Full execution pipeline for real speech audio (Himn.m4a)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Đang giải mã và đọc file âm thanh: {input_path}")
    audio, sr, meta = load_audio(input_path)
    print(f"  ✓ Tần số lấy mẫu: {sr} Hz, Số mẫu: {len(audio):,}, Thời lượng: {len(audio)/sr:.2f}s")

    print("[2/4] Đang tính toán các chỉ số âm học (VAD, Năng lượng, Cao độ F0, Formant)...")
    metrics = compute_speech_metrics(audio, sr)
    print(f"  ✓ Phân đoạn giọng nói: từ {metrics['speech_start_sec']}s đến {metrics['speech_end_sec']}s")
    print(f"  ✓ Cao độ giọng nói F0 trung bình: {metrics['mean_pitch_f0_hz']} Hz (Dải: {metrics['min_pitch_hz']} - {metrics['max_pitch_hz']} Hz)")
    print(f"  ✓ Tỷ số tín hiệu / nhiễu môi trường (SNR ước lượng): {metrics['estimated_snr_db']} dB")
    print(f"  ✓ Các đỉnh Formant chính: {metrics['formants_hz']} Hz")

    print("[3/4] Đang thực hiện xử lý khử nhiễu Spectral Gating & Chuẩn hóa âm lượng...")
    enhanced, enh_info = enhance_speech(audio, sr)
    print(f"  ✓ Mức suy giảm nhiễu nền: -{enh_info['noise_attenuation_db']} dB")
    print(f"  ✓ Biên độ đỉnh sau chuẩn hóa: {enh_info['enhanced_peak']} (-1.0 dBFS)")

    # Save enhanced audio
    enhanced_wav = out_dir / "Himn_enhanced.wav"
    save_wav(enhanced_wav, enhanced, sr)
    print(f"  ✓ Đã lưu file âm thanh đã khử nhiễu: {enhanced_wav}")

    print("[4/4] Đang vẽ các đồ thị phân tích & đối chiếu trực quan...")
    fig_overview = out_dir / "himn_voice_analysis_overview.png"
    fig_enhancement = out_dir / "himn_enhancement_comparison.png"

    plot_speech_overview(audio, sr, metrics, fig_overview)
    plot_speech_enhancement(audio, enhanced, sr, enh_info, fig_enhancement)
    print(f"  ✓ Đã lưu đồ thị phân tích giọng nói: {fig_overview}")
    print(f"  ✓ Đã lưu đồ thị đối chiếu nâng cao: {fig_enhancement}")

    # Build summary report JSON
    # Remove numpy arrays before serializing
    metrics_export = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    summary_report = {
        "input_metadata": meta,
        "speech_analysis": metrics_export,
        "enhancement_results": {
            "output_file": str(enhanced_wav),
            **enh_info,
        },
        "plots_saved": [
            str(fig_overview),
            str(fig_enhancement),
        ],
    }

    report_path = out_dir / "himn_speech_report.json"
    report_path.write_text(json.dumps(summary_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Hoàn tất! Báo cáo âm học lưu tại: {report_path}")
    return summary_report


# =====================================================================
# SYNTHETIC CLEAN VS NOISY PROCESSING (From previous step)
# =====================================================================

def compute_metrics(clean: np.ndarray, noisy: np.ndarray, sample_rate: int) -> dict:
    """Compute comprehensive signal and noise metrics for clean vs noisy pair."""
    min_len = min(len(clean), len(noisy))
    c = clean[:min_len]
    n = noisy[:min_len]
    noise = n - c

    power_clean = np.mean(c ** 2)
    power_noise = np.mean(noise ** 2)
    power_noisy = np.mean(n ** 2)

    snr_db = 10.0 * np.log10(power_clean / max(1e-12, power_noise))
    rms_clean = np.sqrt(power_clean)
    rms_noisy = np.sqrt(power_noisy)
    rms_noise = np.sqrt(power_noise)

    peak_clean = np.max(np.abs(c))
    peak_noisy = np.max(np.abs(n))
    peak_noise = np.max(np.abs(noise))

    crest_clean = peak_clean / max(1e-12, rms_clean)
    crest_noisy = peak_noisy / max(1e-12, rms_noisy)

    mse = np.mean(noise ** 2)
    mae = np.mean(np.abs(noise))

    fft_c = np.abs(np.fft.rfft(c))
    freqs = np.fft.rfftfreq(len(c), 1.0 / sample_rate)
    top_indices = np.argsort(fft_c)[-10:][::-1]
    significant_peaks = []
    max_mag = np.max(fft_c)
    for idx in top_indices:
        if freqs[idx] > 20 and fft_c[idx] > 0.05 * max_mag:
            significant_peaks.append(round(float(freqs[idx]), 1))
    significant_peaks = sorted(list(set(significant_peaks)))

    return {
        "duration_seconds": round(min_len / sample_rate, 3),
        "sample_rate": sample_rate,
        "num_samples": min_len,
        "snr_db": round(float(snr_db), 2),
        "rms_clean": round(float(rms_clean), 4),
        "rms_noisy": round(float(rms_noisy), 4),
        "rms_noise": round(float(rms_noise), 4),
        "peak_clean": round(float(peak_clean), 4),
        "peak_noisy": round(float(peak_noisy), 4),
        "peak_noise": round(float(peak_noise), 4),
        "crest_factor_clean": round(float(crest_clean), 2),
        "crest_factor_noisy": round(float(crest_noisy), 2),
        "mse": float(f"{mse:.6e}"),
        "mae": float(f"{mae:.6e}"),
        "clean_harmonic_peaks_hz": significant_peaks,
        "noise_mean": round(float(np.mean(noise)), 6),
        "noise_std": round(float(np.std(noise)), 6),
    }


def denoise_adaptive_wiener(noisy: np.ndarray, sample_rate: int) -> np.ndarray:
    """Adaptive STFT Wiener Filter for noise floor suppression."""
    nperseg = 1024
    noverlap = nperseg * 3 // 4
    f, t, Zxx = signal.stft(noisy, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    mag2 = np.abs(Zxx) ** 2
    high_freq_mask = f > 1200
    if np.any(high_freq_mask):
        noise_floor = np.median(mag2[high_freq_mask, :])
    else:
        noise_floor = np.percentile(mag2, 10)

    gain = np.maximum(0.02, 1.0 - 1.6 * noise_floor / np.maximum(mag2, 1e-12))
    gain = signal.medfilt2d(gain, kernel_size=[3, 3])
    Zxx_clean = Zxx * gain
    _, denoised = signal.istft(Zxx_clean, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    return denoised[:len(noisy)]


def denoise_harmonic_bandpass(noisy: np.ndarray, sample_rate: int, peaks: list[float] | None = None) -> np.ndarray:
    """Targeted Multiband Butterworth Bandpass filter."""
    if not peaks:
        peaks = [210.0, 420.0]
    filtered_total = np.zeros_like(noisy)
    for p in peaks:
        low = max(20.0, p - 18.0)
        high = min(sample_rate / 2.0 - 20.0, p + 18.0)
        sos = signal.butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        filtered_total += signal.sosfiltfilt(sos, noisy)
    return filtered_total


def plot_comprehensive_overview(clean: np.ndarray, noisy: np.ndarray, sample_rate: int, metrics: dict, output_path: Path) -> Path:
    """Generate high-resolution 6-panel comprehensive audio comparison figure."""
    t = np.arange(len(clean)) / sample_rate
    noise = noisy[:len(clean)] - clean

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    title_text = (
        f"BÁO CÁO PHÂN TÍCH & ĐỐI CHIẾU ÂM THANH: CLEAN vs NOISY (SNR = {metrics['snr_db']} dB)\n"
        f"Sampling Rate: {sample_rate} Hz | Thời lượng: {metrics['duration_seconds']}s | Số mẫu: {metrics['num_samples']:,}"
    )
    fig.suptitle(title_text, fontsize=15, fontweight="bold", color="#F8FAFC", y=0.98)

    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.2], hspace=0.35, wspace=0.22,
                          left=0.07, right=0.95, top=0.91, bottom=0.06)

    # 1. Full Waveform
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1E293B")
    ax1.plot(t, noisy, color="#F87171", alpha=0.65, linewidth=0.8, label=f"Noisy (RMS: {metrics['rms_noisy']:.3f})")
    ax1.plot(t, clean, color="#38BDF8", alpha=0.95, linewidth=1.0, label=f"Clean (RMS: {metrics['rms_clean']:.3f})")
    ax1.set_title("1. Dạng sóng toàn phần (Full Waveform - Time Domain)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax1.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax1.set_ylabel("Biên độ chuẩn hóa [-1, 1]", fontsize=9, color="#94A3B8")
    ax1.set_xlim(0, metrics['duration_seconds'])
    ax1.set_ylim(-1.05, 1.05)
    ax1.tick_params(colors="#94A3B8", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 2. Zoomed-in Waveform
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1E293B")
    t_start, t_end = 0.20, 0.24
    idx_start, idx_end = int(t_start * sample_rate), int(t_end * sample_rate)
    t_zoom = t[idx_start:idx_end] * 1000
    ax2.plot(t_zoom, noisy[idx_start:idx_end], color="#F87171", alpha=0.8, linewidth=1.2, label="Noisy (răng cưa nhiễu)")
    ax2.plot(t_zoom, clean[idx_start:idx_end], color="#38BDF8", alpha=1.0, linewidth=2.0, label="Clean (sin điều hòa mượt)")
    ax2.plot(t_zoom, noise[idx_start:idx_end], color="#FACC15", alpha=0.7, linestyle="--", linewidth=1.0, label="Noise riêng biệt")
    ax2.set_title("2. Cận cảnh chu kỳ sóng [0.20s - 0.24s] (Zoom-in Waveform)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax2.set_xlabel("Thời gian (mili giây - ms)", fontsize=9, color="#94A3B8")
    ax2.set_ylabel("Biên độ", fontsize=9, color="#94A3B8")
    ax2.tick_params(colors="#94A3B8", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 3. FFT Magnitude Spectrum
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1E293B")
    fft_clean = np.abs(np.fft.rfft(clean)) / (len(clean) / 2)
    fft_noisy = np.abs(np.fft.rfft(noisy)) / (len(noisy) / 2)
    fft_noise = np.abs(np.fft.rfft(noise)) / (len(noise) / 2)
    freqs = np.fft.rfftfreq(len(clean), 1.0 / sample_rate)
    db_clean = 20 * np.log10(np.maximum(fft_clean, 1e-6))
    db_noisy = 20 * np.log10(np.maximum(fft_noisy, 1e-6))
    db_noise = 20 * np.log10(np.maximum(fft_noise, 1e-6))

    f_max = 2000
    mask_f = freqs <= f_max
    ax3.plot(freqs[mask_f], db_noisy[mask_f], color="#F87171", alpha=0.6, linewidth=0.9, label="Noisy (mức sàn nhiễu cao)")
    ax3.plot(freqs[mask_f], db_noise[mask_f], color="#FACC15", alpha=0.45, linewidth=0.7, linestyle=":", label="Noise Floor")
    ax3.plot(freqs[mask_f], db_clean[mask_f], color="#38BDF8", alpha=0.95, linewidth=1.4, label="Clean (đỉnh hài sắc nét)")
    for p in metrics["clean_harmonic_peaks_hz"]:
        if p <= f_max:
            ax3.annotate(f"{p:.0f} Hz", xy=(p, db_clean[int(p * len(clean) / sample_rate)]),
                         xytext=(p + 40, db_clean[int(p * len(clean) / sample_rate)] + 6),
                         color="#34D399", fontsize=8, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="#34D399", lw=1.2))
    ax3.set_title("3. Phổ biên độ tần số (FFT Magnitude Spectrum [0 - 2000 Hz])", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax3.set_xlabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax3.set_ylabel("Biên độ (dB FS)", fontsize=9, color="#94A3B8")
    ax3.set_xlim(0, f_max)
    ax3.set_ylim(-85, 5)
    ax3.tick_params(colors="#94A3B8", labelsize=8)
    ax3.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 4. Power Spectral Density (PSD - Welch)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1E293B")
    f_welch, psd_clean = signal.welch(clean, fs=sample_rate, nperseg=1024)
    _, psd_noisy = signal.welch(noisy, fs=sample_rate, nperseg=1024)
    _, psd_noise = signal.welch(noise, fs=sample_rate, nperseg=1024)
    mask_w = f_welch <= f_max
    ax4.semilogy(f_welch[mask_w], psd_noisy[mask_w], color="#F87171", alpha=0.7, label="PSD Noisy")
    ax4.semilogy(f_welch[mask_w], psd_noise[mask_w], color="#FACC15", alpha=0.5, linestyle="--", label="PSD Noise")
    ax4.semilogy(f_welch[mask_w], psd_clean[mask_w], color="#38BDF8", alpha=0.95, label="PSD Clean")
    ax4.set_title("4. Mật độ phổ công suất (Power Spectral Density - Welch)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax4.set_xlabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax4.set_ylabel("Công suất / Tần số (V²/Hz)", fontsize=9, color="#94A3B8")
    ax4.set_xlim(0, f_max)
    ax4.tick_params(colors="#94A3B8", labelsize=8)
    ax4.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 5. Spectrogram Clean
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor("#0F172A")
    f_stft, t_stft, Sxx_clean = signal.spectrogram(clean, fs=sample_rate, nperseg=512, noverlap=384)
    pcm5 = ax5.pcolormesh(t_stft, f_stft, 10 * np.log10(np.maximum(Sxx_clean, 1e-10)),
                          shading="gouraud", cmap="magma", vmin=-90, vmax=-10)
    ax5.set_title("5. Spectrogram: Âm thanh Clean (Vạch tần số rõ nét)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax5.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax5.set_ylabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax5.set_ylim(0, 4000)
    ax5.tick_params(colors="#94A3B8", labelsize=8)
    cb5 = fig.colorbar(pcm5, ax=ax5, orientation="horizontal", pad=0.18, shrink=0.85)
    cb5.set_label("Năng lượng phổ (dB/Hz)", fontsize=8, color="#94A3B8")
    cb5.ax.tick_params(labelsize=7, colors="#94A3B8")

    # 6. Spectrogram Noisy
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor("#0F172A")
    _, _, Sxx_noisy = signal.spectrogram(noisy, fs=sample_rate, nperseg=512, noverlap=384)
    pcm6 = ax6.pcolormesh(t_stft, f_stft, 10 * np.log10(np.maximum(Sxx_noisy, 1e-10)),
                          shading="gouraud", cmap="magma", vmin=-90, vmax=-10)
    ax6.set_title("6. Spectrogram: Âm thanh Noisy (Nhiễu nền trắng phủ khắp băng tần)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax6.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax6.set_ylabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax6.set_ylim(0, 4000)
    ax6.tick_params(colors="#94A3B8", labelsize=8)
    cb6 = fig.colorbar(pcm6, ax=ax6, orientation="horizontal", pad=0.18, shrink=0.85)
    cb6.set_label("Năng lượng phổ (dB/Hz)", fontsize=8, color="#94A3B8")
    cb6.ax.tick_params(labelsize=7, colors="#94A3B8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_denoising_comparison(clean: np.ndarray, noisy: np.ndarray, denoised_bp: np.ndarray, denoised_wiener: np.ndarray, sample_rate: int, metrics_init: dict, output_path: Path) -> Path:
    """Generate detailed denoising visual comparison plot: Clean vs Noisy vs Processed."""
    min_len = min(len(clean), len(noisy), len(denoised_bp), len(denoised_wiener))
    c = clean[:min_len]
    n = noisy[:min_len]
    dbp = denoised_bp[:min_len]
    dw = denoised_wiener[:min_len]
    t = np.arange(min_len) / sample_rate

    snr_bp = 10.0 * np.log10(np.mean(c ** 2) / max(1e-12, np.mean((dbp - c) ** 2)))
    snr_wiener = 10.0 * np.log10(np.mean(c ** 2) / max(1e-12, np.mean((dw - c) ** 2)))

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 11), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    title_text = (
        f"KẾT QUẢ XỬ LÝ KHỬ NHIỄU ÂM THANH (AUDIO DENOISING PERFORMANCE)\n"
        f"SNR Gốc: {metrics_init['snr_db']:.2f} dB  ->  "
        f"Sau Lọc Wiener: {snr_wiener:.2f} dB (+{snr_wiener - metrics_init['snr_db']:.2f} dB)  ->  "
        f"Sau Lọc Bandpass: {snr_bp:.2f} dB (+{snr_bp - metrics_init['snr_db']:.2f} dB)"
    )
    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#F8FAFC", y=0.98)

    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1], hspace=0.35, wspace=0.22,
                          left=0.07, right=0.95, top=0.91, bottom=0.06)

    # 1. Full comparison waveform
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1E293B")
    ax1.plot(t, n, color="#F87171", alpha=0.45, linewidth=0.8, label=f"Noisy (SNR = {metrics_init['snr_db']} dB)")
    ax1.plot(t, dbp, color="#34D399", alpha=0.85, linewidth=1.0, label=f"Denoised Bandpass (SNR = {snr_bp:.1f} dB)")
    ax1.plot(t, c, color="#38BDF8", alpha=0.95, linewidth=0.8, linestyle=":", label="Clean Reference")
    ax1.set_title("1. Dạng sóng toàn phần: Noisy vs Đã khử nhiễu vs Clean", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax1.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax1.set_ylabel("Biên độ", fontsize=9, color="#94A3B8")
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(-1.05, 1.05)
    ax1.tick_params(colors="#94A3B8", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 2. Zoomed-in Waveform Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1E293B")
    t_start, t_end = 0.50, 0.53
    s_idx, e_idx = int(t_start * sample_rate), int(t_end * sample_rate)
    t_ms = (t[s_idx:e_idx] - t_start) * 1000
    ax2.plot(t_ms, n[s_idx:e_idx], color="#F87171", alpha=0.5, linewidth=1.1, label="Noisy (chứa gai nhiễu)")
    ax2.plot(t_ms, dw[s_idx:e_idx], color="#FBBF24", alpha=0.8, linewidth=1.3, label="Denoised Wiener")
    ax2.plot(t_ms, dbp[s_idx:e_idx], color="#34D399", alpha=0.9, linewidth=1.8, label="Denoised Bandpass (khôi phục sin chuẩn)")
    ax2.plot(t_ms, c[s_idx:e_idx], color="#38BDF8", alpha=1.0, linewidth=1.4, linestyle="--", label="Clean Chuẩn")
    ax2.set_title("2. Cận cảnh chu kỳ sóng phục hồi [30ms]", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax2.set_xlabel("Thời gian tương đối (ms)", fontsize=9, color="#94A3B8")
    ax2.set_ylabel("Biên độ", fontsize=9, color="#94A3B8")
    ax2.tick_params(colors="#94A3B8", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 3. Residual Error Signal
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1E293B")
    err_before = n - c
    err_after = dbp - c
    ax3.plot(t, err_before, color="#F87171", alpha=0.45, linewidth=0.7, label=f"Sai số trước lọc (RMS: {np.sqrt(np.mean(err_before**2)):.4f})")
    ax3.plot(t, err_after, color="#34D399", alpha=0.85, linewidth=0.8, label=f"Sai số sau lọc (RMS: {np.sqrt(np.mean(err_after**2)):.4f})")
    ax3.set_title("3. Tín hiệu sai số (Residual Error e(t) = y(t) - clean)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax3.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax3.set_ylabel("Độ lệch biên độ", fontsize=9, color="#94A3B8")
    ax3.set_xlim(0, t[-1])
    ax3.tick_params(colors="#94A3B8", labelsize=8)
    ax3.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 4. FFT Spectrum comparison after denoising
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1E293B")
    freqs = np.fft.rfftfreq(min_len, 1.0 / sample_rate)
    fft_n = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(n)) / (min_len / 2), 1e-6))
    fft_bp = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(dbp)) / (min_len / 2), 1e-6))
    fft_c = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(c)) / (min_len / 2), 1e-6))
    f_mask = freqs <= 2000
    ax4.plot(freqs[f_mask], fft_n[f_mask], color="#F87171", alpha=0.4, linewidth=0.8, label="Noisy (nhiễu nền cao)")
    ax4.plot(freqs[f_mask], fft_bp[f_mask], color="#34D399", alpha=0.9, linewidth=1.2, label="Denoised (hạ triệt để nhiễu nền)")
    ax4.plot(freqs[f_mask], fft_c[f_mask], color="#38BDF8", alpha=0.8, linewidth=1.0, linestyle="--", label="Clean")
    ax4.set_title("4. Phổ tần số FFT sau khử nhiễu (loại bỏ dải nhiễu trắng)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax4.set_xlabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax4.set_ylabel("Biên độ (dB FS)", fontsize=9, color="#94A3B8")
    ax4.set_xlim(0, 2000)
    ax4.set_ylim(-85, 5)
    ax4.tick_params(colors="#94A3B8", labelsize=8)
    ax4.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 5. Spectrogram Before (Noisy)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor("#0F172A")
    f_stft, t_stft, Sxx_n = signal.spectrogram(n, fs=sample_rate, nperseg=512, noverlap=384)
    pcm5 = ax5.pcolormesh(t_stft, f_stft, 10 * np.log10(np.maximum(Sxx_n, 1e-10)),
                          shading="gouraud", cmap="magma", vmin=-90, vmax=-10)
    ax5.set_title(f"5. Spectrogram trước xử lý: Noisy (SNR = {metrics_init['snr_db']} dB)",
                  fontsize=11, fontweight="bold", color="#E2E8F0")
    ax5.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax5.set_ylabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax5.set_ylim(0, 4000)
    ax5.tick_params(colors="#94A3B8", labelsize=8)
    cb5 = fig.colorbar(pcm5, ax=ax5, orientation="horizontal", pad=0.18, shrink=0.85)
    cb5.set_label("Năng lượng phổ (dB/Hz)", fontsize=8, color="#94A3B8")
    cb5.ax.tick_params(labelsize=7, colors="#94A3B8")

    # 6. Spectrogram After (Denoised Bandpass)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor("#0F172A")
    _, _, Sxx_denoised = signal.spectrogram(dbp, fs=sample_rate, nperseg=512, noverlap=384)
    pcm6 = ax6.pcolormesh(t_stft, f_stft, 10 * np.log10(np.maximum(Sxx_denoised, 1e-10)),
                          shading="gouraud", cmap="magma", vmin=-90, vmax=-10)
    ax6.set_title(f"6. Spectrogram sau xử lý: Đã khử nhiễu (SNR = {snr_bp:.1f} dB)",
                  fontsize=11, fontweight="bold", color="#E2E8F0")
    ax6.set_xlabel("Thời gian (giây)", fontsize=9, color="#94A3B8")
    ax6.set_ylabel("Tần số (Hz)", fontsize=9, color="#94A3B8")
    ax6.set_ylim(0, 4000)
    ax6.tick_params(colors="#94A3B8", labelsize=8)
    cb6 = fig.colorbar(pcm6, ax=ax6, orientation="horizontal", pad=0.18, shrink=0.85)
    cb6.set_label("Năng lượng phổ (dB/Hz)", fontsize=8, color="#94A3B8")
    cb6.ax.tick_params(labelsize=7, colors="#94A3B8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_noise_statistical_profile(clean: np.ndarray, noisy: np.ndarray, output_path: Path) -> Path:
    """Plot statistical distribution of noise: Histogram, Q-Q Plot, and Autocorrelation."""
    min_len = min(len(clean), len(noisy))
    noise = noisy[:min_len] - clean[:min_len]

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    # 1. Noise Histogram & Gaussian Fit
    ax1.set_facecolor("#1E293B")
    mu, std = np.mean(noise), np.std(noise)
    count, bins, _ = ax1.hist(noise, bins=70, density=True, alpha=0.75, color="#F87171", edgecolor="#991B1B",
                              label=f"Nhiễu thực tế (μ={mu:.4f}, σ={std:.4f})")
    x_pdf = np.linspace(bins[0], bins[-1], 200)
    p_pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - mu) / std) ** 2)
    ax1.plot(x_pdf, p_pdf, color="#38BDF8", linewidth=2.2, label="Đường cong Gaussian lý thuyết N(μ, σ²)")
    ax1.set_title("Phân phối xác suất của nhiễu (Noise Distribution & Normal Fit)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax1.set_xlabel("Biên độ nhiễu", fontsize=9, color="#94A3B8")
    ax1.set_ylabel("Mật độ xác suất (PDF)", fontsize=9, color="#94A3B8")
    ax1.tick_params(colors="#94A3B8", labelsize=8)
    ax1.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8)

    # 2. Autocorrelation of Noise
    ax2.set_facecolor("#1E293B")
    max_lag = 120
    n_norm = (noise - mu) / (std * np.sqrt(len(noise)))
    corr = signal.correlate(n_norm, n_norm, mode="full")
    lags = signal.correlation_lags(len(noise), len(noise))
    mask_lag = np.abs(lags) <= max_lag
    ax2.stem(lags[mask_lag], corr[mask_lag], linefmt="#FACC15", markerfmt="o", basefmt="gray")
    ax2.set_title("Hàm tự tương quan của nhiễu (Noise Autocorrelation - White Noise Test)", fontsize=11, fontweight="bold", color="#E2E8F0")
    ax2.set_xlabel("Độ trễ mẫu (Lag k)", fontsize=9, color="#94A3B8")
    ax2.set_ylabel("Hệ số tự tương quan r(k)", fontsize=9, color="#94A3B8")
    ax2.set_ylim(-0.2, 1.1)
    ax2.tick_params(colors="#94A3B8", labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def process_audio_pipeline(clean_path: str | Path, noisy_path: str | Path, output_dir: str | Path = "results/audio_analysis") -> dict:
    """Full execution pipeline for Clean vs Noisy comparison."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Đang đọc file âm thanh:\n  - Clean: {clean_path}\n  - Noisy: {noisy_path}")
    clean, sr_clean, _ = load_audio(clean_path)
    noisy, sr_noisy, _ = load_audio(noisy_path)

    if sr_clean != sr_noisy:
        raise ValueError(f"Khác biệt sampling rate: {sr_clean} vs {sr_noisy}")

    sr = sr_clean
    print(f"  ✓ Tần số lấy mẫu: {sr} Hz, Số mẫu: {len(clean):,}, Thời lượng: {len(clean)/sr:.2f}s")

    print("[2/5] Đang tính toán các chỉ số định lượng (SNR, RMS, FFT)...")
    metrics = compute_metrics(clean, noisy, sr)
    print(f"  ✓ Tỷ số tín hiệu trên nhiễu (SNR ban đầu): {metrics['snr_db']:.2f} dB")
    print(f"  ✓ Các đỉnh hài xác định được: {metrics['clean_harmonic_peaks_hz']} Hz")

    print("[3/5] Đang thực hiện xử lý khử nhiễu (Denoising)...")
    denoised_bp = denoise_harmonic_bandpass(noisy, sr, metrics["clean_harmonic_peaks_hz"])
    denoised_wiener = denoise_adaptive_wiener(noisy, sr)

    snr_bp = 10.0 * np.log10(np.mean(clean ** 2) / max(1e-12, np.mean((denoised_bp - clean) ** 2)))
    snr_wiener = 10.0 * np.log10(np.mean(clean ** 2) / max(1e-12, np.mean((denoised_wiener - clean) ** 2)))
    print(f"  ✓ SNR sau lọc Wiener thích nghi: {snr_wiener:.2f} dB (+{snr_wiener - metrics['snr_db']:.2f} dB)")
    print(f"  ✓ SNR sau lọc Bandpass điều hòa: {snr_bp:.2f} dB (+{snr_bp - metrics['snr_db']:.2f} dB)")

    file_denoised_bp = out_dir / "synthetic_denoised_bandpass.wav"
    file_denoised_wiener = out_dir / "synthetic_denoised_wiener.wav"
    save_wav(file_denoised_bp, denoised_bp, sr)
    save_wav(file_denoised_wiener, denoised_wiener, sr)

    print("[4/5] Đang vẽ đồ thị quan sát chất lượng cao...")
    fig_overview = out_dir / "audio_overview_comparison.png"
    fig_denoising = out_dir / "audio_denoising_results.png"
    fig_noise_stat = out_dir / "noise_statistical_profile.png"

    plot_comprehensive_overview(clean, noisy, sr, metrics, fig_overview)
    plot_denoising_comparison(clean, noisy, denoised_bp, denoised_wiener, sr, metrics, fig_denoising)
    plot_noise_statistical_profile(clean, noisy, fig_noise_stat)

    summary_report = {
        "inputs": {
            "clean_file": str(clean_path),
            "noisy_file": str(noisy_path),
            "sample_rate": sr,
            "duration_seconds": metrics["duration_seconds"],
        },
        "initial_metrics": metrics,
        "denoising_performance": {
            "wiener_filter": {
                "output_file": str(file_denoised_wiener),
                "snr_db": round(float(snr_wiener), 2),
                "snr_improvement_db": round(float(snr_wiener - metrics["snr_db"]), 2),
            },
            "harmonic_bandpass_filter": {
                "output_file": str(file_denoised_bp),
                "snr_db": round(float(snr_bp), 2),
                "snr_improvement_db": round(float(snr_bp - metrics["snr_db"]), 2),
            },
        },
        "plots_saved": [
            str(fig_overview),
            str(fig_denoising),
            str(fig_noise_stat),
        ],
    }

    report_path = out_dir / "audio_analysis_report.json"
    report_path.write_text(json.dumps(summary_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[5/5] Hoàn tất! Báo cáo tóm tắt đã lưu tại: {report_path}")
    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Chương trình xử lý âm thanh & vẽ đồ thị quan sát chuyên sâu")
    parser.add_argument("--input", "-i", type=str, default=None, help="Đường dẫn tệp âm thanh đơn (ví dụ Himn.m4a) để phân tích giọng nói & khử nhiễu")
    parser.add_argument("--clean", type=str, default="synthetic_clean.wav", help="Đường dẫn file WAV clean")
    parser.add_argument("--noisy", type=str, default="synthetic_noisy.wav", help="Đường dẫn file WAV noisy")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Thư mục xuất kết quả")
    args = parser.parse_args()

    if args.input:
        out_dir = args.output_dir or "results/himn_analysis"
        process_speech_pipeline(args.input, out_dir)
    else:
        out_dir = args.output_dir or "results/audio_analysis"
        process_audio_pipeline(args.clean, args.noisy, out_dir)


if __name__ == "__main__":
    main()
