"""학습용 전처리 (벡터화) 와 음성 증강

기기 전처리(speakable_arm_ref/tools/ref_frontend.py)와 같은 계산을 numpy 벡터 연산으로 한다.
  PCM16 → HPF → 이득 정규화 → STFT power → Mel 40 → 64프레임 면적 평균 → log
정규화(평균·표준편차)는 학습 데이터 통계로 정하므로 여기서는 log Mel까지만 만든다.
테스트 벡터는 ref_frontend.py로 만들고, 이 구현과의 차이는 export 단계에서 확인한다.
"""
import json
import os
import numpy as np
from scipy.signal import fftconvolve, lfilter, resample_poly

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend_base")


class FastFrontend:
    def __init__(self, base_dir=BASE_DIR):
        self.p = json.load(open(os.path.join(base_dir, "preprocess.json"), encoding="utf8"))
        self.hann = np.load(os.path.join(base_dir, "hann.npy"))
        self.mel_filter = np.load(os.path.join(base_dir, "mel_filter.npy"))
        self._resize = {}

    def resize_matrix(self, t_in):
        """시간 축 면적 가중 평균을 행렬로: (T_in, 64)"""
        if t_in not in self._resize:
            t_out = self.p["time_frames"]
            w = np.zeros((t_in, t_out))
            for j in range(t_out):
                a, b = j * t_in / t_out, (j + 1) * t_in / t_out
                for i in range(int(np.floor(a)), int(np.ceil(b))):
                    ov = min(b, i + 1) - max(a, i)
                    if ov > 0:
                        w[i, j] = ov / (b - a)
            self._resize[t_in] = w
        return self._resize[t_in]

    def log_mel(self, pcm16):
        """pcm16: int16 배열 → log Mel (40, 64) float64"""
        p = self.p
        x = np.asarray(pcm16, dtype=np.float64) / p["pcm_scale"]
        y = lfilter([1.0, -1.0], [1.0, -p["hpf_a"]], x)            # y[n] = x[n] - x[n-1] + a·y[n-1]
        rms = max(np.sqrt(np.mean(y * y)), p["gain_silence_rms_floor"])
        g_db = np.clip(p["gain_target_rms_dbfs"] - 20.0 * np.log10(rms), p["gain_min_db"], p["gain_max_db"])
        g = 10.0 ** (g_db / 20.0)
        peak = np.max(np.abs(y)) * g
        if peak > p["gain_peak_ceiling"]:
            g *= p["gain_peak_ceiling"] / peak
        y = y * g
        fl, hop = p["frame_length"], p["hop_length"]
        frames = np.lib.stride_tricks.sliding_window_view(y, fl)[::hop]
        spec = np.abs(np.fft.rfft(frames * self.hann, n=p["n_fft"], axis=1)) ** 2
        mel = self.mel_filter @ spec.T                                # (40, T)
        mel64 = mel @ self.resize_matrix(mel.shape[1])
        return np.log(np.maximum(mel64, p["log_floor"]))


# ---------------------------------------------------------------- 증강
# 이득 정규화가 있으므로 음량 증강은 하지 않는다. 결과는 PCM16으로 다시 양자화해 기기 입력과 같게 만든다.

def _rms(x):
    return np.sqrt(np.mean(x * x) + 1e-12)


def colored_noise(rng, n):
    """백색~갈색 사이 무작위 기울기 잡음"""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec /= f ** rng.uniform(0.0, 1.0)                               # 진폭 기울기 0(백색) ~ 1(갈색)
    out = np.fft.irfft(spec, n=n)
    return out / _rms(out)


def synthetic_rir(rng, sr, rt60):
    n = int(sr * rt60)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    h[0] = 1.0
    return h / np.sqrt(np.sum(h * h))


class Augmenter:
    """설정 예: speed=[0.9,1.1], snr_db=[10,35], p_noise=0.8, p_reverb=0.3, rt60=[0.2,0.6],
    pad_s=[0,0.2], trim_s=[0,0.05], noise_files=[...16kHz mono wav/flac...]"""

    def __init__(self, cfg, sr=16000, min_samples=8000, max_samples=48000, read_audio=None):
        self.c = cfg
        self.sr = sr
        self.min_n, self.max_n = min_samples, max_samples
        self.noises = []
        for f in cfg.get("noise_files", []):
            a = read_audio(f).astype(np.float64) / 32768.0
            if len(a) > sr:
                self.noises.append(a)

    def _noise(self, rng, n):
        if self.noises and rng.random() < 0.7:
            src = self.noises[rng.integers(len(self.noises))]
            if len(src) <= n:
                src = np.tile(src, n // len(src) + 1)
            s = rng.integers(0, len(src) - n + 1)
            seg = src[s:s + n]
            return seg / _rms(seg)
        return colored_noise(rng, n)

    def __call__(self, pcm16, rng):
        c = self.c
        x = pcm16.astype(np.float64) / 32768.0
        lo, hi = c.get("speed", [1.0, 1.0])
        f = rng.uniform(lo, hi)
        if abs(f - 1.0) > 0.01:
            x = resample_poly(x, 100, int(round(100 * f)))
        if rng.random() < c.get("p_reverb", 0.0):
            h = synthetic_rir(rng, self.sr, rng.uniform(*c.get("rt60", [0.2, 0.6])))
            x = fftconvolve(x, h)[:len(x)]
        t0, t1 = (int(rng.uniform(*c.get("trim_s", [0, 0])) * self.sr) for _ in range(2))
        if len(x) - t0 - t1 >= self.min_n:
            x = x[t0:len(x) - t1]
        p0, p1 = (int(rng.uniform(*c.get("pad_s", [0, 0])) * self.sr) for _ in range(2))
        room = self.max_n - len(x)
        if room < 0:
            return None
        if p0 + p1 > room:
            p0, p1 = p0 * room // (p0 + p1), p1 * room // (p0 + p1)
        x = np.concatenate([np.zeros(p0), x, np.zeros(p1)])
        if rng.random() < c.get("p_noise", 0.0):
            snr = rng.uniform(*c.get("snr_db", [10, 35]))
            x = x + self._noise(rng, len(x)) * _rms(x) / 10 ** (snr / 20)
        else:
            x = x + rng.standard_normal(len(x)) * 10 ** (-70 / 20)     # 무음 구간을 완전한 0으로 두지 않음
        if len(x) < self.min_n:
            return None
        return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
