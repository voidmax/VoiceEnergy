import torch
import librosa
import scipy.interpolate
import pyworld as pw
import numpy as np 
import json
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Processor
from scipy.stats import entropy
from collections import defaultdict


SAMPLE_RATE = 16000
RANDOM_CONST = 5513 * SAMPLE_RATE // 22050
HOP_MS = 250


class Denoiser:
    def __init__(self, device='cpu'):
        self.model = torch.hub.load("facebookresearch/denoiser", "dns64", force_reload=False).eval().to(device)
    
    def denoise(self, wav, sr=None, device='cpu'):
        with torch.no_grad():
            res = self.model(torch.from_numpy(wav).unsqueeze(0).to(device))
        return res.squeeze().cpu().numpy()


class FeatureGenerator:
    def __init__(self):
        pass

    def feature_extraction(self, audio, audio_key):
        raise NotImplementedError


class TempFeatureGenerator(FeatureGenerator):
    def __init__(self):
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    def feature_extraction(self, audio):
        inputs = self.processor([audio], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0]
        
        return {
            "temp_letters": sum(map(lambda x: x != ' ', text)),
            "temp_words":   sum(map(lambda x: x != '', text.split()))
        }


class SimpleFeatureGenerator(FeatureGenerator):
    def feature_extraction(self, audio, audio_key):
        return {
            "audio_rmse":    audio.std(),
            "audio_zcr":     librosa.feature.zero_crossing_rate(audio).mean(),
            "audio_entropy": entropy(audio**2 / (audio**2).sum())
        }


class SreFeatureGenerator(FeatureGenerator):
    def get_f0(self, wav, hop_ms, sr=SAMPLE_RATE, f_min=0, f_max=None):
        """
        Extract f0 (1d-array of frame values) from wav (1d-array of point values).
        Args:
        wav    - waveform (numpy array)
        sr     - sampling rate
        hop_ms - stride (in milliseconds) for frames
        f_min  - f0 floor frequency
        f_max  - f0 ceil frequency
        Returns:
        f0     - interpolated main frequency, shape (n_frames,) 
        """
            
        if f_max is None:
            f_max = sr / 2
        _f0, t = pw.dio(wav.astype(np.float64), sr, frame_period=hop_ms, f0_ceil=f_max) # raw pitch Generator
        f0 = pw.stonemask(wav.astype(np.float64), _f0, t, sr)  # pitch refinement

        return self.convert_continuos_f0(f0)[:, np.newaxis].astype(np.float32)

    def convert_continuos_f0(self, f0):
        uv = np.float32(f0 != 0)

        # get start and end of f0
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]

        # padding start and end of f0 sequence
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0

        # get non-zero frame index
        nz_frames = np.where(f0 != 0)[0]

        # perform linear interpolation
        f = scipy.interpolate.interp1d(nz_frames, f0[nz_frames])
        cont_f0 = f(np.arange(0, f0.shape[0]))

        return np.log(cont_f0)

    def get_loudness(self, wav, sr=SAMPLE_RATE, n_fft=1280, hop_length=320, win_length=None, ref=1.0, min_db=-80.0):
        """
        Extract the loudness measurement of the signal.
        Feature is extracted using A-weighting of the signal frequencies.
        Args:
            wav          - waveform (numpy array)
            sr           - sampling rate
            n_fft        - number of points for fft
            hop_length   - stride of stft
            win_length   - size of window of stft
            ref          - reference for amplitude log-scale
            min_db       - floor for db difference
        Returns:
            loudness     - loudness of signal, shape (n_frames,) 
        """

        A_weighting = librosa.A_weighting(librosa.fft_frequencies(sr, n_fft=n_fft) + 1e-6, min_db=min_db)
        weighting = 10**(A_weighting / 10)

        power_spec = abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length))**2
        loudness = np.mean(power_spec * weighting[:, None], axis=0)
        loudness = librosa.power_to_db(loudness, ref=ref) # in db

        return loudness[:, np.newaxis].astype(np.float32)

    def feature_extraction(self, audio, audio_key):
        loudness = self.get_loudness(audio, hop_length=RANDOM_CONST).squeeze()
        f0_audio = self.get_f0(audio, hop_ms=HOP_MS)[:(10000 // HOP_MS)].squeeze()
        return {
            "f0_mean":            f0_audio.squeeze().mean(),
            "f0_std":             f0_audio.squeeze().std(),
            "f0_mean_der":        np.abs(f0_audio.squeeze()[1:] - f0_audio.squeeze()[:-1]).mean(),
            "loudness_mean":      loudness.mean(),
            "loudness_std":       loudness.std(),
            "loudness_mean_der":  np.abs(loudness.squeeze()[1:] - loudness.squeeze()[:-1]).mean(),
        }
    

class LibrosaFeatureGenerator(FeatureGenerator):
    def feature_extraction(self, audio, audio_key):
        feature_mfccs = librosa.feature.mfcc(audio, sr=SAMPLE_RATE)
        feature_rms = librosa.feature.rms(y=audio)
        feature_chroma = librosa.feature.chroma_stft(audio, sr=SAMPLE_RATE)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=SAMPLE_RATE)
        chroma_cens = librosa.feature.chroma_cens(y=audio, sr=SAMPLE_RATE)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)
        flatness = librosa.feature.spectral_flatness(y=audio)
        return {
            "librosa_mfccs_mean":       np.mean(feature_mfccs),
            "librosa_rms_mean":         np.mean(feature_rms),
            "librosa_chroma_mean":      np.mean(feature_chroma),
            "librosa_centroid_mean":    np.mean(centroid),
            "librosa_tonnetz_mean":     np.mean(tonnetz),
            "librosa_chroma_cens_mean": np.mean(chroma_cens),
            "librosa_spec_bw_mean":     np.mean(spec_bw),
            "librosa_flatness_mean":    np.mean(flatness),
            "librosa_mfccs_std":        np.std(feature_mfccs),
            "librosa_rms_std":          np.std(feature_rms),
            "librosa_chroma_std":       np.std(feature_chroma),
            "librosa_centroid_std":     np.std(centroid),
            "librosa_tonnetz_std":      np.std(tonnetz),
            "librosa_chroma_cens_std":  np.std(chroma_cens),
            "librosa_spec_bw_std":      np.std(spec_bw),
            "librosa_flatness_std":     np.std(flatness),
        }


class EmotionsFeatureGenerator(FeatureGenerator):
    def __init__(self, emotions_path):
        self.info = json.load(open(emotions_path))

        self.emotions_mean = defaultdict(list)
        for chanks in self.info.values():
            current_mean = defaultdict(list)
            for chank in chanks:
                for key, value in chank["emotions_result"].items():
                    current_mean[key].append(value)
            for key in current_mean:
                self.emotions_mean[key].append(np.mean(current_mean[key]))
        for key in self.emotions_mean:
            self.emotions_mean[key] = np.mean(self.emotions_mean[key])

    def feature_extraction(self, audio, audio_key):
        if audio_key[0] not in self.info:
            return self.emotions_mean

        tL = audio_key[1] / SAMPLE_RATE
        tR = audio_key[2] / SAMPLE_RATE
        sumt = 0

        current_mean = defaultdict(float)
        for chank in self.info[audio_key[0]]:
            L = float(chank["results"][0]["start"][:-1])
            R = float(chank["results"][0]["end"][:-1])
            cross = min(tR, R) - max(tL, L)
            if cross <= 0:
                continue
            sumt += cross
            for key, value in chank["emotions_result"].items():
                current_mean[key] += value * cross
            

        if sumt == 0:
            return self.emotions_mean
        for key in current_mean:
            current_mean[key] = current_mean[key] / sumt
        return current_mean


class SpectralFeatureGenerator(FeatureGenerator):
    def feature_extraction(self, audio, audio_key):
        spec = np.abs(np.fft.rfft(audio))
        freq = np.fft.rfftfreq(len(audio), d=1 / SAMPLE_RATE)
        spec = np.abs(spec)
        amp = spec / spec.sum()
        mean = (freq * amp).sum()
        sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
        amp_cumsum = np.cumsum(amp)
        median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
        mode = freq[amp.argmax()]
        Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
        Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
        IQR = Q75 - Q25
        z = amp - amp.mean()
        w = amp.std()
        skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
        kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

        return {
            "spectral_mean": mean,
            "spectral_sd": sd,
            "spectral_median": median,
            "spectral_mode": mode,
            "spectral_Q25": Q25,
            "spectral_Q75": Q75,
            "spectral_IQR": IQR,
            "spectral_skew": skew,
            "spectral_kurt": kurt
        }