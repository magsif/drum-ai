import torch
import torch.nn.functional as F
import torchaudio
from drum_label_map import DrumLabelMap

class Predictor:
    def __init__(self, model, device=None, sample_rate=16000, n_mels=64, n_fft=400):
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.max_length = int(sample_rate * 1.0)  # 1 second

        # Define transforms once
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=n_fft // 2,
            n_mels=n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def predict(self, filepath):
        mel_spec = self._load_audio_to_mel(filepath)  # [1, n_mels, time]
        mel_spec = mel_spec.unsqueeze(0).to(self.device)  # [1, 1, n_mels, time]

        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = F.softmax(outputs, dim=1).squeeze()
            pred_index = torch.argmax(probabilities).item()

        prediction = DrumLabelMap.categories[pred_index]
        prob_dict = {DrumLabelMap.categories[i]: float(probabilities[i].cpu()) for i in range(len(DrumLabelMap.categories))}
        return prediction, prob_dict

    def _load_audio_to_mel(self, filepath):
        waveform, sr = torchaudio.load(filepath)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Pad or trim to 1 second
        if waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_length]

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return mel_spec
