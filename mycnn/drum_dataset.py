import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from drum_label_map import DrumLabelMap

class DrumDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, n_mels=64, n_fft=400):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = n_fft // 2
        self.max_length = int(sample_rate * 1.0)  # 1-second max length
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.audio_files = []
        self.labels = []
        
        for category in os.listdir(audio_dir):
            if category in DrumLabelMap.label_map and os.path.isdir(os.path.join(audio_dir, category)):
                category_dir = os.path.join(audio_dir, category)
                for file in os.listdir(category_dir):
                    if file.endswith(('.wav', '.mp3', '.ogg')):
                        self.audio_files.append(os.path.join(category_dir, file))
                        self.labels.append(DrumLabelMap.label_map[category])
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        waveform, sample_rate = torchaudio.load(audio_file)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        else:
            start = torch.randint(0, waveform.shape[1] - self.max_length + 1, (1,))
            waveform = waveform[:, start:start + self.max_length]
        
        melspec = self.mel_spec(waveform)
        melspec = self.amplitude_to_db(melspec)
        melspec = (melspec - melspec.mean()) / (melspec.std() + 1e-8)
        
        return {"input": melspec, "label": torch.tensor(label)}
