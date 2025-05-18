import torch
import torch.nn.functional as F
from drum_label_map import DrumLabelMap
import librosa
import numpy as np

class Predictor:
    def __init__(self, model, device=None, desired_time_steps=128):
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.desired_time_steps = desired_time_steps

    def predict(self, filepath):
        # Ladda ljud och beräkna mel-spektrogram på samma sätt som i DrumDataset
        mel_spec = self._load_audio_to_mel(filepath)  # ska returnera tensor shape [mel_bins, time]

        # Pad eller trimma tidsdimensionen till desired_time_steps
        time_dim = mel_spec.shape[-1]
        if time_dim < self.desired_time_steps:
            pad_amount = self.desired_time_steps - time_dim
            mel_spec = F.pad(mel_spec, (0, pad_amount))  # pad sista dimensionen till höger
        elif time_dim > self.desired_time_steps:
            mel_spec = mel_spec[:, :self.desired_time_steps]  # trimma till önskad längd

        # Lägg till batch- och kanal-dimension
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, mel_bins, desired_time_steps]

        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = torch.softmax(outputs, dim=1).squeeze()
            pred_index = torch.argmax(probabilities).item()

        prediction = DrumLabelMap.categories[pred_index]
        prob_dict = {DrumLabelMap.categories[i]: probabilities[i].item() for i in range(len(DrumLabelMap.categories))}
        return prediction, prob_dict

    def _load_audio_to_mel(self, filepath):
        y, sr = librosa.load(filepath, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32)
        return mel_tensor

