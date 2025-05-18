import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_spectrogram_dataset(samples_dir, output_dir):
    metadata = []
    for label in os.listdir(samples_dir):
        label_dir = os.path.join(samples_dir, label)
        if not os.path.isdir(label_dir):
            continue

        label_output_dir = os.path.join(output_dir, label)
        os.makedirs(label_output_dir, exist_ok=True)

        for filename in os.listdir(label_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(label_dir, filename)
                y, sr = librosa.load(filepath, sr=None)
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_dB = librosa.power_to_db(S, ref=np.max)

                spectrogram_filename = f"{os.path.splitext(filename)[0]}.png"
                spectrogram_path = os.path.join(label_output_dir, spectrogram_filename)

                plt.figure(figsize=(2.56, 2.56), dpi=100)
                librosa.display.specshow(S_dB, sr=sr, cmap='viridis')
                plt.axis('off')
                plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
                plt.close()

                metadata.append({"file": spectrogram_filename, "label": label})

    metadata_path = os.path.join(output_dir, "metadata.csv")
    pd.DataFrame(metadata).to_csv(metadata_path, index=False)
    return metadata
