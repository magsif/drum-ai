import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

def count_images_per_class(folder_path):
    class_counts = {}
    for class_name in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            class_counts[class_name] = count
    return class_counts

def predict_sample(model, audio_path, class_indices):
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    # Load sound file
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Create figure
    fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, cmap='viridis')
    fig.canvas.draw()

    # Use buffer_rgba() instead of tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    X = np.asarray(buf)
    X = X[:, :, :3]  # drop alpha channel
    plt.close(fig)

    # Prepare input for model
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, axis=0)

    # Predict
    predictions = model.predict(X)
    predicted_index = np.argmax(predictions)
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_index]

    confidence = predictions[0][predicted_index]
    print(f"Predicted class: {predicted_label} (confidence: {confidence:.2f})")
    return predicted_label, confidence



def plot_training_history(history, title='Training History'):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predicted_labels, class_indices, title="Confusion Matrix"):
    labels = list(class_indices.keys())
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(labels)))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()
