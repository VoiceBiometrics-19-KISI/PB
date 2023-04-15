import torch
import torchaudio as torchaudio
from datasets import load_dataset
from transformers import pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

VERSION = 2

# Audio Classification Pipeline
if VERSION == 1:
    dataset = load_dataset("anton-l/superb_demo", "si", split="test")
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-sid")
    predictions = classifier(dataset[1]["file"], top_k=5)
    print(predictions)


else:
    dataset = load_dataset("anton-l/superb_demo", "si", split="test")
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-sid")

    audio_files = ["one_voice/0.wav", "one_voice/1.wav", "second_voice/0.wav", "second_voice/1.wav"]

    def extract_features(file_path):
        signal, frequency = torchaudio.load(file_path)

        # pipeline expects a NumPy array as input
        signal = signal[0].numpy()
        return signal

    # extract the features from each audio file
    features = [extract_features(file) for file in audio_files]

    # predict the speaker label
    predictions = [classifier(feature)[0] for feature in features]

    for prediction in predictions:
        print("Score: ", prediction["score"], " predicted label: ", prediction["label"])


