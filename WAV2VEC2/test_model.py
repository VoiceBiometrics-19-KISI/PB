import librosa
import torch
from datasets import load_dataset
from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

VERSION = 1

# Audio Classification Pipeline
if VERSION == 1:
    dataset = load_dataset("anton-l/superb_demo", "si", split="test")
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-sid")
    labels = classifier(dataset[1]["file"], top_k=5)
    print(labels)


# Model Directly
elif VERSION == 2:
    def map_to_array(example):
        speech, _ = librosa.load(example["file"], sr=16000, mono=True)
        example["speech"] = speech
        return example


    # load a demo dataset and read audio files
    dataset = load_dataset("anton-l/superb_demo", "si", split="test")
    dataset = dataset.map(map_to_array)

    model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")

    # compute attention masks and normalize the waveform if needed
    inputs = feature_extractor(dataset[:2]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
    print(labels)
