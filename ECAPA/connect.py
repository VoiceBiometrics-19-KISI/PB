import torchaudio
import soundfile
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs =torchaudio.load('one_voice/0.wav')
embeddings = classifier.encode_batch(signal)


verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files("one_voice/0.wav", "second_voice/0.wav") # Different Speakers
print("Score: ", score, ", prediction: ", prediction)
score, prediction = verification.verify_files("one_voice/0.wav", "one_voice/1.wav") # Same Speaker
print("Score: ", score, ", prediction: ", prediction)
