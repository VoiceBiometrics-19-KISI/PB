from pathlib import Path
import random
import torch

from speechbrain.pretrained import SpeakerRecognition
from speechbrain.utils.metric_stats import EER

device = 'cuda' if torch.cuda.is_available() else 'cpu'

FINAL_SCORES = []
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb")
# target = 'CN-Celeb_wav/data/'
check = 'C:/Users/Norbix/Desktop/studia/Projekt_badawczy/VoiceBiometrics/WAV2VEC2/testowanie_hindi'
target = 'C:/Users/Norbix/Desktop/studia/Projekt_badawczy/VoiceBiometrics/WAV2VEC2/testowanie_akcent_hindienglish'
check_path = Path(check)
target_path = Path(target)
counter = 103
same_speaker_scores = []
different_speaker_scores = []

for i in range(counter):
    print(i)
    curr_path_check = check_path / f'{i}'
    curr_path_target = target_path / f'{i}'
    index = random.randint(0, 4)
    chosen = ""
    same = []
    diff = []
    mini_count = 0
    #wybierz tekst do sprawdzenia CHECK
    for count, path in enumerate(curr_path_check.rglob("*.wav")):
        mini_count += 1
        if index == count:
            chosen = path

    if mini_count != 5:
        print("AWARIA")
    mini_count = 0
    #wybierz 4 pozostałe teksty z innego języka dla tego samego mówcy
    for count, path in enumerate(curr_path_target.rglob("*.wav")):
        mini_count += 1
        if index != count:
            same.append(path)
    if mini_count != 5:
        print("AWARIA")

    # wybierz 4 pozostałe teksty z innego języka dla innego mówcy
    for j in range(4):
        random_speaker = i
        while random_speaker == i:
            random_speaker = random.randint(0, 102)
        random_clip = random.randint(0, 4)
        random_path = target_path / f'{random_speaker}'
        for count, r_path in enumerate(random_path.rglob("*.wav")):
            if random_clip == count:
                diff.append(r_path)
    same_count = 0
    diff_count = 0
    score_tally = 0

    for x in same:
        score, prediction = verification.verify_files(str(chosen), str(x))  # Different Speakers
        same_speaker_scores.append(score.cpu())
    #     score_tally += score
    #     if prediction:
    #         same_count += 1
    # same_tuple = (same_count, same_speaker_scores)

    score_tally = 0

    for x in diff:
        score, prediction = verification.verify_files(str(chosen), str(x))  # Different Speakers
        different_speaker_scores.append(score.cpu())
        # score_tally += score
        # if not prediction:
        #     diff_count += 1

    # diff_tuple = (diff_count, different_speaker_scores)
    # FINAL_SCORES.append((same_tuple, diff_tuple))

# print(FINAL_SCORES)

positive = torch.Tensor(counter*4)
torch.cat(same_speaker_scores, out=positive)

negative = torch.Tensor(counter*4)
torch.cat(different_speaker_scores, out=negative)

val_eer, threshold = EER(positive, negative)
print("EER: ", val_eer, " THRESHOLD: ", threshold)
