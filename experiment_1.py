from pathlib import Path
import random
import torch

from speechbrain.pretrained import SpeakerRecognition
from speechbrain.utils.metric_stats import EER

FINAL_SCORES = []
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                               run_opts={"device": "cuda:0"})
target = 'CN-Celeb_wav/data/'
target_path = Path(target)
counter = 50

for i in range(counter):
    same_speaker_scores = []
    different_speaker_scores = []

    print(i)
    curr_path = target_path / f'id{i:05}'
    index = random.randint(0, 4)
    chosen = ""
    same = []
    diff = []
    mini_count = 0

    for count, path in enumerate(curr_path.rglob("*.wav")):
        mini_count += 1
        if index == count:
            chosen = path
        else:
            same.append(path)
    if mini_count != 5:
        print("AWARIA")

    for j in range(4):
        random_speaker = i
        while random_speaker == i:
            random_speaker = random.randint(0, 200)
        random_clip = random.randint(0, 4)
        random_path = target_path / f'id{random_speaker:05}'
        for count, r_path in enumerate(random_path.rglob("*.wav")):
            if random_clip == count:
                diff.append(r_path)
    same_count = 0
    diff_count = 0
    score_tally = 0

    for x in same:
        score, prediction = verification.verify_files(str(chosen), str(x))  # Different Speakers
        same_speaker_scores.append(score.cpu())
        score_tally += score
        if prediction:
            same_count += 1
    same_tuple = (same_count, same_speaker_scores)

    score_tally = 0

    for x in diff:
        score, prediction = verification.verify_files(str(chosen), str(x))  # Different Speakers
        different_speaker_scores.append(score.cpu())
        score_tally += score
        if not prediction:
            diff_count += 1

    diff_tuple = (diff_count, different_speaker_scores)
    FINAL_SCORES.append((same_tuple, diff_tuple))
print(FINAL_SCORES)



# positive_scores = torch.tensor(same_speaker_scores)
# negative_scores = torch.tensor(different_speaker_scores)
#
# print(positive_scores[0], positive_scores[1], positive_scores[2])
# print(negative_scores[0], negative_scores)
# val_eer, threshold = EER(positive_scores, negative_scores)
# # Print results
# print("EER: ", val_eer, " THRESHOLD: ", threshold)