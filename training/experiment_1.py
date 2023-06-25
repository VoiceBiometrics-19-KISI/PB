from pathlib import Path
import random

from speechbrain.pretrained import SpeakerRecognition

FINAL_SCORES = []

verification = SpeakerRecognition.from_hparams(source="C:\\Users\\Paulina\\Desktop\\PB\\training\\results\\speaker_id\\1988",
                                               savedir="C:\\Users\\Paulina\\Desktop\\PB\\training\\results\\speaker_id\\1988",
                                               run_opts={"device": "cuda:0"})
# verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
#                                                savedir="pretrained_models/spkrec-ecapa-voxceleb",
#                                                run_opts={"device": "cuda:0"})
target = '../dataset_test_en/'
target_path = Path(target)
for i in range(10):
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
        print("AAAAAAAA")
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
        score_tally += score
        if prediction:
            same_count += 1
    same_tuple = (same_count, (score_tally / 4).cpu().numpy()[0])
    score_tally = 0
    for x in diff:
        score, prediction = verification.verify_files(str(chosen), str(x))  # Different Speakers
        score_tally += score
        if not prediction:
            diff_count += 1
    diff_tuple = (diff_count, (score_tally / 4).cpu().numpy()[0])
    FINAL_SCORES.append((same_tuple, diff_tuple))
print(FINAL_SCORES)
