#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
import os
import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(run_opts["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = (
                wavs.to(run_opts["device"]),
                lens.to(run_opts["device"]),
            )
            emb = compute_embedding(wavs, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict

def compute_embedding_from_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.to(run_opts["device"])
    wav_lens = torch.tensor([waveform.size(1)]).to(run_opts["device"])
    embedding = compute_embedding(waveform, wav_lens)
    return embedding

if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"]+"/VERIF",
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Set up pretrainer from yaml
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(run_opts["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(run_opts["device"])

    # Get filed for testing and get their embeddings
    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)
    file1 = os.path.join(parent_path, "CN_Celeb", "id10270\\5r0dWxy17C8\\00001.wav")
    file2 = os.path.join(parent_path, "CN_Celeb", "id10295\\iVxjBZtQwBg\\00001.wav")

    embedding1 = compute_embedding_from_file(file1)
    embedding2 = compute_embedding_from_file(file2)

    # Cosine simiarity calculations
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    cosine_sim = similarity(embedding1, embedding2)
    print("Cosine similarity:", cosine_sim.item())