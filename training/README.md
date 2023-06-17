# How to train ECAPA-TDNN

All necessary files are available in the directory called training. 
There are 4 python files and 2 yaml files with configuration as follows:

- *prepare_cn_celeb.py* - script that creates train, validation and test splits from the data
- *run_prepare_celeb.py* - script that basically runs the prepare_cn_celeb.py
- *my_ecapa_embeddings.py* - main file with training
- *my_ecapa_verification.py* - file for checking the cosine similarity score between 2 test audio files


### 1.  Preparing the data

The data must be put in a separate directory. The audio files need to be inside the directories with the correct speaker id.
The name given the root dataset directory is "CN_Celeb" and can be changed in run_prepare_cn.py, line 6:

    data_folder = os.path.join(parent_path, "CN_Celeb")

In prepare_cn_celeb.py *path_parts* holds the path elements and is used to find the speaker_id directory in line 98:

    spk_id = path_parts[-3][2:]

The *-3* selects third element of the path from the end. The second brackets excludes the letters 'id' from e.g. "id10821". It can be adjusted.

After the execution of the run_prepare_cn.py three json files appear in the same location as the scripts.
Those files are:
- train.json - containing train split - set to 80%
- valid.json - containing validation split - set to 10%
- test.json - containing test split - set to 10%

### 2. Training

Training configuration is set with mytrain.yaml file. 
There needs to be a path set up to correct value in line 21:

    data_folder: C:\...

It can be root project path, because all the other directories (results etc.) contain a reference to it plus relative path.

When this is complete, to train the following command must be entered into the Terminal:

    python my_ecapa_embeddings.py mytrain.yaml --device cpu

It starts the main training scripts and provides it with parameters set up in the config yaml file.
The training is done default on gpu, but if you don't have gpu or cuda is not properly set up you should use the flag *--device cpu*.

Be aware that when you run it for the first time, the rir_noises.zip file will be downloaded for future augmentation done before training.

The checkpoints are saved to the *.../results/speaker_id/seed/save/..." folder, where seed can be set up at the top of the yaml file, so that every training result is stored in different directory.


### 3. Verification

After the training, the verification can be done with the use of ny_ecapa_verification.py script and ecapa_verif.yaml conf file. In the conf file the model and pretrainer that loads data from the saved embedding_model.ckpt checkpoint are defined. The correct model architecture and model weights from the training are defined.
In lines 104 and 105 of the my_ecapa_verification.py script one must adjust paths to two audio files that are to be tested against each other:

    file1 = os.path.join(parent_path, "CN_Celeb", "id10270/5r0dWxy17C8/00001.wav")

To run the file type in the Terminal:

    python my_ecapa_verification.py ecapa_verif.yaml --device cpu

Again, you don't have to use the flag if your cuda is correctly set up.

The cosine similarity is calculated between two inputs and displayed in the Terminal. The higher the results the better when we compare two utterances of the same speaker, and otherwise in case of different speakers.

#### Requirements: 
soundfile
speechbrain
torch
torchaudio
random
json