# Purpose of this repository
This repository is the source of finetuning data for a project in PSYC3317.01|CSCI3397.01. The data used to finetune the Cellpose model is contained in the `augmented_data` folder. This augmented data was sourced from the `segemented_for_finetuning` folder.
<br> </br>
The `train_cellpose` is used to finetune the model. Dan also created a `convert_files` file that was used after the data augmentation step. The purpose of that script is to take the output of the data augmentation (which had binary masks) and convert them into instance masks. This was needed to train cellpose properly.
