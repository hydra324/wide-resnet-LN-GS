**WRN-GN: Wide ResNets With Group Normalization**

======================================\
Training:
1. Training data is available in the `ciar-10-batches-py` directory
2. Please ensure you have the following dependencies loaded in your virtual environment:\
    a. `PyTorch 1.13.0`\
    b. `Matplotlib 3.5.3`\
    c. `tqdm 4.64.1`
3. Change the `save_dir` in Configure.py to where you want to save the model
4. Use the following command to commence training:
    `python3 main.py train cifar-10-batches-py`

======================================\
Testing:
1. Public Test data is also available in the `ciar-10-batches-py` directory
2. Please ensure you have the following dependencies loaded in your virtual environment:\
    a. `PyTorch 1.13.0`\
    b. `Matplotlib 3.5.3`\
    c. `tqdm 4.64.1`
3. Change the `save_dir` in Configure.py to load the model of your choice. If you want to use the best model, please keep it as it is.
4. The best model is `../saved_models/wrn_v2/model-50-ckpt`
5. Use the following command to run test on public test set:
    `python3 main.py test cifar-10-batches-py`

======================================\
Prediction:
1. Private Test data is also available in the `ciar-10-batches-py` directory as `private_test_images_2022.npy`
2. Please ensure you have the following dependencies loaded in your virtual environment:\
    a. `PyTorch 1.13.0`\
    b. `Matplotlib 3.5.3`\
    c. `tqdm 4.64.1`
3. Change the `save_dir` in Configure.py to load the model of your choice. If you want to use the best model, please keep it as it is.
4. The best model is `../saved_models/wrn_v2/model-50-ckpt`
5. Use the following command to run test on public test set:
    `python3 main.py predict cifar-10-batches-py`
6. The results will be stored in `predictions.npy` file

======================================\
Links to all models trained in experiments:
1. ResNet 164-bottleneck: https://drive.google.com/file/d/1fozi8fvXMiqiG0fOwbNLN6IJMdNkG8am/view?usp=share_link
2. PreAct Reset-164 Bottleneck: https://drive.google.com/file/d/1sz0l_PQaCre2qOVLeiDrdLbji3hgdWDw/view?usp=share_link
3. PreAct WRN-28-10 BatchNorm: https://drive.google.com/file/d/1ruWaG2T-rgkMt1SgXlFql2yoUNK_E0uP/view?usp=share_link
4. PreAct WRN-28-10 GroupNorm + WeightStand: https://drive.google.com/file/d/1Dm58unsaNQfM4QiRWCg30tQWLFWH7ed4/view?usp=share_link