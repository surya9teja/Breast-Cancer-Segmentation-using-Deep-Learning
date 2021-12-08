import os
import random
import numpy as np
import torch
from glob import glob
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from  segmentation_models_pytorch import Unet
import numpy as np
import pandas as pd


from classes import Model_Training as MT
from classes import DataDrive as DD
from classes import loss_functions as LF


""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



if __name__ == "__main__":
    """ seeding"""
    seeding(42)

    """Directories"""
    create_dir("files")

    """Loading Data"""

    train_x = sorted(glob("/home/surya/Projects/Capstone/Datasets/new_dataset/train/image/*"))
    train_y = sorted(glob("/home/surya/Projects/Capstone/Datasets/new_dataset/train/mask/*"))

    valid_x = sorted(glob("/home/surya/Projects/Capstone/Datasets/new_dataset/test/image/*"))
    valid_y = sorted(glob("/home/surya/Projects/Capstone/Datasets/new_dataset/test/mask/*"))

    data_ = f"Dataset size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_)

    """Hyper parameters"""
    size = (512, 512)
    batch_size = 3
    epochs = 20
    learning_rate = 1e-4
    pre_trained = False

    
    """ Dataset and loader"""

    train_data = DD.DataDrive(train_x, train_y)
    valid_data = DD.DataDrive(valid_x, valid_y)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=2)

    encoders = ["resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d", "DenseNet121", "DenseNet201"]
    for encoder_name in encoders:
        device = torch.device('cuda')
        if pre_trained:
            model = Unet(encoder_name, encoder_weights="imagenet", classes=1, activation=None)
            model_name = "Unet_" +"pre_trained"+ encoder_name
        else:
            model = Unet(encoder_name, encoder_weights=None, classes=1, activation=None)
            model_name = "Unet_"+ encoder_name
        checkpoint_path = "files/" + model_name +".pth"
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        loss_function = LF.DiceBCELoss()

        """ Training the model """
        best_valid_loss = float("inf")

        losses_values  = np.array(["epoch", "Train", "Test"])

        M = MT.Model_Training(model, train_loader, valid_loader, optimizer, device, loss_function)

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = M.train()
            valid_loss = M.evaluate() 

            """ Saving the model """
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            losses_values = np.vstack((losses_values, np.array([epoch, train_loss, valid_loss])))
            print(data_str)
        C = pd.Index(["Epoch", "Train", "Valid"], name="columns")
        df = pd.DataFrame(data=losses_values, columns=C)
        df.drop(index=df.index[0], axis=0, inplace=True)
        df.to_csv(model_name+".csv")