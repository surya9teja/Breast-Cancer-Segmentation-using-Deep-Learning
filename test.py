import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import random
import imageio
import torch
from  segmentation_models_pytorch import Unet
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

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

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    test_x = sorted(glob("/home/surya/Projects/Capstone/Datasets/new_dataset/test/image/*"))
    test_y = sorted(glob("/home/surya/Projects/Capstone/Datasets/new_dataset/test/mask/*"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)

    """Check metrics.csv exists or not if exists then clear it"""

    if os.path.exists("results/metrics.csv"):
        os.remove("results/metrics.csv")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoders = ["resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d", "densenet121", "densenet201",
                "pre_trained_resnet50", "pre_trained_resnet101", "pre_trained_resnext50_32x4d", "pre_trained_resnext101_32x8d", 
                "pre_trained_densenet121", "pre_trained_densenet201"]

    for encoder_name in encoders:
        """ Load the checkpoint """
        checkpoint_path = "files/Unet_"+encoder_name+".pth"
        model = Unet(encoder_name = encoder_name.replace("pre_trained_", ""), classes=1)
        model = model.to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        model_name = "Unet_" + encoder_name

        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        time_taken = []

        for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
            """ Extract the name """
            name = x.split("/")[-1].split(".")[0]

            """ Reading image """
            image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
            ## image = cv2.resize(image, size)
            x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
            x = x/255.0
            x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
            x = x.astype(np.float32)
            x = torch.from_numpy(x)
            x = x.to(device)

            """ Reading mask """
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
            ## mask = cv2.resize(mask, size)
            y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
            y = y/255.0
            y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
            y = y.astype(np.float32)
            y = torch.from_numpy(y)
            y = y.to(device)

            with torch.no_grad():
                """ Prediction and Calculating FPS """
                start_time = time.time()
                pred_y = model(x)
                pred_y = torch.sigmoid(pred_y)
                total_time = time.time() - start_time
                time_taken.append(total_time)


                score = calculate_metrics(y, pred_y)
                metrics_score = list(map(add, metrics_score, score))
                pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
                pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
                pred_y = pred_y > 0.5
                pred_y = np.array(pred_y, dtype=np.uint8)

            """ Saving masks """
            ori_mask = mask_parse(mask)
            pred_y = mask_parse(pred_y)* 255
            image = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            ori_mask = cv2.copyMakeBorder(ori_mask, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            pred_y = cv2.copyMakeBorder(pred_y, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.putText(image,model_name, (1,20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA, False)
            cv2.putText(image,"Orginal Image", (160,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA, False)
            cv2.putText(ori_mask,model_name, (1,20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA, False)
            cv2.putText(ori_mask,"Orginal Mask", (160,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA, False)
            cv2.putText(pred_y,model_name, (1,20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA, False)
            cv2.putText(pred_y,"Predicted Mask", (160,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA, False)

            line = np.ones((image.shape[0], 10, 3)) * 128

            cat_images = np.concatenate(
                [image, line, ori_mask, line, pred_y], axis=1
            )
            create_dir("results/" + model_name)
            cv2.imwrite(f"results/{model_name}/{name}.png", cat_images)

        jaccard = metrics_score[0]/len(test_x)
        f1 = metrics_score[1]/len(test_x)
        recall = metrics_score[2]/len(test_x)
        precision = metrics_score[3]/len(test_x)
        acc = metrics_score[4]/len(test_x)
        fps = 1/np.mean(time_taken)
        """ Saving metrics into a csv file """
        with open("results/metrics.csv", "a") as f:
            encoder = model_name.replace("Unet_", "")
            f.write(f"{encoder},{jaccard},Jaccard\n{encoder},{f1},F1\n{encoder},{recall},Recall\n{encoder},{precision}, Precsiion\n{encoder},{acc},Accuracy\n")
        df = pd.read_csv("results/metrics.csv", header=None)
        print(f"Encoder: {model_name} - Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - FPS: {fps:1.4f}")
    df.to_csv("results/metrics.csv", header=["Encoder", "Value", "score category"], index=False)
    #df.to_csv("results/metrics.csv", header=["Encoder", "Jaccard", "F1", "Recall", "Precission", "Accuracy", "FPS"], index=False)