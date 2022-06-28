import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np

encoders = ["None","resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d", "densenet121", "densenet201",
                "pre_trained_resnet50", "pre_trained_resnet101", "pre_trained_resnext50_32x4d", "pre_trained_resnext101_32x8d", 
                "pre_trained_densenet121", "pre_trained_densenet201"]


""" Plotting training and validation loss for each encoder """
sns.set_style("whitegrid", {'grid.linestyle': '--'})
f, ax = plt.subplots(figsize=(18, 10))
best_loss = float("inf")
for encoder_name in encoders:
    model_name = "Unet_" + encoder_name
    df = pd.read_csv("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/files/train_data/"+model_name+".csv")
    x = df["Epoch"]
    y = df["Train"]
    xnew = np.linspace(x.min(), x.max(), 200)

    spl = make_interp_spline(x, y, k = 3)
    ynew = spl(xnew)

    if best_loss > df["Train"].min():
        best_loss = df["Train"].min()
        best_model = model_name
    fig = sns.lineplot(x = xnew, y = ynew, label = encoder_name)
    #print(df["Train"].min(), best_model)
fig.set(xlabel="Epoch", ylabel="Loss", title="Train Loss per Epoch", xlim=(0, 19), xticks=list(range(0, 20)))
ax.text(2, 1.2,"Best Encoder: "+best_model+"\nwith train loss: "+str(best_loss), fontsize=10)
plt.savefig("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/assets/Train_loss.png", dpi=300)

f, ax = plt.subplots(figsize=(18, 10))
best_loss = float("inf")
for encoder_name in encoders:
    model_name = "Unet_" + encoder_name
    df = pd.read_csv("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/files/train_data/"+model_name+".csv")
    x = df["Epoch"]
    y = df["Valid"]
    xnew = np.linspace(x.min(), x.max(), 200)

    spl = make_interp_spline(x, y, k = 3)
    ynew = spl(xnew)

    if best_loss > df["Valid"].min():
        best_loss = df["Valid"].min()
        best_model = model_name
    fig = sns.lineplot(x = xnew, y = ynew, label = encoder_name)
    #print(df["Valid"].min(), best_model)
    fig = sns.lineplot(x = xnew, y = ynew, label = encoder_name)
fig.set(xlabel="Epoch", ylabel="Loss", title="Evaluation Loss per Epoch", xlim=(0, 19), xticks=list(range(0, 20)))
ax.text(7, 1.7,"Best Encoder: "+best_model+"\nwith valid loss: "+str(best_loss), fontsize=10)
plt.savefig("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/assets/Evaluation_loss.png", dpi=300)

""" Evaluation metrics for each encoder """

"""Reading csv file into dataframe"""
fig, ax = plt.subplots(figsize=(18, 10))
results = pd.read_csv("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/results/metrics.csv")
ax = sns.barplot(x="Encoder", y="Value", hue= "score category", data=results)
ax.set(xlabel="Encoder", ylabel="Value", title="Evaluation Metrics")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/home/surya/projects/Breast-Cancer-Segmentation-using-Deep-Learning/assets/metrics.png", dpi=300)
plt.show()