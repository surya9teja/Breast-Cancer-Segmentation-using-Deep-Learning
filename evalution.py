import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_encoder = 'resen50'
model_name = "Unet_"+model_encoder

df = pd.read_csv("/home/surya/Projects/Capstone/"+model_name+".csv")
plt.figure(figsize=(10, 10))
sns.lineplot(x = "Epoch", y = "Train",data = df)
sns.lineplot(x = "Epoch", y = "Valid", data = df)
plt.ylabel("Train and Valid loss")
plt.xticks(rotation = 25)
plt.show()