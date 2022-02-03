# Breast Cancer segmentation using deep learning

The current research aims to design a deep learning model for nuclei segmentation using earlier research studies. This work employs a variety of methodologies, including pre-processing techniques on datasets, a multi-organ transfer learning method for segmentation. The various encoder-based models will be trained using a combination of the TNBC and MoNuSeg datasets but will be evaluated using the TNBC test dataset.

**Project required packages**

- `albumentations==1.1.0`
    
- `imageio==2.12.0`
    
- `matplotlib==3.5.0`
    
- `numpy==1.21.4`
    
- `opencv_python_headless==4.5.4.60`
    
- `pandas==1.3.4`
    
- `scikit_learn==1.0.2`
    
- `scipy==1.7.3`
    
- `seaborn==0.11.2`
    
- `segmentation_models_pytorch==0.2.1`
    
- `torch==1.10.0+cu113`
    
- `tqdm==4.62.3`

To train the model
```
git clone https://github.com/surya9teja/Breast-Cancer-Segmentation-using-Deep-Learning.git
cd Breast-Cancer-Segmentation-using-Deep-Learning
pip install -r requirements.txt
python3 data_pre_process.py
python3 main.py
# for Graphs and evaluation
python3 test.py 
python3 graphs.py 
```
Note: Before running the files it requires datasets from the drive and place them under 'Datasets/'
To install required packages

`pip install -r requirements.txt`

Dataset link available at [google drive](https://drive.google.com/drive/folders/1jpMpMCZmGvZrAyTzix2JDX8GLAa6ZeaO?usp=sharing)

# Results
![Training Loss per epoch](https://github.com/surya9teja/Breast-Cancer-Segmentation-using-Deep-Learning/blob/master/assets/Train_loss.png)
![Validation Loss per epoch](https://github.com/surya9teja/Breast-Cancer-Segmentation-using-Deep-Learning/blob/master/assets/Evaluation_loss.png)
![Evalaution Metrics](https://github.com/surya9teja/Breast-Cancer-Segmentation-using-Deep-Learning/blob/master/assets/metrics.png)
