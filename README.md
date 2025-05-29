# civl4220_building_segmentation
This repo is for submission for CIVL4220 building segmentation project.
## How to use
### Run it on Google Colab (recommended)
Download the Jupyter notebook `demo.ipynb` and the file `kaggle.json`. Upload the notebook to Google Colab. After running the first code block, it should ask you to upload the `kaggle.json` file. Upload it accordingly. Then, you can change the following code block to whatever you want to build before running it.
```{python}
NUM_EPOCHS = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BuildingSegModel(
    arch= "UNET", # change the model arch (decoder) here
    encoder_name= "resnet34", # change the encoder here
    encoder_weights= "imagenet"
)

# Move the model to the device
model.model.to(DEVICE)
```

### Run it locally
1. Step 1: Download the dataset

Please download the Massachusetts Buildings Dataset from this [kaggle page](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset/data?select=png) and then unzip it, put the whole folder under the same directory as the three python file. 

2. Step2: create an environment

Please install Anaconda and create an environment. Then, install all the libraries in `requirement.txt`.

3. Step 3: change the code in `run.py`

In `run.py`, change the following two lists in lines 44-45  to all combinations of encoder and decoder you want to test.
```{python}
ARCHS = ["UNETPLUSPLUS","UNET", "FPN"] # change decoder
ENCODER_NAMES = ["resnet34", "efficientnet-b4", "mobilenet_v2", "mit_b3"] # change encoder
```



