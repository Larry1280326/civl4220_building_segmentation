import os
import torch
import pytorch_lightning as pl
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from data import SegmentationDataset
from model import BuildingSegModel, save_evaluation, plot_loss_curve, visualize

BATCH_SIZE = 8
SHUFFLE = False 
NUM_EPOCHS = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

train_dataset = SegmentationDataset("dataset/png/train", "dataset/png/train_labels", transform=train_transform)
val_dataset = SegmentationDataset("dataset/png/val", "dataset/png/val_labels", transform=val_transform)
test_dataset = SegmentationDataset("dataset/png/test", "dataset/png/test_labels", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)


ARCHS = ["UNETPLUSPLUS","UNET", "FPN"]
ENCODER_NAMES = ["resnet34", "efficientnet-b4", "mobilenet_v2", "mit_b3"]
for arch in ARCHS:
    for encoder_name in ENCODER_NAMES:
        try:
            model = BuildingSegModel(
                arch= arch,
                encoder_name= encoder_name,
                encoder_weights= "imagenet"
            )

            # Move the model to the device
            model.model.to(DEVICE)

            # Train loop
            model.train_loop(train_loader, NUM_EPOCHS)

            # Get the loss curve
            plot_loss_curve(model) 

            # Save the model using pickle
            arch = model.arch
            encoder_name = model.encoder_name
            encoder_weight = model.encoder_weights
            save_dir = f"./saved_models"
            model_filename = save_dir + f"/{arch}_{encoder_name}_{encoder_weight}_{NUM_EPOCHS}.pkl"
            os.makedirs(save_dir, exist_ok=True)
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)
            print(f"Model saved to {model_filename}")

            # Evaluation
            save_evaluation(model, train_loader, "eval/eval_train.csv")
            save_evaluation(model, val_loader, "eval/eval_val.csv")
            save_evaluation(model, test_loader, "eval/eval_test.csv")

            # Visualization on the test loader
            visualize(model, test_loader)
        except:
            print(f"{arch} doesn't support {encoder_name}.")