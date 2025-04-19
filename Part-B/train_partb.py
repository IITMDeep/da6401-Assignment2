import numpy as np
import pandas as pd 
import pytorch_lightning as L
from torchvision import transforms, models,datasets
#import cv2
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader ,random_split,Subset
import matplotlib.pyplot as plt 
import torchvision.models as models
import torch.nn as nn 
import torch.optim as optim 
from torchmetrics import MetricCollection, Accuracy
import torch.nn.functional as F
import torch
#import os
#import albumentations as A
#from albumentations.pytorch.transforms import ToTensorV2
import wandb
import torch
import argparse
# Importing custom modules
from Data_manager_partB import root_dataset,inaturalist_train,inaturalist_val,inaturalist_test
from pretrained_cnn_part_B import lightning_pretrained_CNN 
wandb.login()  # Login to Weights & Biases
def main(args):
    if True:
        # Constructing the experiment's run name based on provided arguments
        experiment_name = f'bs-{args.batch_size}-lr-{args.learning_rate}-ep-{args.epochs}-op-{args.optimizer}-mn-{args.model_name}-ul-{args.unfreeze_layers}'

        # Loading the dataset using a custom dataset handler
        dataset_handler = root_dataset(args.path)
        train_dataset = dataset_handler.get_train_data()
        val_dataset = dataset_handler.get_val_data()

        # Preparing the datasets for training, validation, and testing
        train_set = inaturalist_train(train_dataset, args.model_name)
        val_set = inaturalist_val(val_dataset, args.model_name)
        test_set = inaturalist_test(args.model_name, args.path)

        # Extracting parameters from arguments
        batch_size = args.batch_size
        unfreeze_layers = args.unfreeze_layers
        chosen_optimizer = args.optimizer
        num_epochs = args.epochs
        lr = args.learning_rate

        # Initializing Weights & Biases logging
        wandb.init(project=args.project_name, entity=args.wandb_entity, name=experiment_name)

        # Setting up the WandbLogger for PyTorch Lightning
        wandb_logger = WandbLogger(project='DL-Assignment-2', entity='cs24m019-iitm')

        # Initializing the DataLoaders for training and validation
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        validation_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=2)

        # Initializing the model with the specified parameters
        cnn_model = lightning_pretrained_CNN(args.model_name, unfreeze_layers, chosen_optimizer, lr)

        # Configuring the PyTorch Lightning Trainer
        trainer = L.Trainer(accelerator='auto', devices='auto', max_epochs=num_epochs, logger=wandb_logger)

        # Training the model
        trainer.fit(cnn_model, train_loader, validation_loader)

        # Setting up the test DataLoader
        test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=False, num_workers=1)

        # Testing the model after training
        trainer.test(dataloaders=test_loader)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  #taking commands from command line arguments
    parser.add_argument('-wp','--project_name',type=str,default='Assignment2-CS6910',help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we','--wandb_entity',type=str,default='amar_cs23m011',help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-p', '--path', type=str, help='provide the path where your data is stored in memory,Read the readme for more description')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of epochs to CNN')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size used to train CNN')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'nadam'], help='optimzer algorithm to evaluate the model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--model_name', type=str, default='ResNet50', choices=['ResNet50', 'GoogLeNet', 'InceptionV3'], help='pretrained_model_name')
    parser.add_argument('-ul', '--unfreeze_layers', type=int, default=15, help='number of unfreeze layer to train the model')
    args = parser.parse_args()
    main(args)
