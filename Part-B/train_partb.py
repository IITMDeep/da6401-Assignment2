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
def main(config=None):
    with wandb.init(config=config, ):
        config=wandb.config
        wandb.run.name='bs-'+str(config.batch_size)+'-lr-'+ str(config.learning_rate)+'-ep-'+str(config.epochs)+ '-op-'+str(config.optimizer)+'-mn-'+str(config.model)+'-ul-'+str(config.unfreeze_layers)
        model_name=config.model
        root_obj=root_dataset()
        train_data=root_obj.get_train_data()
        val_data=root_obj.get_val_data()
        dataset1=inaturalist_train(train_data,model_name)
        dataset2=inaturalist_val(val_data,model_name)
        dataset3=inaturalist_test(model_name)
        b_size=config.batch_size
        unfreeze_layers=config.unfreeze_layers
        optimizer=config.optimizer
        epoch=config.epochs
        learning_rate=config.learning_rate
        wandb_logger = WandbLogger(project='DL-Assignment-2', entity='cs24m019-iitm')
        dataloader=DataLoader(dataset=dataset1,batch_size=b_size,shuffle=True,num_workers=2)
        val_dataloader=DataLoader(dataset=dataset2,batch_size=b_size,shuffle=False,num_workers=2)
        model=lightning_pretrained_CNN(model_name,unfreeze_layers,'adam',0.0015)
        trainer = L.Trainer(accelerator='auto',devices="auto",max_epochs=epoch,logger=wandb_logger)
        trainer.fit(model,dataloader,val_dataloader)
        test_dataloader=DataLoader(dataset=dataset3,batch_size=8,shuffle=False,num_workers=1)
        trainer.test(dataloaders=test_dataloader)

        

        
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
