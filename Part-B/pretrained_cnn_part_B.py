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
#from albumentations.pytorch.transforms import ToTensorV2
import wandb
# Define a LightningModule class for pretrained CNN
class lightning_pretrained_CNN(L.LightningModule):
    def __init__(self, model_name, unfreeze_layers, optimizer, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.model_name = model_name
        
        # Initialize model based on the model name provided
        if self.model_name == 'ResNet50':
            self.model = models.resnet50(pretrained=True)
        elif self.model_name == 'GoogLeNet':
            self.model = models.googlenet(pretrained=True)
        elif self.model_name == 'InceptionV3':
            self.model = models.inception_v3(pretrained=True, transform_input=True)
        
        # Freeze layers until the unfreeze_layers level
        freeze_index = 0
        for param in self.model.parameters():
            if freeze_index < (len(list(self.model.parameters())) - (unfreeze_layers + 2)):
                param.requires_grad = False
            else:
                break
            freeze_index += 1
        
        # Adjust the final fully connected layer to match the number of output classes (10)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # Training step
        inputs, labels = batch
        output = self.forward(inputs)
        
        if self.model_name == 'InceptionV3':
            logits = output.logits
            _, preds = torch.max(logits, dim=1)
            loss = F.cross_entropy(logits, labels)  # Calculate loss
        else:
            logits = output
            _, preds = torch.max(logits, dim=1)
            loss = F.cross_entropy(logits, labels)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Log training loss

        return loss

    def configure_optimizers(self):  # Configure optimizer based on provided arguments
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'nadam':
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        return optimizer


    def validation_step(self, batch, batch_idx):  # Validation step
        x, y = batch
        predictions = self.forward(x)  # Forward pass

        # Calculate loss
        validation_loss = F.cross_entropy(predictions, y)
        
        # Calculate accuracy
        _, predicted_labels = torch.max(predictions, dim=1)
        validation_accuracy = (predicted_labels == y).float().mean().item()

        # Log the loss and accuracy during validation
        self.log('val_loss', validation_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', validation_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return validation_loss



    def test_step(self, batch, batch_idx):  # Test step
        x, y = batch
        pred = self.forward(x)  # Forward pass
        
        loss = F.cross_entropy(pred, y)  # Compute loss
        _, predicted = torch.max(pred.data, 1)  # Get predicted class labels
        
        # Calculate accuracy
        accuracy = (predicted == y).float().mean().item()

        # Logging the values
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"test_loss": loss}

