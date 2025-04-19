# da6401-Assignment2
Convolution Neural Network

# PART-A 

## CNN Training Script

This script enables the training of a Convolutional Neural Network (CNN) using PyTorch, providing flexibility in configuring the network architecture, training settings, and hyperparameters through command-line arguments. 


## CNN Architecture

The architecture of the Convolutional Neural Network  is outlined within the script named PartA.ipynb. This architecture is composed of several convolutional layers, each followed by max-pooling layers to reduce dimensionality. Finally, there is a densely connected layer. The structure of this CNN can be tailored to 
specific requirements by adjusting various parameters, offering flexibility in its design to accommodate different datasets and objectives.

## Usage

1. To begin, ensure you have your dataset prepared and accessible. Update the data_path argument in the script to point to the directory containing your dataset.
2. Once your dataset is ready and the data_path argument is set, execute the training script with the desired configurations. 

## command line arguments that can be passed to set parameters
-   Kernel Size : '-ks', '--kernel_size', [3,5] etc
-   Activation: '-a', '--activation', ['ReLU', 'GELU' , 'LeakyReLU' , "SiLU" , "Mish" "elu"]
-   Epochs: '-e', '--epochs', [25,20] etc
-   Dense layer size: '-neu','--dense_nodes', [1024,512] etc
-   kernel Organisation : '-ko', '--kernel_organisation' ,  ['same','double','half','default']
-   Num Of Kernels : '-nk' , '--num_kernel' , [16,32] etc
**Default set to my best configuration**

## Prediction Function

The function model_predict is designed to accept an image as input and utilize the trained Convolutional Neural Network (CNN) model to make a prediction regarding the image's class label. 

## Training Process
In the training process we opitimized the CNN prameters and minimized the cross entropy loss across predicted and actual labels. It invlolved of interating on the 
dataset for many epochs,then updating model parameters with the help of backpropagation, and evaluating the model's performance.

## Plotting Grid

The function plot_grid is responsible for generating a visual representation of the model's predictions on a subset of the validation dataset. It creates a grid layout with 10 rows and 3 columns, where each cell contains an image along with its corresponding true label and the label predicted by the model. 

# PART B

## Function for  Data Loading (`data_load`):

This segment of code is responsible for loading the dataset from the designated directory and applying various transformations to the data to prepare it for training. These transformations typically include resizing the images to a uniform size, performing normalization to ensure consistent data scales.

## Function for Training (`Train`):

This function serves the purpose of training the model over a specified number of epochs. It involves iterating through the dataset, during which the model conducts a forward pass to generate predictions for each batch of data. Subsequently, it computes the loss between the predicted labels and the ground truth labels. 

My code is very much flexible to add in command line arguments . I am adding the list of possible argument below for your reference.Please try to run this on local PC or from command promt by ensuring all the libraries in requirements.txt are already installed in your system. Because in google colab this might give problem .
python train_partb.py -p path to the folder where data reside. For example C:\Users\USER\Downloads\nature_12K\inaturalist_12K\train this where my train data reside then you just given till 
C:\Users\USER\Downloads\nature_12K\inaturalist_12K position , program will add train and val portion itself.

| Name        | Default Value   | Description |
| ------------- |:-------------:| -----:|
| `-wp ,--project_name`     | DA6401-Assignment2 | it will make login into wandb in the project_name project |
| `-e,--epochs` | 15      |    number of epochs your algorithm iterate |
|`-b,--batch_size`|16      |batch size your model used to train |
|`-o,--optimizer`|adam|Choices=['sgd', 'adam', 'nadam']|
|`-lr,--learning_rate`|0.001|Learning rate used to optimize model parameters|
|`-m,--model_name`|ResNet50|Choices=['ResNet50', 'GoogLeNet', 'InceptionV3']|
|`-we,--wandb_entity`|amar_cs23m011|Project name used to track experiments in Weights & Biases dashboard|
| `-ul,--unfreeze_layers` | 15      |    number of layer to unfreeze for model training |
|`-p,--path`|mandatory field      |location where your data stored. |
Few example are shown below to how to give inputs:-
```
```
This will run my best model which i get by validation accuracy. after that it will create a log in a project named DA6401-assignment2 by default until user dont specify project name.
```
parameters: {
      'epochs': {
            'values': [10]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'batch_size': {
            'values': [16]
        },
        'optimizer':{
              'values': ['adam']
        },
        'model':{
            'values':['ResNet50']
        },
        'unfreeze_layers':{
            'values':[15]
        }
      
    }```
Now if you want to change the number of layer I just have to execute the following the command.
```
python train_partb.py -e 6
```
this will change the number of epoch to 6. Similarly we can use other commands as well. 
