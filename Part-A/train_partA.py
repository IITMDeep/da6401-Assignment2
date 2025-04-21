import argparse
import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import torchvision
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
import wandb
from torchvision.transforms import RandomCrop, RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, Compose

arguments = argparse.ArgumentParser()
arguments.add_argument('-nk' , '--num_kernel',type=int,default=32)
arguments.add_argument('-ko', '--kernel_organisation' , type=str, help="['same','double','half','default']",default="default")
arguments.add_argument('-e', '--epochs',  type=int, default=25)
arguments.add_argument('-ks', '--kernel_size', type=int, default=3)
arguments.add_argument('-neu','--dense_nodes', type=int,help="number of neurons in dense layer", default=1024)
arguments.add_argument('-a', '--activation',type=str,help="['ReLU', 'GELU' , 'LeakyReLU' , 'SiLU'' , 'Mish','elu' ]", default = "ReLU")
args_cmd = arguments.parse_args()
print(args_cmd.dense_nodes)
# configuration cell
epochs = args_cmd.epochs

activationFunctions = dict()
activationFunctions["conv1"] = args_cmd.activation
activationFunctions["conv2"] = args_cmd.activation
activationFunctions["conv3"] = args_cmd.activation
activationFunctions["conv4"] = args_cmd.activation
activationFunctions["conv5"] = args_cmd.activation
activationFunctions["fc1"] = args_cmd.activation
list_kernelSize= [args_cmd.kernel_size]*5
listDropout = [0,0,0.3]

#to set number of kernels in each convo layer
fo = args_cmd.kernel_organisation
kernelNumber = []
first_ker = args_cmd.num_kernel

if(fo=="same"):
    for i in range(5):
        kernelNumber.append(first_ker)
elif(fo=="double"):
    for i in range(5):
        kernelNumber.append(first_ker)
        first_ker=2*first_ker
elif(fo=="half"):
    for i in range(5):
        if(first_ker<1):
            first_ker=1
        kernelNumber.append(first_ker)
        first_ker=first_ker//2
elif(fo=="default"):
    kernelNumber.append(32)
    kernelNumber.append(32)
    kernelNumber.append(64)
    kernelNumber.append(64)
    kernelNumber.append(128)


# kernelNumber = [32,32]+[64,64]+[128]
classes = 10
learningRate = 1e-4
nodesfc1 = args_cmd.dense_nodes
lr_schedule = 1 # per 10 epochs half the learningRate
modelName = 'Best_CNN_5Layers_iNaturalist'


# check if CUDA is available
cuda = torch.cuda.is_available()
if cuda == True:
    device = torch.device("cuda")
if cuda != True:
    device = torch.device("cpu")
    
print(device)

################################## util.py##############################################################


## dataloader
def loader(t1data, valdata, t2data, batch):
    bs = batch
    should_shuffle_train = True
    should_shuffle_test = False

    allLoaders = dict()
    for phase, dataset, shuffle_flag in zip(
        ['train', 'valid', 'test'],
        [t1data, valdata, t2data],
        [should_shuffle_train, should_shuffle_train, should_shuffle_test]
    ):
        allLoaders[phase] = torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,
            num_workers=4,
            shuffle=shuffle_flag
        )

    return allLoaders



## transforms to match realModel input dims
def transform():
    string = 'Normalize'
    sizeChange = 224
    valResize = 256
    valCenterCrop = sizeChange

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    t1_t = Compose([
        RandomResizedCrop(sizeChange),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean_vals, std=std_vals)
    ])

    val_t = Compose([
        Resize(valResize),
        CenterCrop(valCenterCrop),
        ToTensor(),
        Normalize(mean=mean_vals, std=std_vals)
    ])

    resize_tuple = (sizeChange, sizeChange)
    t2_t = Compose([
        Resize(resize_tuple),
        ToTensor(),
        Normalize(mean=mean_vals, std=std_vals)
    ])

    transforms = dict()
    for k, v in zip(['training', 'validation', 'test'], [t1_t, val_t, t2_t]):
        transforms[k] = v

    return transforms


def activationFun(activation):
    act=activation
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'elu':
        return nn.ELU()
    elif activation.lower() == 'silu':
        return nn.SiLU()
    elif activation.lower() == 'mish':
        return nn.Mish()
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    
## Load dataset fn
def data_load():
    transforms=transform()
    #/kaggle/input/dl-assignment-2-data/inaturalist_12K/train
    t1set  = torchvision.datasets.ImageFolder('/kaggle/input/inature-dataset/inaturalist_12K/train', transforms['training'])
    train, val = random_split(t1set, [8000, 1999])
    #/kaggle/input/dl-assignment-2-data/inaturalist_12K/val
    t2set   = torchvision.datasets.ImageFolder('/kaggle/input/inature-dataset/inaturalist_12K/val', transforms['test'])
    lables = t1set.classes
    return train, val, t2set, lables

class blockConv(nn.Module):
    def __init__(self, channelsIn, channelsOut, kernel=3, BN=True, NL="ReLU", stride=1, padding=0):
        super(blockConv, self).__init__()
        KL = channelsOut
        self.BN = BN
        self.NL = NL

        bias_flag = False
        self.conv = nn.Conv2d(
            in_channels=channelsIn,
            out_channels=KL,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias_flag
        )

        if self.BN:
            epsilon = 1e-3
            self.bn = nn.BatchNorm2d(KL, eps=epsilon)

        self.act = activationFun(self.NL)

    def forward(self, x):
        x = self.conv(x)
        if self.BN:
            x = self.bn(x)
        out = self.act(x)
        return out



class fc_block(nn.Module):
    runs = 1

    def __init__(self, channelsIn, channelsOut, BN=False, NL="relu"):
        super(fc_block, self).__init__()
        x = channelsOut
        self.fc = nn.Linear(channelsIn, x)
        self.BN = BN
        self.NL = NL

        if self.BN:
            eps_val = 1e-3
            self.bn = nn.BatchNorm2d(x, eps=eps_val)

        self.act = activationFun(self.NL)

    def forward(self, x):
        x = self.fc(x)
        if self.BN:
            x = self.bn(x)
        return self.act(x)


def get_fc_in(dim, list_kernelSize, kernelNumber):
    fc_in = (dim - list_kernelSize[0] + 1)
    fc_in = ((fc_in - 2) // 2) + 1  # max pool after conv1

    index = 1
    while index < 5:
        temp = fc_in - list_kernelSize[index] + 1
        fc_in = ((temp - 2) // 2) + 1  # max pool after conv
        index += 1

    area = fc_in * fc_in
    ans = area * kernelNumber[4]
    return ans

def config_str_list_int(s):
    l=1
    return list(map(int, s[3:].split('-')))

def config_str_list_float(s):
    l=1
    return list(map(float, s.split('-')))


# test_data_path = 'give/path'
# class_images = [[] for _ in range(len(os.listdir(test_data_path)))]
# def gen_random_images():
#     for i, folder in enumerate(os.listdir(test_data_path)):
#         folder_path = os.path.join(test_data_path, folder)
#         for image_name in random.sample(os.listdir(folder_path),3):
#             image_path = os.path.join(folder_path, image_name)
#             img = Image.open(image_path).resize((256, 256))
#             class_images[i].append((img, folder))

# def get_prediction(image):
#     trained_model.eval()
#     transform = Compose([Resize((224, 224)), ToTensor()])
#     input_tensor = transform(image).to(device)
#     input_batch = input_tensor.unsqueeze(0)
#     with torch.no_grad():
#         output = trained_model(input_batch)
#     _, predicted_class = torch.max(output, 1)
#     predicted_class_idx = predicted_class.item()
#     return predicted_class_idx

class CNN_5layer(nn.Module):
    def __init__(self, list_kernelSize, kernelNumber, activationFunctions, listDropout, nodesfc1, classes):
        super(CNN_5layer, self).__init__()
        self.dim = 224
        self.listDropout = listDropout
        bol_batchnorm_enabled = False

        self.conv1 = nn.Sequential(
            blockConv(3, kernelNumber[0], kernel=list_kernelSize[0], BN=bol_batchnorm_enabled, NL=activationFunctions['conv1']),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            blockConv(kernelNumber[0], kernelNumber[1], kernel=list_kernelSize[1], BN=True, NL=activationFunctions['conv2']),
            nn.MaxPool2d(kernel_size=2)
        )

        if self.listDropout[0] != 0:
            self.dropout1 = nn.Dropout(p=self.listDropout[0])

        self.conv3 = nn.Sequential(
            blockConv(kernelNumber[1], kernelNumber[2], kernel=list_kernelSize[2], BN=True, NL=activationFunctions['conv3']),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            blockConv(kernelNumber[2], kernelNumber[3], kernel=list_kernelSize[3], BN=True, NL=activationFunctions['conv4']),
            nn.MaxPool2d(kernel_size=2)
        )

        if self.listDropout[1] != 0:
            self.dropout2 = nn.Dropout(p=self.listDropout[1])

        self.conv5 = nn.Sequential(
            blockConv(kernelNumber[3], kernelNumber[4], kernel=list_kernelSize[4], BN=True, NL=activationFunctions['conv5']),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1_in_features = get_fc_in(self.dim, list_kernelSize, kernelNumber)
        self.fc1 = fc_block(self.fc1_in_features, nodesfc1, NL=activationFunctions['fc1'])

        if self.listDropout[2] != 0:
            self.dropout3 = nn.Dropout(p=self.listDropout[2])

        self.fc2 = nn.Linear(nodesfc1, classes)

    def forward(self, x):
        if x.shape[2] != self.dim:
            print("input dim not matched")
            return

        x = self.conv1(x)
        x = self.conv2(x)

        if self.listDropout[0] != 0:
            x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)

        if self.listDropout[1] != 0:
            x = self.dropout2(x)

        x = self.conv5(x)
        batch_dim = x.shape[0]
        x = x.view(batch_dim, -1)

        x = self.fc1(x)

        if self.listDropout[2] != 0:
            x = self.dropout3(x)

        x = self.fc2(x)

        return x

def train(totalEpoch, allLoaders, realModel, opt, criterion, scheduler, cuda):
    for epoch in range(totalEpoch):
        # Initialize loss variables for training and validation
        training_loss, validation_loss = 0.0, 0.0
        
        ######################
        # Training phase
        ######################
        realModel.train()
        correct_train, total_train = 0, 0
        for data, target in allLoaders['train']:
            # Transfer to GPU if necessary
            if cuda:
                data, target = data.cuda(), target.cuda()
            
            # Zero the gradients before each update
            opt.zero_grad()

            # Forward pass
            output = realModel(data)
            loss = criterion(output, target)
            
            # Backpropagation and optimization
            loss.backward()
            opt.step()
            
            # Accumulate training loss
            training_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        # Training accuracy and loss
        train_accuracy = (correct_train / total_train) * 100
        training_loss /= len(allLoaders['train'])
        
        ######################
        # Validation phase
        ######################
        realModel.eval()
        correct_valid, total_valid = 0, 0
        
        for data, target in allLoaders['valid']:
            if cuda:
                data, target = data.cuda(), target.cuda()

            # Forward pass
            output = realModel(data)
            loss = criterion(output, target)

            # Accumulate validation loss
            validation_loss += loss.item()

            # Calculate accuracy
            _, val_predicted = torch.max(output.data, 1)
            total_valid += target.size(0)
            correct_valid += (val_predicted == target).sum().item()

        # Validation accuracy and loss
        valid_accuracy = (correct_valid / total_valid) * 100
        validation_loss /= len(allLoaders['valid'])

        # Step the scheduler
        scheduler.step()

        # Print the results for this epoch
        print(f'Epoch: {epoch}\tTraining Loss: {training_loss:.6f}\tTrain Accuracy: {train_accuracy:.2f}\t'
              f'Validation Loss: {validation_loss:.6f}\tValidation Accuracy: {valid_accuracy:.2f}')
        
        # Optionally log to wandb
        # wandb.log({'epoch': epoch, 'train_loss': training_loss, 'train_accuracy': train_accuracy,
        #            'val_loss': validation_loss, 'val_accuracy': valid_accuracy})

    return realModel, valid_accuracy

def sp_train():
    default_config = {
        'epochs': 2,
        'kernel_size_config': '1) 5-5-3-3-3',
        'no_kernel_config': '1) 16-16-16-16-16',
        'dropout_config': '0-0-0.4',
        'fc1_nodes': 32,
        'batch_size': 64
    }

    # Initialize wandb run with default configuration
    wandb.init(config=default_config)
    config = wandb.config

    kernel_sizes = config.kernel_size_config
    kernel_numbers = config.no_kernel_config
    dropout_config = config.dropout_config
    fc1_nodes = str(config.fc1_nodes)
    batch_size = str(config.batch_size)

    run_name = f"kSizes:[{kernel_sizes}] kNumbers:[{kernel_numbers}] dp:[{dropout_config}] fc1:[{fc1_nodes}] bs:[{batch_size}]"
    wandb.run.name = run_name

    # Create the CNN model with the provided configuration
    model = CNN_5layer(
        config_str_list_int(config.kernel_size_config),
        config_str_list_int(config.no_kernel_config),
        {'conv1': 'relu', 'conv2': 'relu', 'conv3': 'relu', 'conv4': 'relu', 'conv5': 'relu', 'fc1': 'relu'},
        config_str_list_float(config.dropout_config),
        config.fc1_nodes, 10
    ).to(device)

    # Optimizer setup
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )

    # Train the model and evaluate on validation accuracy
    trained_model, val_accuracy = train(
        totalEpoch=config.epochs,
        allLoaders=allLoaders,
        realModel=model,
        opt=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=StepLR(optimizer, step_size=10, gamma=0.5),
        cuda=cuda
    )

    # Log validation accuracy to wandb
    wandb.log({'val_accuracy': val_accuracy})

    


batch = 64
dataT1, dataVal, dataT2, lables = data_load()
allLoaders = loader(dataT1, dataVal, dataT2,  batch)

realModel = CNN_5layer(list_kernelSize, kernelNumber, activationFunctions, listDropout, nodesfc1, classes)
realModel = realModel.to(device)

bol=False
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(realModel.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=bol)
scheduler = StepLR(opt, step_size=10, gamma=lr_schedule)

trained_model,_ = train(
                      totalEpoch = epochs,
                      allLoaders = allLoaders,
                      realModel = realModel,
                      opt = opt,
                      criterion = criterion,
                      scheduler = scheduler,
                      cuda = cuda
                     )

trained_model.eval()
test_acc,num_correct,num_examples, test_loss = 0,0,0,0
loader=allLoaders['test']
for data, target in loader:
    bol=cuda
    if bol:
        data, target = data.cuda(), target.cuda()

    output = trained_model(data)
    loss = criterion(output, target)



    test_loss += loss.item()

    _, test_predicted = torch.max(output.data, 1)
    num_examples += target.size(0)
    num_correct += (test_predicted == target).sum().item()


    test_acc = (num_correct / num_examples) * 100
    test_loss = test_loss / len(loader)

print('Test Accuracy of the realModel is : {}%'.format(test_acc, 2))



sweep_config = {
    'name' : 'DL Assigment 2 Part-A',
    'method': 'bayes', 
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': dict({
        'kernel_size_config':{
            'values': ['1) 5-5-3-3-3', 
                       '2) 3-3-3-3-3', 
                       '3) 3-3-3-5-5'
                       ]
        },
        'no_kernel_config': {
            'values': [ 
                      '1) 64-64-32-32-16', 
                      '2) 32-32-64-64-128']
        },
        'dropout_config':{
            'values':['0-0-0.5','0-0-0.3']
        },
        'fc1_nodes':{
            'values': [512,1024]
        },
        'batch_size': {
            'values':[32]
        },
        'epochs': {
            'values':[25]
        },        
    })
}


sweep_id = wandb.sweep(sweep_config, project='DL-Assignment-2')
wandb.agent(sweep_id, sp_train,count=20)
wandb.finish()

'''
# Question-4
fig, axs = plt.subplots(10, 3, figsize=(10, 30))
for i in range(10):
    for j in range(3):
        image, true_label = class_images[i][j]
        axs[i, j].imshow(image)
        axs[i, j].axis('off')
        axs[i, j].set_title(f"True Label: {true_label} | Predicted Label: {lables[get_prediction(image)]}", fontsize=10, ha='center', pad=10, loc='center', color='green')
wandb.init(project='DL-Assignment-2', name = 'Grid')
wandb.log({'Grid': wandb.Image(plt)})
wandb.finish()
plt.show()
'''