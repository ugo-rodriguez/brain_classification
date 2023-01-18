#CUDA_VISIBLE_DEVICES=0

import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Classification_ASD')

import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Librairies')

import os

import numpy as np
import cv2

import torch
from torch import nn
from torchvision.models import resnet50

import monai
import pandas as pd

from net_classification_ASD import BrainNet,BrainIcoNet, BrainIcoAttentionNet
from data_classification_ASD import BrainIBISDataModuleforClassificationASD

from transformation import RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterSphereTransform

from sklearn.metrics import classification_report

import numpy as np
import random
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl 

from vtk.util.numpy_support import vtk_to_numpy
import vtk

import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight

import utils
from utils import ReadSurf, PolyDataToTensors

import pandas as pd


batch_size = 10
num_workers = 12 #6-12
image_size = 224
noise_lvl = 0.03
dropout_lvl = 0.2
num_epochs = 100
ico_lvl = 1
radius=2
lr = 1e-5

mean = 0
std = 0.01

min_delta_early_stopping = 0.00
patience_early_stopping = 30

path_data = "/MEDUSA_STOR/ugor/IBIS_sa_eacsf_thickness"
train_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold1_train.csv"
val_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold1_val.csv"
test_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold1_test.csv"
path_ico = '/MEDUSA_STOR/ugor/Sphere_Template/sphere_f327680_v163842.vtk'

path_test = '/Ico42F1' 
way = "/work/ugor/source/brain_classification/Classification_ASD/Checkpoint"
path_checkpoint = way + path_test

list_name_model = os.listdir(path_checkpoint)
list_name_model = ['epoch=38-val_loss=0.65.ckpt','epoch=48-val_loss=0.67.ckpt']

list_fold = ['/F0','/F1','/F2-2','/F3-2','/F4'] #+
list_name_model = ['epoch=38-val_loss=0.65.ckpt','epoch=118-val_loss=0.54.ckpt','epoch=55-val_loss=0.68.ckpt','epoch=32-val_loss=0.74.ckpt','epoch=39-val_loss=0.76.ckpt'] #+

list_fold = ['/IcoF0','/IcoF1','/IcoF2','/IcoF3','/IcoF4'] #+
list_name_model = ['epoch=47-val_loss=0.62.ckpt','epoch=81-val_loss=0.55.ckpt','epoch=57-val_loss=0.63.ckpt','epoch=5-val_loss=0.68.ckpt','epoch=47-val_loss=0.71.ckpt']#+


y_pred_total = [] 
y_true_total = []

for i in range(len(list_name_model)):
    train_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold"+str(i)+"_train.csv" #+
    val_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold"+str(i)+"_val.csv" #+
    test_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold"+str(i)+"_test.csv" #+

    path_test = list_fold[i] #+
    name_model = list_name_model[i]
    print('Nummmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmber : ', i)
    print('Path : ',path_test,'   Name model : ',name_model)

    path_model = way + path_test + '/' + name_model

    list_train_transform = []    
    list_train_transform.append(CenterSphereTransform())
    list_train_transform.append(NormalizePointTransform())
    list_train_transform.append(RandomRotationTransform())
    list_train_transform.append(GaussianNoisePointTransform(mean,std))
    list_train_transform.append(NormalizePointTransform())

    train_transform = monai.transforms.Compose(list_train_transform)

    list_val_and_test_transform = []    
    list_val_and_test_transform.append(CenterSphereTransform())
    list_val_and_test_transform.append(NormalizePointTransform())

    val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

    brain_data = BrainIBISDataModuleforClassificationASD(batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform, num_workers=num_workers)

    nbr_features = brain_data.get_features()
    weights = brain_data.get_weigths()

    model = BrainIcoNet(nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, weights,radius=radius,lr=lr)
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint['state_dict'])

    trainer = Trainer(max_epochs=num_epochs,accelerator="gpu")

    trainer.test(model, datamodule=brain_data)

    y_pred,y_true = model.get_y_for_report_classification()
    y_pred_total.extend(y_pred)
    y_true_total.extend(y_true)

target_names = ['no ASD','ASD']
cr = classification_report(y_true_total, y_pred_total, target_names=target_names)
print(cr)

print(2)