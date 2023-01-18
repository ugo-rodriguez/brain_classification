#CUDA_VISIBLE_DEVICES=0

import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Checkpoint')
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Librairies')

import numpy as np
import cv2

import torch
from torch import nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

import monai
import pandas as pd

from net_classification_ASD import BrainNet,BrainIcoNet, BrainIcoAttentionNet
from data_classification_ASD import BrainIBISDataModuleforClassificationASD

from transformation import RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterSphereTransform


import numpy as np
import random
import torch
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

import plotly.express as pd

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRendererWithFragments, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_scene



print('Import Done')

batch_size = 6
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
patience_early_stopping = 20

num_workers = 12 #6-12

path_data = "/MEDUSA_STOR/ugor/IBIS_sa_eacsf_thickness"
train_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12.csv"
val_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12.csv"
test_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12.csv"
path_ico = '/MEDUSA_STOR/ugor/Sphere_Template/sphere_f327680_v163842.vtk'



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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####My data set
brain_data = BrainIBISDataModuleforClassificationASD(batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)
nbr_features = brain_data.get_features()
weights = brain_data.get_weigths()
brain_data.setup()
nbr_brain = brain_data.val_dataset.__len__()
l_all_gray = []













# # #####My model with x
# n_fold = 0

# list_fold = ['F0','F1','F2-2','F3-2','F4']
# list_name_fold = ['epoch=38-val_loss=0.65.ckpt','epoch=118-val_loss=0.54.ckpt','epoch=55-val_loss=0.68.ckpt','epoch=32-val_loss=0.74.ckpt','epoch=39-val_loss=0.76.ckpt']

# name_fold = list_fold[n_fold]
# name = '/'+name_fold+'/'+list_name_fold[n_fold]
# path_model = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Checkpoint"+name


# model = BrainNet(nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, weights,radius=radius,lr=lr,name=name)
# checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['state_dict'])
# model = model.to(device)

# ##### I initialize the gradcam
# model_cam = nn.Sequential(model.TimeDistributed.module,model.WV, model.Classification)
# target_layers = [model_cam[0].layer4[-1]]

# cam = GradCAM(model=model_cam, target_layers=target_layers)

# n_targ = 1
# targets = [ClassifierOutputTarget(n_targ)]

# for j in range(nbr_brain):
#     hemisphere = brain_data.val_dataset.df.loc[j]['Hemisphere']
#     if hemisphere == 'left': 
#         V, F, VF, FF, Y = brain_data.val_dataset.__getitem__(j)
#         #if Y.item() == 1:
#         V = V.unsqueeze(dim=0).to(device)
#         F = F.unsqueeze(dim=0).to(device)
#         VF = VF.unsqueeze(dim=0).to(device)
#         FF = FF.unsqueeze(dim=0).to(device)

#         x,PF = model.render(V,F,VF,FF)
#         input_tensor_cam = x.squeeze(dim=0)


#         #####For each images, I apply gradcam
#         l_gray = []
#         for i in range(12):
#             input_tensor_cami = input_tensor_cam[i].unsqueeze(dim=0)
#             grayscale_cam = cam(input_tensor=input_tensor_cami, targets=targets)


#             grayscale_cami = grayscale_cam[0,:]

#             l_gray.append(torch.tensor(grayscale_cami).unsqueeze(dim=0))

#         t_gray = torch.cat(l_gray,dim=0)
#         t_gray = t_gray.unsqueeze(dim=0)
#         print(j)

#         l_all_gray.append(t_gray)

# #cv2.imshow("Image ",visualization)
# t_all_gray = torch.cat(l_all_gray,dim=0)
# t = torch.mean(t_all_gray,dim=0)
# print(t_all_gray.shape)

# for i in range(12):
#     input_tensor_cami = input_tensor_cam[i]
#     input_tensor_cami = np.array(input_tensor_cami.permute(1,2,0).cpu())
#     print('Hi ', torch.max(t[i]))
#     input_tensor_cami = input_tensor_cami/np.max(input_tensor_cami)
#     visualization = show_cam_on_image(input_tensor_cami, t[i], use_rgb=True)

#     title  = 'GradCam/LeftImage'+name_fold+'_V'+str(i)+'.png'
#     cv2.imwrite(title, visualization)







# ####My model with x
# l_final = []

# list_fold = ['F0','F1','F2-2','F3-2','F4']
# list_name_fold = ['epoch=38-val_loss=0.65.ckpt','epoch=118-val_loss=0.54.ckpt','epoch=55-val_loss=0.68.ckpt','epoch=32-val_loss=0.74.ckpt','epoch=39-val_loss=0.76.ckpt']

# for n_fold in range(5):
#     train_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR2-V06_12_goodpredfold"+str(n_fold)+".csv" #GP
#     val_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR2-V06_12_goodpredfold"+str(n_fold)+".csv" #GP
#     test_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR2-V06_12_goodpredfold"+str(n_fold)+".csv" #GP

#     brain_data = BrainIBISDataModuleforClassificationASD(batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers) #GP
#     nbr_features = brain_data.get_features()#GP
#     brain_data.setup() #GP
#     nbr_brain = brain_data.val_dataset.__len__()#GP

#     name_fold = list_fold[n_fold]
#     name = '/'+name_fold+'/'+list_name_fold[n_fold]
#     path_model = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Checkpoint"+name

#     model = BrainNet(nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, weights,radius=radius,lr=lr,name=name)
#     checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['state_dict'])
#     model = model.to(device)

#     ##### I initialize the gradcam
#     model_cam = nn.Sequential(model.TimeDistributed.module,model.WV, model.Classification)
#     target_layers = [model_cam[0].layer4[-1]]

#     cam = GradCAM(model=model_cam, target_layers=target_layers)

#     n_targ = 1
#     targets = [ClassifierOutputTarget(n_targ)]

#     for j in range(nbr_brain):
#         hemisphere = brain_data.val_dataset.df.loc[j]['Hemisphere']
#         if hemisphere == 'right':
#             V, F, VF, FF, Y = brain_data.val_dataset.__getitem__(j)
#             #if Y.item() == 1:
#             V = V.unsqueeze(dim=0).to(device)
#             F = F.unsqueeze(dim=0).to(device)
#             VF = VF.unsqueeze(dim=0).to(device)
#             FF = FF.unsqueeze(dim=0).to(device)

#             x,PF = model.render(V,F,VF,FF)
#             input_tensor_cam = x.squeeze(dim=0)
            
#             print(x.shape)

#             #####For each images, I apply gradcam
#             l_gray = []
#             for i in range(12):
#                 input_tensor_cami = input_tensor_cam[i].unsqueeze(dim=0)
#                 grayscale_cam = cam(input_tensor=input_tensor_cami, targets=targets)

#                 grayscale_cami = grayscale_cam[0,:]
#                 l_gray.append(torch.tensor(grayscale_cami).unsqueeze(dim=0))

#             t_gray = torch.cat(l_gray,dim=0)
#             t_gray = t_gray.unsqueeze(dim=0)
#             print(j)

#             l_all_gray.append(t_gray)

#     #cv2.imshow("Image ",visualization)
#     t_all_gray = torch.cat(l_all_gray,dim=0)
#     #t = torch.mean(t_all_gray,dim=0)
#     #print(t_all_gray.shape)
#     l_final.append(t_all_gray)

# t_final = torch.cat(l_final,dim=0)
# t = torch.mean(t_final,dim=0)
# torch.save(t,'RightASDGPgradcam.pt')

# for i in range(12):
#     input_tensor_cami = input_tensor_cam[i]
#     input_tensor_cami = np.array(input_tensor_cami.permute(1,2,0).cpu())
#     print('Hi ', torch.max(t[i]))
#     input_tensor_cami = input_tensor_cami/np.max(input_tensor_cami)
#     visualization = show_cam_on_image(input_tensor_cami, t[i], use_rgb=True)

#     title  = 'GradCam/RightASDGPImage'+'Total'+'_V'+str(i)+'.png'
#     #cv2.imwrite(title, visualization)
















































####Total pour ICO
l_final = []

list_fold = ['IcoF0','IcoF1','IcoF2','IcoF3','IcoF4']
list_name_fold = ['epoch=47-val_loss=0.62.ckpt','epoch=81-val_loss=0.55.ckpt','epoch=57-val_loss=0.63.ckpt','epoch=5-val_loss=0.68.ckpt','epoch=47-val_loss=0.71.ckpt']

list_num_fold = [2]
list_fold = ['IcoF2']
list_name_fold = ['epoch=57-val_loss=0.63.ckpt']


for i in range(len(list_num_fold)):
    n_fold = list_num_fold[i]
    train_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12_812857.csv" #GP
    val_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12_812857.csv" #GP
    test_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12_812857.csv" #GP

    brain_data = BrainIBISDataModuleforClassificationASD(batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers) #GP
    nbr_features = brain_data.get_features()#GP
    brain_data.setup() #GP
    nbr_brain = brain_data.val_dataset.__len__()#GP

    name_fold = list_fold[i]
    name = '/'+name_fold+'/'+list_name_fold[i]
    path_model = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Checkpoint"+name

    model = BrainIcoNet(nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, weights,radius=radius,lr=lr,name=name)
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    ##### I initialize the gradcam
    model_cam = nn.Sequential(model.TimeDistributed, model.IcosahedronConv2d, model.pooling,model.Classification)
    target_layers = [model_cam[0].module.layer4[-1]]

    # model_cam[1].set_gradcam(True)

    cam = GradCAM(model=model_cam, target_layers=target_layers)

    n_targ = 1
    targets = [ClassifierOutputTarget(n_targ)]

    for j in range(nbr_brain):
        hemisphere = brain_data.val_dataset.df.loc[j]['Hemisphere']
        if hemisphere == 'right':
            V, F, VF, FF, Y = brain_data.val_dataset.__getitem__(j)
            #if Y.item() == 1:
            V = V.unsqueeze(dim=0).to(device)
            F = F.unsqueeze(dim=0).to(device)
            VF = VF.unsqueeze(dim=0).to(device)
            FF = FF.unsqueeze(dim=0).to(device)

            x,PF = model.render(V,F,VF,FF)

            l_gray = []


            # input_tensor_cam = x.squeeze(dim=0)
            # model_cam[1].keep_neighbors_for_gradcam(model.TimeDistributed.module(input_tensor_cam))
            #print(x.shape)
            
            #####For each images, I apply gradcam
            
            # for i in range(12):
            #     input_tensor_cami = input_tensor_cam[i].unsqueeze(dim=0)
            #     #print(input_tensor_cami.shape)
            #     # model_cam[1].keep_position_for_gradcam(i)
            #     grayscale_cam = cam(input_tensor=input_tensor_cam, targets=targets)

            #     grayscale_cami = grayscale_cam[0,:]
            #     l_gray.append(torch.tensor(grayscale_cami).unsqueeze(dim=0))

            input_tensor_cam = x
            grayscale_cam = torch.Tensor(cam(input_tensor=input_tensor_cam, targets=targets))

            #print(grayscale_cam.shape)

            #grayscale_cami = grayscale_cam[0,:]
            #l_gray.append(torch.tensor(grayscale_cami).unsqueeze(dim=0))

            # t_gray = torch.cat(l_gray,dim=0)
            # t_gray = t_gray.unsqueeze(dim=0)
            print(j)

            l_final.append(grayscale_cam.unsqueeze(dim=1))

    #cv2.imshow("Image ",visualization)
    #t_all_gray = torch.cat(l_all_gray,dim=0)
    #t = torch.mean(t_all_gray,dim=0)
    #print(t_all_gray.shape)
    #l_final.append(t_all_gray)

t_final = torch.cat(l_final,dim=1)
t = torch.mean(t_final,dim=1)
#torch.save(t,'Right812857gradcam.pt')

for i in range(12):
    print(input_tensor_cam.shape)
    image = input_tensor_cam[0][i]
    image = np.array(image.permute(1,2,0).cpu())
    print('Hi ', torch.max(t[i]))
    image = image/np.max(image)
    visualization = show_cam_on_image(image, t[i], use_rgb=True)

    title  = 'GradCam/Individual_image/Right812857_V'+str(i)+'.png'
    cv2.imwrite(title, visualization)









print(2)