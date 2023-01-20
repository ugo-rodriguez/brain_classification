import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Librairies')

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl 
from torch.nn.functional import normalize

import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight

import utils
from utils import ReadSurf, PolyDataToTensors

import pandas as pd

class BrainIBISDatasetforClassificationASD(Dataset):
    def __init__(self,df,path_data,path_ico,transform = None,version=None, column_subject_id='Subject_ID', column_age='Age',column_hemisphere = 'Hemisphere',column_ASD = 'ASD_administered',column_gender = 'Gender',column_MRI_Age = 'MRI_Age'):
        self.df = df
        self.path_data = path_data
        self.path_ico = path_ico
        self.transform = transform
        self.version = version
        self.column_subject_id =column_subject_id
        self.column_age = column_age
        self.column_hemisphere = column_hemisphere
        self.column_ASD = column_ASD
        self.column_gender = column_gender
        self.column_MRI_Age = column_MRI_Age

        self.change_rotation = False
        for i in range(len(self.transform)):
            if self.transform.transforms[i].__class__.__name__ == 'ApplyRotationTransform':
                self.change_rotation = True
                self.index_ApplyRotationTransform = i

    def __len__(self):
        return(len(self.df)) 

    def __getitem__(self,idx):

        row = self.df.loc[idx]

        l_version = ['V06','V12']
        idx_version = int(row[self.column_age])
        version = l_version[idx_version]

        hemisphere = row[self.column_hemisphere]

        verts, faces, vertex_features, face_features, information, Y = self.getitem_per_hemisphere_age(version,hemisphere, idx)

        return  verts,faces,vertex_features,face_features,information,Y
    
    def getitem_per_hemisphere_age(self,age,hemisphere,idx):
        #Load Data
        row = self.df.loc[idx]
        number_brain = row[self.column_subject_id]

        version = age

        idx_ASD = int(row[self.column_ASD])

        list_gender = ['Male','Female']
        gender = float(list_gender.index(row[self.column_gender]))

        MRI_Age = float(row[self.column_MRI_Age])

        information = torch.tensor([gender])

        l_features = []

        path_eacsf = f"{self.path_data}/{number_brain}/{version}/eacsf/{hemisphere}_eacsf.txt"
        path_sa =    f"{self.path_data}/{number_brain}/{version}/sa/{hemisphere}_sa.txt"
        path_thickness = f"{self.path_data}/{number_brain}/{version}/thickness/{hemisphere}_thickness.txt"

        eacsf = open(path_eacsf,"r").read().splitlines()
        eacsf = torch.tensor([float(ele) for ele in eacsf])
        l_features.append(eacsf.unsqueeze(dim=1))

        sa = open(path_sa,"r").read().splitlines()
        sa = torch.tensor([float(ele) for ele in sa])
        l_features.append(sa.unsqueeze(dim=1))

        thickness = open(path_thickness,"r").read().splitlines()
        thickness = torch.tensor([float(ele) for ele in thickness])
        l_features.append(thickness.unsqueeze(dim=1))

        vertex_features = torch.cat(l_features,dim=1)

        Y = torch.tensor([idx_ASD])

        #Load  Icosahedron
        reader = utils.ReadSurf(self.path_ico)
        verts, faces, edges = utils.PolyDataToTensors(reader)

        nb_faces = len(faces)

        #Transformations
        if self.transform:        
            verts = self.transform(verts)

        #Face Features
        faces_pid0 = faces[:,0:1]         
    
        offset = torch.zeros((nb_faces,vertex_features.shape[1]), dtype=int) + torch.Tensor([i for i in range(vertex_features.shape[1])]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])      
        
        face_features = torch.take(vertex_features,faces_pid0_offset)

        return verts, faces,vertex_features,face_features, information, Y

class BrainIBISDataModuleforClassificationASD(pl.LightningDataModule):
    def __init__(self,batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform=None,val_and_test_transform=None, num_workers=6, pin_memory=False, persistent_workers=False):
        super().__init__()
        self.batch_size = batch_size
        self.path_data = path_data
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.path_ico = path_ico
        self.train_transform = train_transform
        self.val_and_test_transform = val_and_test_transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers=persistent_workers

        self.weights = []

        self.df_train = pd.read_csv(self.train_path)
        self.df_val = pd.read_csv(self.val_path)
        self.df_test = pd.read_csv(self.test_path)

        y_train = np.array(self.df_train.loc[:,'ASD_administered'])
        labels = np.unique(y_train)
        class_weights_train  = torch.tensor(class_weight.compute_class_weight('balanced',classes=labels,y=y_train)).to(torch.float32) 
        self.weights.append(class_weights_train) 

        y_val = np.array(self.df_val.loc[:,'ASD_administered'])
        class_weights_val = torch.tensor(class_weight.compute_class_weight('balanced',classes=labels,y=y_val)).to(torch.float32)
        self.weights.append(class_weights_val) 

        y_test = np.array(self.df_test.loc[:,'ASD_administered'])
        class_weights_test = torch.tensor(class_weight.compute_class_weight('balanced',classes=labels,y=y_test)).to(torch.float32)
        self.weights.append(class_weights_test) 

        self.setup()


    def setup(self,stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_dataset = BrainIBISDatasetforClassificationASD(self.df_train,self.path_data,self.path_ico,self.train_transform)
        self.val_dataset = BrainIBISDatasetforClassificationASD(self.df_val,self.path_data,self.path_ico,self.val_and_test_transform)
        self.test_dataset = BrainIBISDatasetforClassificationASD(self.df_test,self.path_data,self.path_ico,self.val_and_test_transform)

        V, F, VF, FF, information, Y = self.train_dataset.__getitem__(0)
        self.nbr_features = V.shape[1]
        self.nbr_information = information.shape[0]

    def train_dataloader(self):
        df_healthy = self.df_train.query('ASD_administered == 0')
        df_ASD = self.df_train.query('ASD_administered == 1')
        nbr_ASD = len(df_ASD)
        df_healthy = df_healthy.sample(nbr_ASD)
        new_df_train = pd.concat([df_healthy,df_ASD]).reset_index()
        self.train_dataset = BrainIBISDatasetforClassificationASD(new_df_train,self.path_data,self.path_ico,self.train_transform)
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)        

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def get_features(self):
        return self.nbr_features

    def get_weigths(self):
        return self.weights

    def get_nbr_information(self):
        return self.nbr_information
