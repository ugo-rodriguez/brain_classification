#CUDA_VISIBLE_DEVICES=1

import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Librairies')

import numpy as np
import torch
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import monai
import nibabel as nib


from net_classification_ASD import BrainNet,BrainIcoNet, BrainIcoAttentionNet
from data_classification_ASD import BrainIBISDataModuleforClassificationASD
from logger_classification_ASD import BrainNetImageLogger

from transformation import RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterSphereTransform


print("Import // done")

def main():

    name = "Test"
    print('name to save checkpoints : ',name)

    batch_size = 10 #8-20
    num_workers = 12 #6-12


    image_size = 224
    noise_lvl = 0.01
    dropout_lvl = 0.2
    num_epochs = 1
    ico_lvl = 1
    radius = 2.0
    lr = 1e-5

    mean = 0
    std = 0.005

    min_delta_early_stopping = 0.00
    patience_early_stopping = 40

    path_data = "/MEDUSA_STOR/ugor/IBIS_sa_eacsf_thickness"
    train_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold4_train.csv"
    val_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold4_val.csv"
    test_path = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDHR-V06_12fold4_test.csv"
    path_ico = '/MEDUSA_STOR/ugor/Sphere_Template/sphere_f327680_v163842.vtk'

    list_nb_verts_ico = [12,42]
    nb_verts_ico = list_nb_verts_ico[ico_lvl-1]

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

    checkpoint_callback = ModelCheckpoint(
        dirpath='Checkpoint/'+name,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=200,
        monitor='val_loss'
    )

    logger = TensorBoardLogger(save_dir="test_tensorboard", name="my_model")  

    brain_data = BrainIBISDataModuleforClassificationASD(batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)
    nbr_features = brain_data.get_features()
    weights = brain_data.get_weigths()

    model = BrainIcoNet(nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, weights,radius=radius,lr=lr,name=name)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode="min")

    image_logger = BrainNetImageLogger(num_features = nbr_features,num_images = nb_verts_ico,mean = 0,std=noise_lvl)

    #trainer = Trainer(max_epochs=num_epochs,callbacks=[early_stop_callback])
    trainer = Trainer(log_every_n_steps=20,reload_dataloaders_every_n_epochs=True,logger=logger,max_epochs=num_epochs,callbacks=[early_stop_callback,checkpoint_callback,image_logger],accelerator="gpu") #,profiler="advanced"

    trainer.fit(model,datamodule=brain_data)

    trainer.test(model, datamodule=brain_data)

    torch.save(model.state_dict(), 'ModeleBrainClassificationIcoConvAvgPooling.pth')

if __name__ == '__main__':
    main()
