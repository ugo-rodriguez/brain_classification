import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Librairies')

import os
import numpy as np
import torch
import vtk
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import utils
from utils import ReadSurf, PolyDataToTensors

# datastructures
from pytorch3d.structures import Meshes

from fsl.data import gifti

import pandas as pd

print("Import done")

path_csv =  "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness/DX_IBIS_Sept22.csv"

df = pd.read_csv(path_csv)
title = df.columns

#Extraction of information from DX_IBIS
list_V06_csv = df[["V06 demographics,ASD_Ever_DSMIV",'V06 demographics,CandID']]
list_V06_csv["Age"] = "0"
list_V06_csv = list_V06_csv.rename(columns={"V06 demographics,ASD_Ever_DSMIV":"ASD",'V06 demographics,CandID':'ID'})
list_V06_csv = list_V06_csv.query("ASD != '.' and ASD != 'No DSMIV ever administered' ")

list_V12_csv = df[["V12 demographics,ASD_Ever_DSMIV",'V12 demographics,CandID']]
list_V12_csv["Age"] = "1"
list_V12_csv = list_V12_csv.rename(columns={"V12 demographics,ASD_Ever_DSMIV":"ASD",'V12 demographics,CandID':'ID'})
list_V12_csv = list_V12_csv.query("ASD != '.' and ASD != 'No DSMIV ever administered' ")


#We use a filter to know if we have the features (eacsf, sa, thickness) of the brain
path_ID = '/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain/Regression/ID_with_inf.csv'

df_ID = pd.read_csv(path_ID)

for i in range(len(df_ID)):
    df_ID.iloc[i,0] = str(df_ID.iloc[i,0])

list_V06_csv = pd.merge(list_V06_csv,df_ID,on=['ID'])
list_V06_csv = list_V06_csv.query("V06 == 1.0")
list_V06_csv = list_V06_csv.drop(['V06','V12'], axis=1)

list_V12_csv = pd.merge(list_V12_csv,df_ID,on=['ID'])
list_V12_csv = list_V12_csv.query("V12 == 1.0")
list_V12_csv = list_V12_csv.drop(['V06','V12'], axis=1)

#Risk
path =  "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness/DemIBIS1-2.csv"

df_Risk = pd.read_csv(path)

list_Risk = df_Risk[["CandID",'Risk']]
list_Risk = list_Risk.rename(columns={'CandID':'ID'})

for i in range(len(list_Risk)):
    list_Risk.iloc[i,0] = str(list_Risk.iloc[i,0])

list_V06_csv =  pd.merge(list_V06_csv,list_Risk,on=['ID'])  
list_V06_csv = list_V06_csv.query("not(Risk.isnull())")

list_V12_csv = pd.merge(list_V12_csv,list_Risk,on=['ID'])  
list_V12_csv = list_V12_csv.query("not(Risk.isnull())")

#Last step
save_name_06 = 'dataASDHR2-V06.csv'
save_path_06 = '/work/ugor/source/brain_classification/Classification_ASD/Data/'+save_name_06

save_name_12 = 'dataASDHR2-V12.csv'
save_path_12 = '/work/ugor/source/brain_classification/Classification_ASD/Data/'+save_name_12

dataASDHR_V06 = list_V06_csv.query("Risk == 'HR'")
dataASDHR_V06 = dataASDHR_V06.rename(columns={'ID':'Subject_ID'})
for i in range(len(dataASDHR_V06)):
    ASD = dataASDHR_V06.iloc[i,0] 
    if (ASD == 'ASD-'):
        dataASDHR_V06.iloc[i,3] += '-'
    else:
        dataASDHR_V06.iloc[i,3] += '+'
dataASDHR_V06 = dataASDHR_V06.drop(['ASD'], axis=1)   
dataASDHR_V06 = dataASDHR_V06.rename(columns={'Risk':'ASD_administered'})

dataASDHR_V12 = list_V12_csv.query("Risk == 'HR'")
dataASDHR_V12 = dataASDHR_V12.rename(columns={'ID':'Subject_ID'})
for i in range(len(dataASDHR_V12)):
    ASD = dataASDHR_V12.iloc[i,0] 
    if (ASD == 'ASD-'):
        dataASDHR_V12.iloc[i,3] += '-'
    else:
        dataASDHR_V12.iloc[i,3] += '+'
dataASDHR_V12 = dataASDHR_V12.drop(['ASD'], axis=1) 
dataASDHR_V12 = dataASDHR_V12.rename(columns={'Risk':'ASD_administered'})


#dataASDHR_V06.to_csv(save_path_06,index=False)

#dataASDHR_V12.to_csv(save_path_12,index=False)


#V06 and V12
dataASDHR_V06_12 = pd.merge(dataASDHR_V06.drop(['Age'], axis=1),dataASDHR_V12.drop(['ASD_administered','Age'], axis=1),on=['Subject_ID'])

save_name_06_12 = 'dataASDHR2-V06_12.csv'
save_path_06_12 = '/work/ugor/source/brain_classification/Classification_ASD/Data/'+save_name_06_12

#Add hemisphere
dataASDHR_V06_12 = pd.concat([dataASDHR_V06,dataASDHR_V12],ignore_index = True)
dataASDHR_V06_12L = dataASDHR_V06_12.copy()
dataASDHR_V06_12R = dataASDHR_V06_12.copy()
dataASDHR_V06_12L["Hemisphere"] = "left"
dataASDHR_V06_12R["Hemisphere"] = "right"
dataASDHR_V06_12 = pd.concat([dataASDHR_V06_12L,dataASDHR_V06_12R],ignore_index = True)
dataASDHR_V06_12 = dataASDHR_V06_12.sort_values(by = 'Subject_ID')

save_name_06_12 = 'dataASDHR-V06_12.csv'
save_path_06_12 = '/work/ugor/source/brain_classification/Classification_ASD/Data/'+save_name_06_12
#dataASDHR_V06_12.to_csv(save_path_06_12,index=False)



###Add demographics
path_DX_and_Dem =  "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness/DX_and_Dem.csv"
df_demographics = pd.read_csv(path_DX_and_Dem)[['CandID','Gender','MRI_Age','ASD_DX','Project']]
df_demographics = df_demographics.rename(columns={'CandID':'Subject_ID'})
for i in range(len(df_demographics)):
    df_demographics.iloc[i,0] = str(df_demographics.iloc[i,0])

dataASDHR_V06_12 = pd.merge(dataASDHR_V06_12,df_demographics,on=['Subject_ID'])

save_name_06_12 = 'dataASDdemographics-V06_12.csv'
save_path_06_12 = '/work/ugor/source/brain_classification/Classification_ASD/Data/'+save_name_06_12
#dataASDHR_V06_12.to_csv(save_path_06_12,index=False)

print(2)