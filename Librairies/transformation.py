import numpy as np
import torch
import pytorch3d
# from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


import utils
from utils import GetUnitSurf, RandomRotation

# class RandomRotationTransform:
#     def __call__(self, verts):
#         theta = np.random.random()*360.0
#         vector = np.random.random(3)*2.0 - 1.0
#         vector = vector/np.linalg.norm(vector)

#         c = np.cos(theta)
#         s = np.sin(theta)

#         ux,uy,uz = vector[0],vector[1],vector[2]

#         rotation_matrix = torch.tensor([[(ux**2)*(1-c)+c,ux*uy*(1-c)-uz*s,ux*uz*(1-c)+uy*s],
#             [ux*uy*(1-c)+uz*s,(uy**2)*(1-c)+c,uy*uz*(1-c)-ux*s],
#             [ux*uz*(1-c)-uy*s,uy*uz*(1-c)+ux*s,(uz**2)*(1-c)+c]]
#         )

#         rotation_matrix = rotation_matrix.type(torch.float32) 

#         verts = torch.transpose(torch.mm(rotation_matrix,torch.transpose(verts,0,1)),0,1)

#         return verts

class RotationTransform:
    def __call__(self, verts, rotation_matrix):
        verts = torch.transpose(torch.mm(rotation_matrix,torch.transpose(verts,0,1)),0,1)
        return verts

class RandomRotationTransform:
    def __call__(self, verts):
        rotation_matrix = pytorch3d.transforms.random_rotation()
        rotation_transform = RotationTransform()
        verts = rotation_transform(verts,rotation_matrix)
        return verts

class ApplyRotationTransform:
    def __init__(self):            
        self.rotation_matrix = pytorch3d.transforms.random_rotation()

    def __call__(self, verts):
        rotation_transform = RotationTransform()
        verts = rotation_transform(verts,self.rotation_matrix)
        return verts
    
    def change_rotation(self):
        self.rotation_matrix = pytorch3d.transforms.random_rotation()

class GaussianNoisePointTransform:
    def __init__(self, mean=0.0, std = 0.1):            
        self.mean = mean
        self.std = std

    def __call__(self, verts):
        noise = np.random.normal(loc=self.mean, scale=self.std, size=verts.shape)
        verts = verts + noise
        verts = verts.type(torch.float32)
        return verts

class NormalizePointTransform:
    def __call__(self, verts, scale_factor=1.0):
        verts = verts/np.linalg.norm(verts, axis=1, keepdims=True)*scale_factor
        verts = torch.Tensor(verts)
        verts = verts.type(torch.float32)
        return verts

class CenterSphereTransform:
    def __call__(self, verts,mean_arr = None):
        #calculate bounding box
        mean_v = [0.0] * 3
        # bounds_max_v = [0.0] * 3

        v = torch.Tensor(verts)

        bounds = torch.tensor([torch.min(v[:,0]),torch.max(v[:,0]),torch.min(v[:,1]),torch.max(v[:,1]),torch.min(v[:,2]),torch.max(v[:,2])])

        mean_v[0] = (bounds[0] + bounds[1])/2.0
        mean_v[1] = (bounds[2] + bounds[3])/2.0
        mean_v[2] = (bounds[4] + bounds[5])/2.0
        # bounds_max_v[0] = torch.max(bounds[0], bounds[1])
        # bounds_max_v[1] = torch.max(bounds[2], bounds[3])
        # bounds_max_v[2] = torch.max(bounds[4], bounds[5])
        
        #centering points of the shape
        mean_arr = torch.tensor(mean_v)

        # print("Mean:", mean_arr)
        verts = verts - mean_arr
        return verts 

print(2)
