import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Librairies')


import numpy as np

import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import utils 

import post_process 

# ID_patient = '' #101247
# version = 'V06'
# hemishpere = 'left'
# path = '/ASD/Autism2/IBIS1/IBIS/Proc_Data/'+ID_patient+'/'+version+'/mri/registered_v3.6/sMRI/CIVETv2.0_Surface'
# path_white_surface = path+'/stx_stx_noscale_'+ID_patient+'_'+version+'_t1w_white_surface_rsl_'+hemishpere+'_327680_native_NRRDSpace.vtk'

# path_ico = '/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk'

# surface_ws = utils.ReadSurf(path_white_surface)
# verts_ws, faces_ws, edges_ws = utils.PolyDataToTensors(surface_ws)
# v_ws = torch.Tensor(verts_ws)
# f_ws = torch.tensor(faces_ws)
# e_ws = torch.tensor(edges_ws)

# surface_ico = utils.ReadSurf(path_ico)
# verts_ico, faces_ico, edges_ico = utils.PolyDataToTensors(surface_ico)
# v_ico = torch.Tensor(verts_ico)
# f_ico = torch.tensor(faces_ico)
# e_ico = torch.tensor(edges_ico)

#path_brain = '/tools/atlas/Surface/CIVET_160K/icbm_surface/icbm_avg_white_sym_mc_left_hires_flatten.vtk'
path_brain = '/tools/atlas/Surface/CIVET_160K/icbm_surface/icbm_avg_mid_sym_mc_right_hires.vtk'
surface = utils.ReadSurf(path_brain)
verts, faces, edges = utils.PolyDataToTensors(surface)

nbr_verts = verts.shape[0]
nbr_faces = faces.shape[0]

Colors = vtk.vtkDoubleArray()
Colors.SetNumberOfComponents(1)
Colors.SetName('Colors')

#Nbr_points = surface.GetNumberOfPoints()

PF = torch.load('PF.pt', map_location=torch.device('cpu')).squeeze(dim=2)[0]
num_views = PF.shape[0]
image_size = PF.shape[1]
gradcam = torch.load('Right812857gradcam.pt', map_location=torch.device('cpu'))

# L_faces_color = [[] for i in range(nbr_verts)]
# for cam in range(num_views):
#     print(cam)
#     for i in range(224):
#         for j in range(224):
#             index_face = PF[cam][i][j].item()
#             if  index_face != -1:
#                 face = faces[index_face]
#                 gradcam_val = gradcam[cam][i][j].item()
#                 L_faces_color[face[0].item()].append(gradcam_val)
#                 L_faces_color[face[1].item()].append(gradcam_val)
#                 L_faces_color[face[2].item()].append(gradcam_val)

gradcam_points = torch.zeros(nbr_verts,3)
intermediaire_gradcam_faces = torch.zeros(nbr_faces)
gradcam_faces = torch.zeros(nbr_faces)
gradcam_count = torch.zeros(nbr_faces)

for cam in range(num_views):

    reshape_size = [image_size*image_size]

    PF_image = PF[cam]
    PF_image = PF_image.contiguous().view(reshape_size)

    gradcam_image = gradcam[cam]
    gradcam_image = gradcam_image.contiguous().view(reshape_size)

    intermediaire_gradcam_faces[PF_image] = gradcam_image

    gradcam_faces += intermediaire_gradcam_faces
    gradcam_count[PF_image] += 1.0

    intermediaire_gradcam_faces = torch.zeros(nbr_faces)


print(2)
zeros_count = ((gradcam_count == 0).nonzero(as_tuple=True)[0])
gradcam_count[zeros_count] = torch.ones(zeros_count.shape[0])
gradcam_faces = torch.div(gradcam_faces,gradcam_count)
gradcam_faces[-1] = gradcam_faces[-2]



for i in range(3):
    ID_verts = faces[:,i]
    gradcam_points[:,i][ID_verts.long()] = gradcam_faces
gradcam_points = torch.max(gradcam_points,dim=1)[0].to(torch.double)
print(gradcam_points)


Colors = vtk.vtkDoubleArray()
Colors.SetNumberOfComponents(1)
Colors.SetName('Colors')

for c in range(nbr_verts):
    Colors.InsertNextTypedTuple([gradcam_points[c].item()])
surface.GetPointData().SetScalars(Colors)

writer = vtk.vtkPolyDataWriter()
writer.SetInputData(surface)
title = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/GradCam/Surface/right812857_brain.vtk"
writer.SetFileName(title)
writer.Update()
writer.Write()



print(2)