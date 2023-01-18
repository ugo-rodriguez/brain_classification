import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Librairies')

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import argparse
import sys
import os
import math
from collections import namedtuple
from utils import * 


def ReadFile(filename, array_name="RegionId"):
    inputSurface = filename
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(inputSurface)
    reader.Update()
    vtkdata = reader.GetOutput()
    label_array = vtkdata.GetPointData().GetArray(array_name)
    print(array_name)
    return vtkdata, label_array

def Write(vtkdata, output_name):
    outfilename = output_name
    print("Writting:", outfilename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(outfilename)
    polydatawriter.SetInputData(vtkdata)
    polydatawriter.Write()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
    parser.add_argument('--median_filter', type=bool, help='Apply a median filter', default=False)
    parser.add_argument('--array_name', type=str, help='Name of the array in the surface mesh to use', default='RegionId')
    parser.add_argument('--out', type=str, help='Output model with labels', default="out.vtk")


    args = parser.parse_args()
    surf, labels = ReadFile(args.surf, args.array_name)

    if(args.median_filter):
        nbr_points = surf.GetNumberOfPoints()
        list_values = vtk_to_numpy(surf.GetPointData().GetArray(args.array_name))
        list_median_values = np.zeros(nbr_points)
        for i in range(nbr_points):
            list_neighbors = GetNeighbors(surf,i)
            list_neighbors.append(i)
            list_neighbors = GetAllNeighbors(surf,list_neighbors)
            median_value = np.median(list_values[list_neighbors])
            list_median_values[i] = median_value
        print(np.max(np.abs(list_values - list_median_values)))
        list_median_values = numpy_to_vtk(list_median_values)
        list_median_values.SetName(args.array_name)
        surf.GetPointData().SetScalars(list_median_values)


    #Write(surf, args.out)