import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

name = '812857'
list_orientation = ['A','I','L','P','R','S']
list_hemisphere = ['left','right']
list_img = [cv2.imread("Classification_ASD/Gradcam/Image_brain/"+name+"_"+hemisphere+"_"+orientation+".png",cv2.IMREAD_COLOR) for orientation in list_orientation for hemisphere in list_hemisphere]
#list_img = [cv2.imread("Image_brain/"+name+"_"+hemisphere+"_"+orientation+".png",cv2.IMREAD_COLOR) for orientation in list_orientation for hemisphere in list_hemisphere]

#Determine standard size
list_taille = torch.tensor([[img.shape[0],img.shape[1]] for img in list_img])
nrb_rows = torch.min(list_taille,dim=0)[0][0]
nbr_columns = torch.min(list_taille,dim=0)[0][1]

for i in range(len(list_img)):
    #resize images
    list_img[i] = list_img[i][:nrb_rows,:nbr_columns]
    #Change features position for color
    temp = np.copy(list_img[i][:,:,0])
    list_img[i][:,:,0] = np.copy(list_img[i][:,:,2])
    list_img[i][:,:,2] = np.copy(temp)

image_top = cv2.hconcat([list_img[8],list_img[0],list_img[1],list_img[5]])
image_middle = cv2.hconcat([list_img[4],list_img[6],list_img[7],list_img[9]])
image_bottom = cv2.hconcat([list_img[11],list_img[10],list_img[2],list_img[3]])
image_final = [image_top,image_middle,image_bottom]
image_final = cv2.vconcat(image_final)

plt.figure(1)
plt.imshow(image_final)

print(2)