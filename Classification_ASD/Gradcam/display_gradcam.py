import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


###Display 1 : To get 12 images for each views. We concatene the images of 5 folds
# list_fold = ['F0','F1','F2-2','F3-2','F4']

# for i in range(12):
#     list_img = [cv2.imread("GradCam/LeftImage"+fold+"_V"+str(i)+".png",cv2.IMREAD_COLOR) for fold in list_fold]
#     image_final = cv2.hconcat(list_img)

#     plt.figure(i)
#     plt.imshow(image_final)



###Display 2 : To get an average image for each views
name = "RightASDGPImage"
list_img = [cv2.imread("Classification_ASD/Gradcam/Individual_image/"+name+"_V"+str(i)+".png",cv2.IMREAD_COLOR) for i in range(12)]
#list_img = [cv2.imread("Individual_image/"+name+"_V"+str(i)+".png",cv2.IMREAD_COLOR) for i in range(12)]
image_top = cv2.hconcat(list_img[:6])
image_bottom = cv2.hconcat(list_img[6:])
image_final = cv2.vconcat([image_top,image_bottom])

plt.figure(1)
plt.imshow(image_final)

print(2)