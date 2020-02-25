import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    f, axarr=plt.subplots(2,3)
    img = cv2.imread('LeNet++/Final_Plots/Not_MNIST/Vanilla_0_2D_plot.png')
    axarr[0,0].imshow(img)
    axarr[0,0].set_title('Softmax')
    axarr[0,0].axis('off')

    img = cv2.imread('LeNet++/Final_Plots/Not_MNIST/Ring_50.0_0_2D_plot.png')
    axarr[0,1].imshow(img)
    axarr[0,1].set_title('Objectosphere')
    axarr[0,1].axis('off')
    
    img = cv2.imread('LeNet++/Final_Plots/Not_MNIST/Center_50.0_0_2D_plot.png')
    axarr[0,2].imshow(img)
    axarr[0,2].set_title('Center')
    axarr[0,2].axis('off')
    
    img = cv2.imread('LeNet++/Final_Plots/Not_MNIST/Vanilla_0_Hist.png')
    axarr[1,0].imshow(img)
    axarr[1,0].set_title('Softmax')
    axarr[1,0].axis('off')
    
    img = cv2.imread('LeNet++/Final_Plots/Not_MNIST/Ring_50.0_0_Hist.png')
    axarr[1,1].imshow(img)
    axarr[1,1].set_title('Objectosphere')
    axarr[1,1].axis('off')
    
    img = cv2.imread('LeNet++/Final_Plots/Not_MNIST/Center_50.0_0_Hist.png')
    axarr[1,2].imshow(img)
    axarr[1,2].set_title('Center')
    axarr[1,2].axis('off')
    
    file_name = 'result_collage.png'
    plt.savefig(file_name, dpi=1200)
    plt.show()

