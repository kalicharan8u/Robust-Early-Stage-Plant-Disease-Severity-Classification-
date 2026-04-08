import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def Image_Results():
    I = [96, 371, 400, 484, 794]
    Images = np.load('Image.npy', allow_pickle=True)
    GT = np.load('Ground_Truth.npy', allow_pickle=True)
    UNET = np.load('Unet.npy', allow_pickle=True)
    RESUNET = np.load('Unet3+.npy', allow_pickle=True)
    PROPOSED = np.load('Seg_APSPN.npy', allow_pickle=True)
    for i in range(len(I)):
        plt.subplot(2, 3, 1)
        plt.title('Original')
        plt.imshow(Images[I[i]])
        plt.subplot(2, 3, 2)
        plt.title('GroundTruth')
        plt.imshow(GT[I[i]])
        plt.subplot(2, 3, 3)
        plt.title('Unet Image')
        plt.imshow(UNET[I[i]])
        plt.subplot(2, 3, 4)
        plt.title('Unet3+ Image')
        plt.imshow(RESUNET[I[i]])
        plt.subplot(2, 3, 5)
        plt.title('Proposed')
        plt.imshow(PROPOSED[I[i]])
        plt.tight_layout()
        plt.show()


def Sample_Images():
    Orig = np.load('Image.npy', allow_pickle=True)
    label = np.load('Target.npy', allow_pickle=True)
    Classes = ['Apple', 'Cherry', 'Citrus', 'Corn', 'Grape']
    if label.shape[1] > 5:
        label = label[:, :5]
    for i in range(label.shape[1]):
        tar = label[:, i]
        ind1 = np.where(tar == 1)[0]
        image = Orig[ind1]
        ind = [1, 2, 4, 5, 6, 7]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle(Classes[i] + " Sample Images")
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(image[ind[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(image[ind[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(image[ind[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(image[ind[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(image[ind[4]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(image[ind[5]])
        plt.show()


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
