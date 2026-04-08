import os
from skimage.feature import graycomatrix
import cv2
import pandas as pd
from skimage.measure import regionprops, label
from numpy import matlib
from ECO import ECO
from Global_Vars import Global_Vars
from Image_Results import *
from LBP import lbp_calculated_pixel
from Model_Bi_GRU import Model_Bi_GRU
from Model_CNN import Model_CNN
from Model_Feat_ViT import Model_Feat_ViT
from Model_RAN import Model_RAN
from Model_ResNet import Model_ResNet
from Model_SPN import Model_SPN
from Model_Sparse_Attention_ResNet import Model_Sparse_attention_ResNet
from Model_VGG16 import Model_VGG16
from Objfun import objfun
from Plot_results import *
from GLCM import contrast_feature, homogeneity_feature, energy_feature, correlation_feature, dissimilarity_feature
from Proposed import Proposed
from QOS import QSO
from SAA import SAA
from SCO import SCO


def ReadImage(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (256, 256))
    return image


# Read Dataset
an = 0
if an == 1:
    Image = []
    Target = []
    path = './Dataset/PlantDiseasesDataset'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir) - 1):
            sub_foder = folder + '/' + in_dir[j]
            dir = os.listdir(sub_foder)
            for k in range(len(dir)):
                file = sub_foder + '/' + dir[k]
                sub_dir = os.listdir(file)
                for m in range(len(sub_dir)):
                    Target.append(dir[k])
                    FileName = file + '/' + sub_dir[m]
                    Img = ReadImage(FileName)
                    Image.append(Img)

    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index.npy', index)
    np.save('Image.npy', Shuffled_Datas)
    np.save('Target.npy', Shuffled_Target)

# Generate Ground_Truth Dataset
an = 0
if an == 1:
    Images = np.load('Image.npy', allow_pickle=True)
    GT = []
    for i in range(len(Images)):
        print(i)
        image = Images[i]
        img = np.zeros(image.shape, dtype=np.uint8)
        max_val = np.max(image)
        thresh = max_val - (max_val * 0.2)
        index = np.where(image >= thresh)
        img[index[0], index[1]] = 255
        img = img.astype(np.uint8)
        GT.append(img)
    np.save('Ground_Truth.npy', GT)

# Optimization for Segmentation
an = 0
if an == 1:
    Data = np.load('Image.npy', allow_pickle=True)
    GT = np.load('Ground_Truth.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.GT = GT
    Npop = 10
    Chlen = 3  # Hidden Neuron, Epoch, Step per Epochs
    xmin = matlib.repmat([5, 5, 300], Npop, 1)
    xmax = matlib.repmat([255, 50, 1000], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun
    Max_iter = 50

    print("SCO...")
    [bestfit1, fitness1, bestsol1, time1] = SCO(initsol, fname, xmin, xmax, Max_iter)  # SCO

    print("SAA...")
    [bestfit4, fitness4, bestsol4, time4] = SAA(initsol, fname, xmin, xmax, Max_iter)  # SAA

    print("ECO...")
    [bestfit2, fitness2, bestsol2, time2] = ECO(initsol, fname, xmin, xmax, Max_iter)  # ECO

    print("QSO...")
    [bestfit3, fitness3, bestsol3, time3] = QSO(initsol, fname, xmin, xmax, Max_iter)  # QSO

    print("Proposed")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Best_Sol.npy', BestSol)

# APSPN Segmentation
an = 0
if an == 1:
    Image = np.load('Image.npy', allow_pickle=True)
    Gt = np.load('Ground_Truth.npy', allow_pickle=True)
    sol = np.load('Best_Sol.npy', allow_pickle=True)
    Eval, Images = Model_SPN(Image, Gt, sol)
    np.save('Eval_all1.npy', Eval)
    np.save('Seg_UNet_GAN.npy', Images)

# Feature Extraction Local Binary pattern
an = 0
if an == 1:
    image = np.load('Seg_APSPN.npy', allow_pickle=True)
    LBP = []
    for i in range(len(image)):
        print(i)
        image1 = image[i]
        height, width = image1.shape
        img_lbp = np.zeros((height, width),
                           np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(image1, i, j)
        LBP.append(img_lbp)
    np.save("LBP.npy", np.asarray(LBP))

# GLCM
an = 0
if an == 1:
    Prep_Images = np.load('Seg_APSPN.npy', allow_pickle=True)
    matrix_coocurrence = []
    GLCM_Data = []
    GLCM = []
    for i in range(len(Prep_Images)):
        print(i)
        image = Prep_Images[i]
        # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = graycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                          normed=False, symmetric=False)
        A = (contrast_feature(matrix_coocurrence))
        B = (dissimilarity_feature(matrix_coocurrence))
        C = (homogeneity_feature(matrix_coocurrence))
        D = (energy_feature(matrix_coocurrence))
        E = (correlation_feature(matrix_coocurrence))
        GLCM_Data = np.append(E, np.append(D, np.append(C, np.append(B, A))))
        GLCM.append(GLCM_Data)
    np.save("GLCM.npy", np.asarray(GLCM))

# Pattern Concatenation
an = 0
if an == 1:
    LBP = np.load('LBP.npy', allow_pickle=True)
    GLCM = np.load('GLCM.npy', allow_pickle=True)
    Pattern = np.concatenate([LBP.reshape(LBP.shape[0], -1), GLCM], axis=1)
    np.save('Texture.npy', Pattern)

# Feature Extraction using Geometry
an = 0
if an == 1:
    image = np.load('Seg_APSPN.npy', allow_pickle=True)
    features = []
    for n in range(image.shape[0]):
        img = image[n]
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        labeled_image = label(binary)
        regions = regionprops(labeled_image)
        for i, region in enumerate(regions):
            print(n, i)
            Area = region.area
            Perimeter = region.perimeter
            Eccentricity = region.eccentricity
            Extent = region.extent
            Solidity = region.solidity
            MajorAxisLength = region.major_axis_length
            MinorAxisLength = region.minor_axis_length
            Orientation = region.orientation
            Centroid_X = region.centroid[1]
            Centroid_Y = region.centroid[0]
            EquivalentDiameter = region.equivalent_diameter

            Feat = [Area, Perimeter, np.float64(Eccentricity), Extent, Solidity, MajorAxisLength, MinorAxisLength, Orientation,
                    Centroid_X, Centroid_Y, EquivalentDiameter]
        features.append(np.asarray(Feat))
    np.save('Geometry_Feat.npy', np.asarray(features))

# Feature Extraction using ViT
an = 0
if an == 1:
    Image = np.load('Seg_APSPN.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Feat = Model_Feat_ViT(Image, Target)
    np.save('ViT_Feat.npy', Feat)

# Feature Fusion
an = 0
if an == 1:
    Set_1 = np.load('Texture.npy', allow_pickle=True)
    Set_2 = np.load('Geometry_Feat.npy', allow_pickle=True)
    Set_3 = np.load('ViT_Feat.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Feature = Model_Sparse_attention_ResNet(Set_1, Set_2, Set_3, Target)
    np.save('Feature_Fusion.npy', Feature)

# Classification
an = 0
if an == 1:
    Feature = np.load('Feature_Fusion.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    EVAL = []
    Epochs = [20, 40, 60, 80]
    for i in range(len(Epochs)):
        learnperc = round(Feature.shape[0] * 0.75)
        Train_Data = Feature[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feature[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((5, 25))
        Eval[0, :], pred = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[i])  # Model CNN
        Eval[1, :], pred1 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[i])  # Model VGG16
        Eval[2, :], pred2 = Model_ResNet(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[i])  # Model ResNet
        Eval[3, :], pred3 = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[i])  # Model RAN
        Eval[4, :], pred4 = Model_Bi_GRU(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[i])  # Model BiGRU
        EVAL.append(Eval)
    np.save('Evaluate_all.npy', EVAL)  # Save Eval all

plotConvResults()
Plots_Results()
Plot_ROC_Curve()
plot_seg_results()
Table()
Proposed_PlotsResults()
Image_Results()
Sample_Images()
