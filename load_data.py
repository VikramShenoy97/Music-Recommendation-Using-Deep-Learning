import os
import re
import numpy as np
import cv2
from import_data import create_spectrogram
from slice_spectrogram import slice_spect
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

"""
Converts images and labels into training and testing matrices.
"""
def load_dataset(verbose=0, mode=None, datasetSize=1.0):
    create_spectrogram(verbose, mode)
    slice_spect(verbose, mode)

    # datasetSize is a float value which returns a fraction of the dataset.
    # If set as 1.0 it returns the entire dataset.
    # If set as 0.5 it returns half the dataset.

    if mode=="Train":
        genre = {
        "Hip-Hop": 0,
        "International": 1,
        "Electronic": 2,
        "Folk" : 3,
        "Experimental": 4,
        "Rock": 5,
        "Pop": 6,
        "Instrumental": 7
        }
        if(verbose > 0):
            print "Compiling Training and Testing Sets ..."
        filenames = [os.path.join("Train_Sliced_Images", f) for f in os.listdir("Train_Sliced_Images")
                       if f.endswith(".jpg")]
        images_all = [None]*(len(filenames))
        labels_all = [None]*(len(filenames))
        for f in filenames:
            index = int(re.search('Train_Sliced_Images/(.+?)_.*.jpg', f).group(1))
            genre_variable = re.search('Train_Sliced_Images/.*_(.+?).jpg', f).group(1)
            temp = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images_all[index] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            labels_all[index] = genre[genre_variable]

        if(datasetSize == 1.0):
            images = images_all
            labels = labels_all

        else:
            count_max = int(len(images_all)*datasetSize / 8.0)
            count_array = [0, 0, 0, 0, 0 ,0, 0, 0]
            images = []
            labels = []
            for i in range(0, len(images_all)):
                if(count_array[labels_all[i]] < count_max):
                    images.append(images_all[i])
                    labels.append(labels_all[i])
                    count_array[labels_all[i]] += 1
            images = np.array(images)
            labels = np.array(labels)

        images = np.array(images)
        labels = np.array(labels)
        labels = labels.reshape(labels.shape[0],1)
        train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.05, shuffle=True)

        # Convert the labels into one-hot vectors.
        train_y = np_utils.to_categorical(train_y)
        test_y = np_utils.to_categorical(test_y, num_classes=8)
        n_classes = len(genre)
        genre_new = {value: key for key, value in genre.items()}

        if os.path.exists('Training_Data'):
            train_x = np.load("Training_Data/train_x.npy")
            train_y = np.load("Training_Data/train_y.npy")
            test_x = np.load("Training_Data/test_x.npy")
            test_y = np.load("Training_Data/test_y.npy")
            return train_x, train_y, test_x, test_y, n_classes, genre_new

        if not os.path.exists('Training_Data'):
            os.makedirs('Training_Data')
        np.save("Training_Data/train_x.npy", train_x)
        np.save("Training_Data/train_y.npy", train_y)
        np.save("Training_Data/test_x.npy", test_x)
        np.save("Training_Data/test_y.npy", test_y)
        return train_x, train_y, test_x, test_y, n_classes, genre_new

    if mode=="Test":
        if(verbose > 0):
            print "Compiling Training and Testing Sets ..."
        filenames = [os.path.join("Test_Sliced_Images", f) for f in os.listdir("Test_Sliced_Images")
                       if f.endswith(".jpg")]
        images = []
        labels = []
        for f in filenames:
            song_variable = re.search('Test_Sliced_Images/.*_(.+?).jpg', f).group(1)
            tempImg = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images.append(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
            labels.append(song_variable)

        images = np.array(images)

        return images, labels
