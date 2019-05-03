import os
import re
from PIL import Image

"""
Slice the spectrogram into multiple 128x128 images which will be the input to the
Convolutional Neural Network.
"""
def slice_spect(verbose=0, mode=None):
    if mode=="Train":
        if os.path.exists('Train_Sliced_Images'):
            return
        labels = []
        image_folder = "Train_Spectogram_Images"
        filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                       if f.endswith(".jpg")]
        counter = 0
        if(verbose > 0):
            print "Slicing Spectograms ..."
        if not os.path.exists('Train_Sliced_Images'):
            os.makedirs('Train_Sliced_Images')
        for f in filenames:
            genre_variable = re.search('Train_Spectogram_Images/.*_(.+?).jpg', f).group(1)
            img = Image.open(f)
            subsample_size = 128
            width, height = img.size
            number_of_samples = width / subsample_size
            for i in range(number_of_samples):
                start = i*subsample_size
                img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))
                img_temporary.save("Train_Sliced_Images/"+str(counter)+"_"+genre_variable+".jpg")
                counter = counter + 1
        return

    elif mode=="Test":
        if os.path.exists('Test_Sliced_Images'):
            return
        labels = []
        image_folder = "Test_Spectogram_Images"
        filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                       if f.endswith(".jpg")]
        counter = 0
        if(verbose > 0):
            print "Slicing Spectograms ..."
        if not os.path.exists('Test_Sliced_Images'):
            os.makedirs('Test_Sliced_Images')
        for f in filenames:
            song_variable = re.search('Test_Spectogram_Images/(.+?).jpg', f).group(1)
            img = Image.open(f)
            subsample_size = 128
            width, height = img.size
            number_of_samples = width / subsample_size
            for i in range(number_of_samples):
                start = i*subsample_size
                img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))
                img_temporary.save("Test_Sliced_Images/"+str(counter)+"_"+song_variable+".jpg")
                counter = counter + 1
        return
