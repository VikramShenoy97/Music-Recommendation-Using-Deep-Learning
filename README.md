# Music Recommendation Using Deep Learning

Music Recommendation using latent feature vectors obtained from a network trained on the Free Music Archive dataset.

## Overview

The basic idea of this project is to recommend music using computer vision through a convolutional neural network. The network is first trained as a classifier with the labels being the 8 different genres of songs from the dataset. The trained network is then modified by discarding the softmax layer i.e. creating a new model which works as an encoder. This encoder takes as input slices of a spectrogram one at a time and outputs a 32 dimensional latent representation of that respective slice. This generates multiple latent vectors for one spectrogram depending on how many slices were generated. These multiple vectors are then averaged to get one latent representation for each spectrogram. The Cosine similarity metric is used to generate a similarity score between one anchor song and the rest of the songs in the test set. The two songs with the highest similarity score with respect to the anchor song are then outputted as the recommendations.

![project_architecture](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Images/dlmusic.jpg)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

For using this project, you need to install Keras, Scikit-learn, PIL, Librosa, OpenCV, and Pandas

```
pip install keras
pip install scikit-learn
pip install pillow
pip install librosa
pip install cv2
pip install pandas
```

### Dataset
The fma_small dataset consists of 8000 mp3 files from the [Free Music Archive](https://github.com/mdeff/fma).

Each file in fma_small is  a 30 second clip of music. The dataset is balanced and has 8 genres ( Hip-Hop, International, Electronic, Folk, Experimental, Rock, Pop, and Instrumental).

The dataset is stored in the folder **Dataset** as *fma_small* *(File too large to upload onto Github)*.

For testing the recommendation system, I've used 30 songs from my itunes library. I've manually converted the songs into 30 second clips and then I ran my code in *test* mode.

```
30 Seconds To Mars - Night of the hunter (Acoustic)
Afrojack - The spark
Alesso - Heros
Awolnation - Sail
Boyce Avenue - Wonderwall
Bruno Mars - Just the way you are
Bruno Mars - Locked out of heaven
Calvin Harris - Summer
Calvin Harris - Sweet Nothing
Coldplay - Magic
Coldplay - Paradise
Coldplay - Viva La Vida
Coldplay - The Scientist
Daft Punk - Instant crush
Daft Punk - Lose yourself to dance
Don Omar - Danza Kuduro
Enrique Iglesias - Bailando
Imagine Dragons - Demons
Imagine Dragons - It's Time
Jennifer Lopez - On the floor 
John Mayer - Say
Kanye West - Stronger
Katy Perry - Dark Horse
Katy Perry - Fireworks
Khalid - Location
Lana Del Rey - Young and Beautiful
Maroon5 - Moves Like Jagger
Passenger - Let Her Go
Wiz Khalifa - Black and Yellow
Wiz Khalifa - Young, Wild and Free
```

## Training

Run the script *train.py* in the terminal as follows.
```
Python train.py
```

### Data Preprocessing

The *train.py* script runs **import_data.py**, **slice_spectrogram.py**, and **load_data.py** in the back.

### import_data.py
• **Train Mode** - In training mode, the script converts the files from *fma_small* into mel-spectrograms and stores them into a folder called *Train_Spectrogram_Images*.


• **Test Mode** - In testing mode, the script converts the songs from *DLMusicTest_30* into mel-spectrograms and stores them into a folder called *Test_Spectrogram_Images*.

### slice_spectrogram.py
• **Train Mode** - In training mode, the script slices the spectrograms from the *Train_Spectrogram_Images* folder into 128x128 slices and stores them into the *Train_Sliced_Images* folder.


• **Test Mode** - In testing mode, the script slices the spectrograms from the *Test_Spectrogram_Images* folder into 128x128 slices and stores them into the *Test_Sliced_Images* folder.

### load_data.py
• **Train Mode** - In training mode, the script imports images from *Train_Sliced_Images*, converts them into grayscale, and then exports them as numpy matrices for training and testing. This is saved as *train_x.npy*, *train_y.npy*, *test_x.npy*, and *test_y.npy* in the *Training_Data* folder.


• **Test Mode** - In testing mode, the script imports images from *Test_Sliced_Images*, converts them into grayscale, and returns them as images and labels.

### Neural Network Architecture

Convolutional Neural Network that is used for this recommendation system.
![model_architecture](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Saved_Model/Model_Architecture.jpg)

### Model and History

The trained network is then saved as *Model.h5* and it's history is saved as *training_history.csv* in the *Saved_Model* folder.

### Training Performance

```
Final Training Accuracy = 77.85%
Final Validation Accuracy = 66.11 %
```

### Prediction On Test Set 
(*This test set is a small part of fma_small dataset that hasn't been trained on*)
![training_prediction](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Graphs/Training_Prediction.jpg)

### Accuracy Graph
![accuracy_graph](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Graphs/Accuracy_Graph.png)

### Loss Graph
![loss_graph](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Graphs/Loss_Graph.png)

### Confusion Matrix
![confusion_matrix](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Graphs/Confusion_Matrix.png)


## Recommendation

### Testing

Run the script *recommendation.py* in the terminal as follows.
```
Python recommendation.py
```

This will give you a list of songs.
```
['Bailando' 'BlackandYellow' 'DanzaKuduro' 'DarkHorse' 'Demons'
'Fireworks' 'Heros' 'InstantCrush' 'ItsTime' 'JustTheWayYouAre'
'LetHerGo' 'Location' 'LockedOutOfHeaven' 'LoseYourselfToDance' 'Magic'
'MovesLikeJagger' 'NightOfTheHunter' 'OnTheFloor' 'Paradise' 'Sail' 'Say'
'Spark' 'Stronger' 'Summer' 'SweetNothing' 'VivaLaVida' 'Wonderwall'
'YoungAndBeautiful' 'YoungWildAndFree']
```

Enter an anchor song for which you want similar recommendations (Choose one from the above list).
```
Enter a Song Name:
TheScientist
```
### Results
The code generates two recommendations for the song The Scientist by Coldplay.

![Output_1](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Images/Output_1.png)

### More Results

![Output_2](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Images/Output_2.png)
![Output_3](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning/blob/master/Images/Output_3.png)


## Built With

* [Keras](https://keras.io) - Deep Learning Framework
* [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) - Cloud Service

## Authors

* **Vikram Shenoy** - *Initial work* - [Vikram Shenoy](https://github.com/VikramShenoy97)

## Acknowledgments

* Project is inspired by **Sander Dieleman's** blog, [*Recommending music on Spotify with Deep Learning.*](http://benanne.github.io/2014/08/05/spotify-cnns.html)
