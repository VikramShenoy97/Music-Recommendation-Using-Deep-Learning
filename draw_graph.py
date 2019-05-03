import os
import pandas as pd
import math
from PIL import Image
import numpy as np
import random
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from load_data import load_dataset
from sklearn.metrics import confusion_matrix
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

model = load_model("Saved_Model/Model.h5")
model.set_weights(model.get_weights())
train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(verbose=1, mode="Train", datasetSize=0.75)

train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

train_x = train_x / 255.
test_x = test_x / 255.

predictions = model.predict(test_x, verbose=0)

train_x = train_x * 255.
test_x = test_x * 255.
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2])
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2])
train_x = train_x.astype(int)
test_x = test_x.astype(int)

sample_indices = random.sample(range(test_x.shape[0]), 10)
sample_images = [test_x[i] for i in sample_indices]
sample_labels = [np.argmax(test_y[i]) for i in sample_indices]
sample_labels_softmax = [predictions[i] for i in sample_indices]
predicted = [np.argmax(predictions[i]) for i in sample_indices]

fig = plt.figure(figsize=(15, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    truth_genre = genre[truth]
    prediction = predicted[i]
    prediction_genre = genre[prediction]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(180, 70, "Truth: {0}\nPrediction: {1}".format(truth_genre, prediction_genre),
             fontsize=12, color=color)
    plt.imshow(sample_images[i])
plt.savefig("Graphs/Training_Prediction.jpg", cmap='gray_r')
plt.show()
plt.close()

filename = "Saved_Model/training_history.csv"
history = pd.read_csv(filename, header=0, low_memory=False)
history_array = history.values
epochs = history_array[:, 0]
training_accuracy = history_array[:, 1]
training_loss = history_array[:, 2]
val_accuracy = history_array[:, 3]
val_loss = history_array[:, 4]

py.sign_in('VikramShenoy','x1Un4yD3HDRT838vRkFA')


trace0 = go.Scatter(
x = epochs,
y = training_accuracy,
mode = "lines",
name = "Training Accuracy"
)

trace1 = go.Scatter(
x = epochs,
y = val_accuracy,
mode = "lines",
name = "Validation Accuracy"
)
data = go.Data([trace0, trace1])
layout = go.Layout()
fig = go.Figure(data=data, layout=layout)
fig['layout']['xaxis'].update(title="Number of Epochs", range = [min(epochs), max(epochs)], dtick=len(epochs)/10, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
fig['layout']['yaxis'].update(title="Accuracy", range = [0, 1], dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',linewidth=2)
py.image.save_as(fig, filename="Graphs/Accuracy_Graph.png")

print "Accuracy Graph Created"


trace0 = go.Scatter(
x = epochs,
y = training_loss,
mode = "lines",
name = "Training Loss"
)

trace1 = go.Scatter(
x = epochs,
y = val_loss,
mode = "lines",
name = "Validation Loss"
)
data = go.Data([trace0, trace1])
layout = go.Layout()
fig = go.Figure(data=data, layout=layout)
fig['layout']['xaxis'].update(title="Number of Epochs", range = [min(epochs), max(epochs)], dtick=len(epochs)/10, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
fig['layout']['yaxis'].update(title="Loss", dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',linewidth=2)
py.image.save_as(fig, filename="Graphs/Loss_Graph.png")
print "Loss Graph Created"

y_pred = np.argmax(predictions, axis=1)
y_test = np.argmax(test_y, axis=1)
confusion_matrix = confusion_matrix(y_test, y_pred)
labels = ["Hip-Hop", "International", "Electronic", "Folk", "Experimental", "Rock", "Pop", "Instrumental"]

trace = go.Heatmap(z=confusion_matrix, x=labels, y=labels, reversescale=False, colorscale='Viridis')
data=[trace]
layout = go.Layout(
title='Confusion Matrix',
width = 800, height = 800,
showlegend = True,
xaxis = dict(dtick=1, tickangle=45),
yaxis = dict(dtick=1, tickangle=45))
fig = go.Figure(data=data, layout=layout)
py.image.save_as(fig, filename="Graphs/Confusion_Matrix.png")
print "Confusion Matrix Created"
