# %%
import streamlit as st
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from word_splitter import Preprocessor
from word_embedder import WordEmbedder
import lightning as L
from numpy import argmax
from matplotlib import pyplot as plt


st.set_page_config(
    page_title="Large Langstrumpf Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title("Large Langstrumpf Model")

# Data Loading and Preprocessing
training_text = "Pipilotta Viktualia Pfefferminza Rollgardina Efraimstochter Langstrumpf EOF"
training_text = st.sidebar.text_area("Gib einen Trainingstext ein", training_text)
pp = Preprocessor()
pp.fit(training_text)
X, y = pp.make_data(training_text)

epochs = st.sidebar.number_input("Anzahl Epochen", 0, 100, 50)

vocabulary_size = pp.vocabulary_size

# Model Loading
data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
loader = DataLoader(data)
model = WordEmbedder(vocabulary_size=vocabulary_size)
trainer = L.Trainer(max_epochs=epochs)
trainer.fit(model, train_dataloaders=loader)

# Visualization
word = st.sidebar.selectbox("Word eingeben", training_text.split())
word = pp.transform(word)

predictions = list(model.predict(torch.tensor(word, dtype=torch.float32)).detach().numpy())[0]
input_x = [1]*vocabulary_size
input_y = list(range(1, vocabulary_size+1))
hidden_x = [2, 2]
hidden_y = [vocabulary_size/2, vocabulary_size/2+1]
output_x = [3]*vocabulary_size
output_y = range(1, vocabulary_size+1)

w1 = model.layer_01.weight.detach().numpy()
w1 = ((w1-w1.min()) / (w1.max()-w1.min()))*3

w2 = model.layer_02.weight.detach().numpy()
w2 = ((w2-w2.min()) / (w2.max()-w2.min()))*3

embeddings = model.embedd(torch.tensor(word, dtype=torch.float32)).detach().numpy()[0]
embeddings = (embeddings-min(embeddings))/(max(embeddings)-min(embeddings))

MARKER_SIZE = 24
MARKER_COLOR = "black"
EDGE_COLOR = "black"
EDGE_WIDTH = 1


fig, ax = plt.subplots()
ax.axis(False)

# Connection Input-Hidden
for i in range(len(input_x)):
    for j in range(len(hidden_x)):
        ax.plot([input_x[i], hidden_x[j]], [input_y[i], hidden_y[j]], color=EDGE_COLOR, linewidth=w1[j,i])

# Connections Hidden-Output
for i in range(len(output_x)):
    for j in range(len(hidden_x)):
        ax.plot([output_x[i], hidden_x[j]], [output_y[i], hidden_y[j]], color=EDGE_COLOR, linewidth=w2[i,j])

# Nodes
#ax.plot(input_x, input_y, "o", color=MARKER_COLOR, markersize=MARKER_SIZE)
#ax.plot(hidden_x, hidden_y, "o", color=MARKER_COLOR, markersize=MARKER_SIZE)
#ax.plot(output_x, output_y, "o", color=MARKER_COLOR, markersize=MARKER_SIZE)

for i in range(len(hidden_x)):
    ax.plot(hidden_x[i], hidden_y[i], "o", color=MARKER_COLOR, markersize=MARKER_SIZE, alpha=min(embeddings[i]+0.1, 1))

for i in range(len(input_x)):
    ax.plot(input_x[i], input_y[i], "o", color=MARKER_COLOR, markersize=MARKER_SIZE, alpha=min(1, word[0][i]+0.1))

for i in range(len(output_x)):
    ax.plot(output_x[i], output_y[i], "o", color=MARKER_COLOR, markersize=MARKER_SIZE, alpha=min(1, predictions[i]+0.1))

for i, training_text in enumerate(list(pp.encoder.categories_[0])[::-1]):
    ax.text(x = 0.4, y=vocabulary_size-i, s=training_text)

for i, training_text in enumerate(list(pp.encoder.categories_[0])[::-1]):
    ax.text(x = 3.2, y=vocabulary_size-i, s=training_text)

for i, value in enumerate(word[0]):
    ax.text(x=0, y=i+1, s=value)

for i, value in enumerate(predictions):
    ax.text(x=4, y=i+1, s=round(value, 2))

st.pyplot(fig)



