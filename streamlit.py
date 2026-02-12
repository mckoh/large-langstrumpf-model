# %%
import streamlit as st
import torch
import lightning as L
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from word_splitter import Preprocessor
from word_embedder import WordEmbedder
from matplotlib import pyplot as plt
from loss_logger import LossHistory
from pandas import DataFrame


MARKER_SIZE = 24
MARKER_COLOR = "black"
EDGE_COLOR = "black"
EDGE_WIDTH = 1
HIDDEN_SIZE = 2
N_LAYERS = 2

st.set_page_config(
    page_title="Large Langstrumpf Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Large Langstrumpf Model")

st.sidebar.header("Modell Einstellungen")
# Data Loading and Preprocessing
training_text = st.sidebar.text_area(
    "Trainingstext",
    "Pipilotta Viktualia Pfefferminza Rollgardina Efraimstochter Langstrumpf"
)

if "text" not in st.session_state:
    st.session_state["text"] = training_text
else:
    training_text = st.session_state["text"]

pp = Preprocessor()
pp.fit(training_text)
X, y = pp.make_data(training_text)

epochs = st.sidebar.slider("Anzahl Epochen", 0, 100, 50)

vocabulary_size = pp.vocabulary_size
data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
loader = DataLoader(data, batch_size=vocabulary_size)
loss_history = LossHistory()

# Model Loading
def train():
    model = WordEmbedder(vocabulary_size=vocabulary_size)
    trainer = L.Trainer(max_epochs=epochs, callbacks=[loss_history])
    trainer.fit(model, train_dataloaders=loader)
    return model

if 'model' not in st.session_state:
    model = train()
    st.session_state["model"] = model
    loss = loss_history.train_losses
    st.session_state["loss"] = loss
else:
    model = st.session_state["model"]
    loss = st.session_state["loss"]

# Loss Plot
st.sidebar.header("Loss Plot")
fig, ax = plt.subplots()
ax.plot(loss, label="Train Loss")
ax.set_xlabel("Iterationen")
ax.set_ylabel("Loss")
ax.set_title("Loss-Verlauf")
st.sidebar.pyplot(fig)

if st.sidebar.button("Train Model"):
    model = train()
    st.session_state["model"] = model
    loss = loss_history.train_losses
    st.session_state["loss"] = loss

# Get Model weights after Training
w1 = model.layer_01.weight.detach().numpy()
w2 = model.layer_02.weight.detach().numpy()

# Page Content
tab1, tab2 = st.tabs(["📈 Model Test", "🗃 Gewichte"])

with tab2:

    st.header(f"Insgesamt hat unser Modell {vocabulary_size * HIDDEN_SIZE * N_LAYERS} Gewichte")

    st.subheader("Gewichte auf Ebene 1")
    df1 = DataFrame(w1)
    df1.columns = pp.encoder.categories_
    df1.index = ["Weights to H1", "Weights to H2"]
    st.write(df1)

    st.subheader("Gewichte auf Ebene 2")
    df2 = DataFrame(w2.T)
    df2.columns = pp.encoder.categories_
    df2.index = ["Weights from H1", "Weights from H2"]
    st.write(df2)

with tab1:
    # Get Test Input
    st.header("Modell Testen")
    word = st.selectbox("Welches Wort wollen wir durch das Modell schicken?", training_text.split())
    word = pp.transform(word)

    # Create Grid for visualization
    input_x = [1]*vocabulary_size
    input_y = list(range(1, vocabulary_size+1))
    hidden_x = [2, 2]
    hidden_y = [vocabulary_size / HIDDEN_SIZE, vocabulary_size / HIDDEN_SIZE + 1]
    output_x = [3]*vocabulary_size
    output_y = range(1, vocabulary_size+1)

    # Scale the Weights for plotting from 0 to 1
    # Multiply by 3 to get appropriate line thickness in plot
    w1 = ((w1-w1.min()) / (w1.max()-w1.min())) * 3
    w2 = ((w2-w2.min()) / (w2.max()-w2.min())) * 3

    # Get Predictions for visualization (softmax will already scale them
    # from 0 to 1)
    predictions = list(model.predict(torch.tensor(word, dtype=torch.float32)).detach().numpy())[0]

    # Determine the activation of the hidden layer
    # Scale the activation from 0 to 1 for plotting
    activation = model.embedd(torch.tensor(word, dtype=torch.float32)).detach().numpy()[0]
    activation = (activation-min(activation))/(max(activation)-min(activation))

    # Start the plot
    fig, ax = plt.subplots()
    ax.axis(False)

    # Draw the edges (input-hidden)
    for i in range(len(input_x)):
        for j in range(len(hidden_x)):
            ax.plot(
                [input_x[i], hidden_x[j]],
                [input_y[i], hidden_y[j]],
                color=EDGE_COLOR,
                linewidth=w1[j,i]
            )

    # Draw the edges (hidden-output)
    for i in range(len(output_x)):
        for j in range(len(hidden_x)):
            ax.plot(
                [output_x[i], hidden_x[j]],
                [output_y[i], hidden_y[j]],
                color=EDGE_COLOR,
                linewidth=w2[i,j]
            )

    # Plot a white dot for every neuron to hide the edges
    ax.plot(input_x, input_y, "o", color="white", markersize=MARKER_SIZE+4)
    ax.plot(hidden_x, hidden_y, "o", color="white", markersize=MARKER_SIZE+4)
    ax.plot(output_x, output_y, "o", color="white", markersize=MARKER_SIZE+4)

    # Plot the actual neurons and make them darker/lighter based on activation
    for i in range(len(hidden_x)):
        ax.plot(hidden_x[i], hidden_y[i], "o", color=MARKER_COLOR, markersize=MARKER_SIZE, alpha=min(activation[i]+0.1, 1))

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


st.sidebar.markdown(
    "<div style='text-align:center; color:#999; margin-top:60px;'>"
    "Made with ❤️ by </BR> Michael Kohlegger (2026)"
    "</div>",
    unsafe_allow_html=True
)
