import streamlit as st
import torch
import lightning as L
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from word_splitter import Preprocessor
from word_embedder import WordEmbedder
from loss_logger import LossHistory
from pandas import DataFrame
from string import punctuation


MARKER_SIZE = 24
MARKER_COLOR = "black"
EDGE_COLOR = "black"
EDGE_WIDTH = 1
HIDDEN_SIZE = 2
N_LAYERS = 2


def train(epochs):
    data = TensorDataset(
        torch.tensor(st.session_state["X"], dtype=torch.float32),
        torch.tensor(st.session_state["y"], dtype=torch.float32)
    )
    loader = DataLoader(data, batch_size=st.session_state["vocabulary_size"])
    loss_history = LossHistory()
    model = WordEmbedder(vocabulary_size=st.session_state["vocabulary_size"])
    trainer = L.Trainer(max_epochs=epochs, callbacks=[loss_history])
    trainer.fit(model, train_dataloaders=loader)
    loss = loss_history.train_losses
    st.session_state["w1"] = model.layer_01.weight.detach().numpy()
    st.session_state["w2"] = model.layer_02.weight.detach().numpy()
    st.session_state["model"] = model
    st.session_state["loss"] = loss


def preprocess(training_text):
    pp = Preprocessor()
    pp.fit(training_text)
    st.session_state["pp"] = pp
    st.session_state["X"], st.session_state["y"] = pp.make_data(training_text)
    st.session_state["vocabulary_size"] = pp.vocabulary_size
    st.session_state["words"] = pp.encoder.categories_[0]


def clean(text):
    output_text = ""
    for char in text:
        if char not in punctuation:
            output_text += char
    return output_text


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
    "Pipilotta Viktualia Pfefferminza Rollgardina Efraimstochter Langstrumpf EOS",
    key="input_text"
)

epochs = st.sidebar.slider("Anzahl Epochen", 0, 100, 50)
temp = st.sidebar.slider("Temperatur", min_value=0.1, max_value=8.0, value=0.3, step=0.1)

if "pp" not in st.session_state:
    preprocess(training_text)

if 'model' not in st.session_state:
    train(epochs)

if st.sidebar.button("Train Model"):
    preprocess(training_text)
    train(epochs)

# Loss Plot
st.sidebar.header("Loss Plot")
fig, ax = plt.subplots()
ax.plot(st.session_state["loss"], label="Train Loss")
ax.set_xlabel("Iterationen")
ax.set_ylabel("Loss")
ax.set_title("Loss-Verlauf")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
st.sidebar.pyplot(fig)

# Page Content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Model Test", "🗃 Gewichte", "📈 Embeddings", "ℹ️ Erklärung", "🔬 Experimentideen"])

with tab2:

    st.header(f"Insgesamt hat unser Modell {st.session_state['vocabulary_size'] * HIDDEN_SIZE * N_LAYERS} Gewichte")

    st.subheader("Gewichte auf Ebene 1")
    df1 = DataFrame(st.session_state["w1"])
    df1.columns = st.session_state["words"]
    df1.index = ["Weights to H1", "Weights to H2"]
    st.write(df1)

    st.subheader("Gewichte auf Ebene 2")
    df2 = DataFrame(st.session_state["w2"].T)
    df2.columns = st.session_state["words"]
    df2.index = ["Weights from H1", "Weights from H2"]
    st.write(df2)

with tab1:
    # Get Test Input
    st.header("Modell Testen")
    word = st.selectbox("Welches Wort wollen wir durch das Modell schicken?", clean(training_text).split())
    word = st.session_state["pp"].transform(word)

    # Create Grid for visualization
    input_x = [1]*st.session_state["vocabulary_size"]
    input_y = list(range(1, st.session_state["vocabulary_size"]+1))
    hidden_x = [2, 2]
    hidden_y = [st.session_state["vocabulary_size"] / HIDDEN_SIZE, st.session_state["vocabulary_size"] / HIDDEN_SIZE + 1]
    output_x = [3]*st.session_state["vocabulary_size"]
    output_y = range(1, st.session_state["vocabulary_size"]+1)

    # Scale the Weights for plotting from 0 to 1
    # Multiply by 3 to get appropriate line thickness in plot
    w1 = ((st.session_state["w1"] - st.session_state["w1"].min()) / (st.session_state["w1"].max() - st.session_state["w1"].min())) * 3
    w2 = ((st.session_state["w2"] - st.session_state["w2"].min()) / (st.session_state["w2"].max() - st.session_state["w2"].min())) * 3

    # Calculate predictions with temperature scaling
    word_tensor = torch.tensor(word, dtype=torch.float32)
    with torch.no_grad():
        # Get raw logits (either via forward or directly from layers)
        if hasattr(st.session_state["model"], "forward"):
            logits = st.session_state["model"](word_tensor)
        else:
            # Fallback calculating manually if predict() applies Softmax internally
            emb = st.session_state["model"].embedd(word_tensor)
            logits = torch.matmul(emb, torch.tensor(st.session_state["w2"]))

        # Apply Softmax with Temperature
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)
        originals = list(logits.detach().numpy())[0]
        predictions = list(probs.detach().numpy())[0]

    # Determine the activation of the hidden layer
    # Scale the activation from 0 to 1 for plotting
    activation = st.session_state["model"].embedd(word_tensor).detach().numpy()[0]
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

    for i, t_text in enumerate(list(st.session_state["words"])[::-1]):
        ax.text(x=0.4, y=st.session_state["vocabulary_size"]-i, s=t_text)

    for i, t_text in enumerate(list(st.session_state["words"])[::-1]):
        ax.text(x=3.2, y=st.session_state["vocabulary_size"]-i, s=t_text)

    for i, value in enumerate(word[0]):
        ax.text(x=0, y=i+1, s=value)

    for i, value in enumerate(predictions):
        ax.text(x=4.5, y=i+1, s=round(value, 2))

    for i, value in enumerate(originals):
        ax.text(x=4, y=i+1, s=round(value, 2))

    ax.text(x=4, y=i+2, s="logit", weight='bold')
    ax.text(x=4.5, y=i+2, s="probability", weight='bold')
    ax.text(x=0, y=i+2, s="input", weight='bold')

    st.pyplot(fig)

with tab3:

    fig, ax = plt.subplots(figsize=(15,8))

    ax.scatter(x=w1[0,:], y=w1[1,:], color="k", label="word", marker="o", s=50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc=0)

    for i, w_text in enumerate(st.session_state["words"]):
        plt.text(x=w1[0,i]+0.02, y=w1[1,i]+0.02, s=w_text)

    ax.set_ylabel("weights to hidden 2")
    ax.set_xlabel("weights to hidden 1")
    ax.set_title("Visualization of Embeddings")
    st.pyplot(fig)

with tab4:
    st.markdown("""# 🤖 Kurzanleitung: Large Langstrumpf Model (LLM)

Der Demonstrator veranschaulicht interaktiv, wie ein einfaches Neurologisches Netz Wort-Embeddings lernt und Vorhersagen für den nächsten Token (das nächste Wort) trifft.

## 1. Modell konfigurieren & trainieren (Seitenleiste)

1. **Trainingstext eingeben:** Passe den Text im Feld *Trainingstext* an oder nutze den Standardsatz (z. B. den Namen von Pippi Langstrumpf).
2. **Epochen wählen:** Lege über den Schieberegler *Anzahl Epochen* fest, wie intensiv das Modell auf den Text trainiert wird.
3. **Temperatur einstellen:** Steuere über *Temperatur*, wie „spitz“ oder „flach“ die Wahrscheinlichkeitsverteilung bei der Vorhersage ausfällt (niedrige Werte verdeutlichen die Top-Vorhersage, höhere Werte verteilen die Wahrscheinlichkeiten gleichmäßiger).
4. **Modell neu starten:** Klicke auf **Train Model**, um das Wortinventar neu zu verarbeiten und das Netzwerk frisch zu trainieren.
5. **Loss-Verlauf beobachten:** Das Diagramm *Loss-Verlauf* zeigt direkt, wie schnell der Trainingsfehler mit den Iterationen sinkt.

## 2. Die Hauptansichten (Tabs)

### 📈 Model Test

* **Worteingabe:** Wähle über das Dropdown-Menü ein Wort aus dem Trainingstext aus.
* **Architektur-Visualisierung:**
* **Input Layer (links):** Zeigt den One-Hot-Vektor des gewählten Eingabeworts.
* **Hidden Layer (Mitte):** Stellt die zwei versteckten Neuronen dar; die Stärke der Aktivierung wird durch die Helligkeit der Knoten symbolisiert.
* **Verbindungen (Kanten):** Die Liniendicke entspricht den trainierten Gewichtungen.
* **Output Layer (rechts):** Zeigt die Rohwerte (**Logits**) sowie die durch die *Softmax-Funktion* (inkl. Temperatur) berechneten **Wahrscheinlichkeiten** für das nächste Wort.

### 🗃 Gewichte

* Zeigt die genauen Matrizen der gelernten Gewichte von Ebene 1 ($W_1$) und Ebene 2 ($W_2$).
* Gibt Aufschluss darüber, wie die Wörter im 2D-Hidden-Space gewichtet und transformiert werden.

### 📈 Embeddings

* Visualisiert die gelernten 2D-Worteinbettungen in einem Koordinatensystem.
* Wörter, die im Text in ähnlichen Kontexten stehen oder ähnliche Übergänge haben, rücken in dieser 2D-Projektion näher zusammen.""")

with tab5:
    st.markdown("""# 🧪 Experimente & Aufgaben für Nutzer

## Experiment 1: Der Einfluss der Epochen (Unterfitting vs. Konvergenz)

* **Aktion:** Setze die *Anzahl Epochen* zuerst auf **0** oder **5** und klicke auf **Train Model**. Beobachte das Verhalten im Tab **📈 Model Test** und **📈 Embeddings**. Erhöhe danach schrittweise auf **50** oder **100** Epochen.
* **Was man lernt:** Bei wenigen Epochen sind die Gewichte noch zufällig verteilt – das Modell „weiß“ noch nicht, welches Wort auf welches folgt, und der Loss ist hoch. Mit mehr Epochen sinkt der Loss, die Kanten im Netzwerk werden gezielt dicker/dünner und die Wort-Embeddings ordnen sich im 2D-Raum an.

## Experiment 2: Die Magie der Temperatur (Softmax-Scaling)

* **Aktion:** Wähle im Tab **📈 Model Test** ein Wort aus (z. B. *„Pfefferminza“*). Verändere nun den Schieberegler **Temperatur** von **0.1** schrittweise bis auf **8.0**, ohne das Modell neu zu trainieren.
* **Was man lernt:** Bei **niedriger Temperatur (0.1)** wird der Unterschied zwischen den Logits extrem verstärkt (*„Winner-takes-all“*): Die Wahrscheinlichkeit für das nachfolgende Wort geht nahezu auf 1.0 (100 %). Bei **hoher Temperatur (8.0)** werden die Logits geglättet: Das Modell wird „unsicherer“ und verteilt die Wahrscheinlichkeiten fast gleichmäßig über das gesamte Vokabular.

## Experiment 3: Wort-Verbindungen im 2D-Raum (Embeddings erforschen)

* **Aktion:** Trainiere das Modell mit dem Standardtext. Wechsel in den Tab **📈 Embeddings** und schaue dir die Positionen der Wörter an.
* **Was man lernt:** Das Modell nutzt genau **2 versteckte Neuronen** (`HIDDEN_SIZE = 2`), um das gesamte Vokabular in eine 2D-Ebene zu projizieren. Nutzer können direkt sehen, welche Wörter im Vektorraum nah beieinander liegen und wie die Eingabe über die Matrix $W_1$ auf die x- und y-Achse transformiert wird.

## Experiment 4: Eigene Sätze & Sequenzen testen

* **Aktion:** Ersetze den Trainingstext durch eigene, wiederkehrende Muster oder Reime, wie z. B.:
* `eins zwei drei eins zwei drei EOS`
* `ich liebe data science und ich liebe machine learning EOS`
Klicke auf **Train Model** und teste im Tab **📈 Model Test**, wie gut das Modell logische Nachfolger vorhersagt (z. B. was folgt auf *„eins“* oder *„liebe“*?).
* **Was man lernt:** Wie das Vokabular dynamsich neu erstellt wird und wie Mehrfachvorkommen von Wörtern die Vorhersagewahrscheinlichkeiten im Modell verändern.""")

st.sidebar.markdown(
    "<div style='text-align:center; color:#999; margin-top:60px;'>"
    "Made with ❤️ by <br> Michael Kohlegger (2026)"
    "</div>",
    unsafe_allow_html=True
)