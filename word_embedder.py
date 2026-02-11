
from torch.nn import Linear, CrossEntropyLoss, Softmax
from torch.optim import Adam
from lightning import LightningModule
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


class WordEmbedder(LightningModule):

    def __init__(self, vocabulary_size=5):
        super().__init__()

        self.layer_01 = Linear(
            in_features=vocabulary_size,
            out_features=2,
            bias=False
        )

        self.layer_02 = Linear(
            in_features=2,
            out_features=vocabulary_size,
            bias=False
        )

        self.loss_function = CrossEntropyLoss()

    def forward(self, input):
        hidden = self.layer_01(input)
        output = self.layer_02(hidden)
        return output

    def predict(self, input):
        logits = self.forward(input)
        return F.softmax(logits, dim=1)

    def embedd(self, input):
        return self.layer_01(input)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, target_i = batch
        output_i = self.forward(input_i)
        loss = self.loss_function(output_i, target_i)
        return loss

    def get_weights(self):
        return DataFrame({
            "input_no": [1, 2, 3, 4, 5],
            "token": ["Michael", "ist", "toll", "EOS", "Christian"],
            "w1": self.layer_01.weight.detach()[0].numpy(),
            "w2": self.layer_01.weight.detach()[1].numpy(),
        })

    def plot_weights(self):
        sns.scatterplot(self.get_weights(), x="w1", y="w2")
        for row in range(5):
            plt.text(
                x=self.get_weights()["w1"].values[row]+0.01,
                y=self.get_weights()["w2"].values[row]+0.01,
                s=self.get_weights()["token"].values[row],
                size="medium",
                color="black",
                weight="semibold"
            )