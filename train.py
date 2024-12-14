import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Train:
    def __init__(self, model, x_train, y_train, x_val=None, y_val=None):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.train_losses = []
        self.val_losses = []

    def train(self, epochs=100, learning_rate=0.01, optimizer="Adam", loss="mse", verbose=True):
        for epoch in range(epochs):
            output = self.model.forward(self.x_train)
            train_loss = self.model.backpropagation(self.y_train, learning_rate, optimizer, loss)
            self.train_losses.append(train_loss)

            if self.x_val is not None and self.y_val is not None:
                val_output = self.model.forward(self.x_val)
                val_loss, _ = self.model.compute_loss(self.y_val, val_output, loss)
                self.val_losses.append(val_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}", end="")
                if self.x_val is not None:
                    print(f", Val Loss: {val_loss:.4f}")
                else:
                    print("")

    def plot_losses(self, save_path=None):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label="Training Loss", color="blue", linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label="Test Loss", color="m", linestyle="--", linewidth=2)

        plt.xlabel("Epochs", fontsize=14, fontweight='bold')
        plt.ylabel("Loss", fontsize=14, fontweight='bold')
        # plt.title("Training and Validation Loss", fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Ensuring only integers on the x-axis
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend(fontsize=12)
        plt.tight_layout()


        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')  # High DPI for publication
            print(f"Plot saved as {save_path}")

        plt.show()

    def predict(self, x):
        return self.model.forward(x)
