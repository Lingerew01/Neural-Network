import pandas as pd
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses=None):

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid()
    plt.show()


def model_plot(depth, data):
    data = pd.read_csv("data.csv")
    thick = data.iloc[7:13, :]
    init_depth = pd.DataFrame([[0] * data.shape[1]], columns=data.columns)
    depth = pd.concat([init_depth, thick.cumsum()], ignore_index=True)

    for col in data.columns:
        plt.figure(figsize =(4, 6))
        depth_values = depth[col].values
        resistivity_values = data.loc[0:6, col].values
        plt.plot(resistivity_values,-depth_values,drawstyle="steps-mid", label = "inverted", linestyle = "--", color = 'r')
        plt.xlabel("Resistivity")
        plt.ylabel("Depth")
        plt.grid(True,linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        plt.show()









