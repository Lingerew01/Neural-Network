import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from Data_loader import Data_Loader
from NN import Net
from train import Train

data_loader = Data_Loader("Total_VES_RG.csv",
                          "models_option.csv",
                          "responses_option.csv")
data_loader.read_data()
x_obs, y_model, x_response = data_loader.prepare_data()

x_obs = x_obs.iloc[0:17, :].to_numpy().T
x = x_response.iloc[0:17, :].to_numpy().T
y_model = y_model.T

print(x_obs.shape, x.shape, y_model.shape)

# Log transform
x_obs_log = np.log(x_obs)
x_log = np.log(x)
y_model_log = np.log(y_model)

# Normalize data
scalar = Normalizer()
x_obs_scaled = scalar.fit_transform(x_obs_log)
x_scaled = scalar.fit_transform(x_log)
y_scaled = scalar.fit_transform(y_model_log)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled,
                                   random_state=40,
                                   test_size=0.30,
                                   shuffle=True)

# Define the network
layer_sizes = [x_train.shape[1], 26, 64, 26, y_train.shape[1]]
np.random.seed(40)
model = Net(layer_sizes, activation="sigmoid")

# Train the network
trainer = Train(model, x_train, y_train)
trainer.train(epochs=15, learning_rate=0.001, optimizer="Adam")

# Plot training and validation losses
trainer.plot_losses(save_path="publication_loss_plot.png")

# Make predictions
predictions = trainer.predict(x_train)

print(f"Shape of predictions: {predictions.shape}")
num_features = predictions.shape[0]  # Adjust based on the actual number of features
sss = pd.DataFrame(predictions.T, columns=[f"Feature_{i}" for i in range(1, num_features + 1)])
ttt = pd.DataFrame(y_train.T, columns=[f"Feature_{i}" for i in range(1, num_features + 1)])
sss.to_csv("NN_out_original.csv", index=False)
ttt.to_csv("Y_train.csv", index=False)



















