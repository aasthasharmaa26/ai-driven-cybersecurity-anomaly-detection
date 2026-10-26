import torch
import torch.nn as nn

loss_function = nn.MSELoss()

def predict_autoencoder(model, vector):

    x = torch.tensor(

        vector,

        dtype=torch.float32

    )

    with torch.no_grad():

        reconstructed = model(x)

    loss = loss_function(

        reconstructed,

        x

    )

    return loss.item()