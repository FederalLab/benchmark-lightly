import torch
import torch.nn as nn
import torch.nn.functional as F


class EMNISTAE(nn.Module):
    """Recommended model to use for EMNIST AutoEncoder experiments.

    Reddi S, Zaheer M, Sachan D, et al. Adaptive methods for nonconvex optimization[C]//Proceeding of 32nd Conference on Neural Information Processing Systems (NIPS 2018). 2018.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 1000, bias=True),
            nn.Sigmoid(),

            nn.Linear(1000, 500, bias=True),
            nn.Sigmoid(),

            nn.Linear(500, 250, bias=True),
            nn.Sigmoid(),

            nn.Linear(250, 30, bias=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(30, 250, bias=True),
            nn.Sigmoid(),

            nn.Linear(250, 500, bias=True),
            nn.Sigmoid(),

            nn.Linear(500, 1000, bias=True),
            nn.Sigmoid(),

            nn.Linear(1000, 784, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        shape = x.size()
        x = torch.flatten(x, start_dim=1)

        latent_vec = self.encoder(x)

        output = self.decoder(latent_vec)

        return output.reshape(*shape)


if __name__ == "__main__":
    x = torch.randn([128, 1, 28, 28])
    ae = EMNISTAE()
    y = ae(x)
    F.mse_loss(x, y).backward()
