import torch.nn as nn
import torch.nn.functional as F # Contains some additional functions such as activations
import torch
# The Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, n_features, dropout_prob=0.2):
        super(Autoencoder, self).__init__()

        # dropout
        self.drop = nn.Dropout(dropout_prob)
        # # encoder
        # self.enc1 = nn.Linear(in_features=n_features, out_features=int(n_features/2))
        # self.bnorm1 = nn.BatchNorm1d(int(n_features/2))
        # self.enc2 = nn.Linear(in_features=int(n_features/2), out_features=int(n_features/4))
        # self.bnorm2 = nn.BatchNorm1d(int(n_features/4))
        # self.enc3 = nn.Linear(in_features=int(n_features/4), out_features=int(n_features/8))
        # self.bnorm3 = nn.BatchNorm1d(int(n_features/8))
        #
        #
        # # decoder
        # self.dec3 = nn.Linear(in_features=int(n_features/8), out_features=int(n_features/4))
        # self.dec2 = nn.Linear(in_features=int(n_features/4), out_features=int(n_features/2))
        # self.dec1 = nn.Linear(in_features=int(n_features/2), out_features=n_features)

        self.enc1 = Autoencoder._block(n_features, int(n_features/2))
        self.enc2 = Autoencoder._block(int(n_features/2), int(n_features/4))
        self.enc3 = Autoencoder._block(int(n_features/4), int(n_features/8))

        self.dec3 = Autoencoder._block(int(n_features/8), int(n_features/4))
        self.dec2 = Autoencoder._block(int(n_features/4), int(n_features/2))
        # self.dec1 = Autoencoder._block(int(n_features/2), n_features)
        self.dec1 = nn.Linear(int(n_features/2), n_features)


    def forward(self, data):

        z = self.enc1(data)
        z = self.enc2(z)
        z = self.enc3(z)

        z = self.drop(z)

        z = self.dec3(z)
        z = self.dec2(z)
        logits = self.dec1(z)


        # z = self.enc1(data)
        # z = torch.tanh(z)
        # z = self.bnorm1(z)
        # z = self.enc2(z)
        # z = torch.tanh(z)
        # # z = self.bnorm2(z)
        # z = self.enc3(z)
        # z = torch.tanh(z)
        # # z = self.bnorm3(z)
        #
        #
        # # z = self.drop(z)
        #
        # z = self.dec3(z)
        # z = torch.tanh(z)
        # # z = self.bnorm2(z)
        # z = self.dec2(z)
        # z = torch.tanh(z)
        # # z = self.bnorm1(z)
        # logits = self.dec1(z)

        return logits

    @staticmethod
    def _block(in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            # nn.BatchNorm1d(num_features=out_features),
            nn.Tanh(),
        )