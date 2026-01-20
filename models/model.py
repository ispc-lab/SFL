import torch
import torch.nn as nn
import torch.nn.functional as F

class PredModel(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes=1, dropout_rate=0.1):
        super(PredModel, self).__init__()

        self.encoder = encoder

        if hidden_dim <= 256:
            self.use_projection = True
            self.feature_projection = nn.Linear(hidden_dim, 32)
            # Initialize weights using Kaiming Normal
            nn.init.kaiming_normal_(self.feature_projection.weight, nonlinearity='leaky_relu')
            if self.feature_projection.bias is not None:
                nn.init.constant_(self.feature_projection.bias, 0)
        elif hidden_dim == 32:
            self.use_projection = False
        else:
            self.use_projection = True
            self.feature_projection = nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(128, 32),
                )
            # # Initialize weights using Kaiming Normal
            for m in self.feature_projection:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')


        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

        # # Initialize weights using Kaiming Normal
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

    def forward(self, batch):
        repr, _ = self.encoder(batch)
        if self.use_projection:
            repr = self.feature_projection(repr)
        x = self.mlp(repr)
        return x, repr

