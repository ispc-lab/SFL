import torch.nn as nn
import torch
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.1, include_decoder_layers=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.include_decoder_layers = include_decoder_layers

        if self.include_decoder_layers:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, out_dim)
            self.fc4 = nn.Linear(out_dim, 1)
            torch.nn.init.normal_(self.fc4.weight)
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.fc2 = nn.Linear(hidden_dim, out_dim)
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if self.include_decoder_layers:
            x = self.dropout(F.relu(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

        return x


class CNNEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, sequence_length, num_kernels, kernel_length):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=sequence_length, out_channels=num_kernels, kernel_size=kernel_length)
        self.conv2 = nn.Conv1d(in_channels=num_kernels, out_channels=num_kernels * 2, kernel_size=kernel_length)
        self.conv3 = nn.Conv1d(in_channels=num_kernels * 2, out_channels=num_kernels * 3, kernel_size=kernel_length)
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x)
        x = x.squeeze(2)
        return x
    

class DeepDTA(nn.Module):
    def __init__(self,):
        super().__init__()
        # ---- encoder hyper-parameter ----
        num_drug_embeddings = 64
        num_target_embeddings = 25
        drug_dim = 128
        target_dim = 128
        drug_length = 85
        target_length = 1200
        num_filters = 32
        drug_filter_length = 8
        target_filter_length = 8

        self.drug_encoder = CNNEncoder(
            num_embeddings=num_drug_embeddings,
            embedding_dim=drug_dim,
            sequence_length=drug_length,
            num_kernels=num_filters,
            kernel_length=drug_filter_length,
        )

        self.target_encoder = CNNEncoder(
            num_embeddings=num_target_embeddings,
            embedding_dim=target_dim,
            sequence_length=target_length,
            num_kernels=num_filters,
            kernel_length=target_filter_length,
        )

        self.decoder = MLPDecoder(
            in_dim=192,
            hidden_dim=1024,
            out_dim=512,
            dropout_rate=0.2,
            include_decoder_layers=True,
        )

    def forward(self, batch):
        x_drug, x_target = batch['drug'], batch['protein']
        drug_emb = self.drug_encoder(x_drug)
        target_emb = self.target_encoder(x_target)
        comb_emb = torch.cat((drug_emb, target_emb), dim=1)
        return comb_emb, None