import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules import encoder


class SentimentCNN(nn.Module):
    """
    The embedding layer + CNN model that will be used to perform sentiment analysis.
    """

    def __init__(self, embed, output_size, num_filters=100,
                 kernel_sizes=(3, 4, 5), freeze_embeddings=True, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentCNN, self).__init__()

        # set class vars
        self.num_filters = num_filters

        # 1. embedding layer
        self.embed = embed
        # (optional) freeze embedding weights
        if freeze_embeddings:
            self.embed.embedding.requires_grad = False

        # 2. convolutional layers
        out_channels = [num_filters]*len(kernel_sizes)
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=1,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            embedding_dim=self.embed.embedding_dim,
        )

        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)

        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds = self.embed(x)  # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)

        # get output of each conv-pool layer, already concat
        x = self.conv_pool(embeds)

        # add dropout
        x = self.dropout(x)

        # final logit
        logit = self.fc(x)

        # sigmoid-activated --> a class score
        return self.sig(logit)
