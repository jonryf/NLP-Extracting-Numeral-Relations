import torch
from torch import nn
import torch.nn.functional as F

from settings import MAX_LENGTH
from utils import get_device


class RelationEncoder(nn.Module):
    """
    Implementation of this module based on https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    """

    def __init__(self, input_size, hidden_size):
        """

        :param input_size: size of vocabulary
        :param hidden_size: embedded size
        """
        super(RelationEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        Forward pass data

        :param input: relation, one word at the time
        :param hidden: hidden state
        :return: current output and hidden vector for the state
        """
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=get_device())


class TweetEncoder(nn.Module):
    """
    Implementation of this module partly based on https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        """

        :param hidden_size: hidden GRU size
        :param output_size: output size
        :param dropout_p: dropout probability
        :param max_length: max sentence length (number of tokens)
        """
        super(TweetEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """

        :param input: tweet, one word at the time
        :param hidden: current hidden state (start with the hidden state from the encoder)
        :param encoder_outputs: outputs from the relation encoder
        :return: output, hidden state
        """
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden_ = self.gru(output, hidden)
        return output, hidden_


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.nn_layer = nn.Linear(512, 2)

    def forward(self, hidden_features, encoder_output):
        """
        Forward pass data

        :param hidden_features:
        :param encoder_output:
        :return: binary classification
        """
        output = torch.cat((hidden_features[0][0], encoder_output), 0)
        output = self.nn_layer(output)
        return output.view(-1).unsqueeze(0)
