import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):

    def __init__(self, config):
        super(RNN, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                           num_layers=config.n_layers, dropout=config.dp_ratio,
                           bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return outputs


class SequenceLabeler(nn.Module):
    def __init__(self, config):
        super(SequenceLabeler, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = nn.Linear(config.d_embed, config.d_proj)
        self.rnn = RNN(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()

        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2

        self.out = nn.Sequential(
            nn.Linear(seq_in_size, seq_in_size),
            self.relu,
            self.dropout,
            nn.Linear(seq_in_size, seq_in_size),
            self.relu,
            self.dropout,
            nn.Linear(seq_in_size, config.d_out))

    def init_hidden(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        return (Variable(inputs.data.new(*state_shape).zero_()),
                Variable(inputs.data.new(*state_shape).zero_()))

    def forward(self, batch):
        word_embed = self.embed(batch)
        if self.config.fix_emb:
            word_embed = Variable(word_embed.data)
        if self.config.projection:
            word_embed = self.relu(self.projection(word_embed))
        word_embed = self.dropout(word_embed) # This seem to work better.
        rnn_out = self.rnn(word_embed)
        scores = self.out(rnn_out)
        return scores

    def evaluate(self, data_iter, loss_fn, config, special_tokens=None):
        self.eval()
        data_iter.init_epoch()
        if not special_tokens:
            special_tokens = set()

        # calculate accuracy on validation set
        n_correct = 0
        n_total = 0
        total_loss = 0.0
        batch_idx = 0
        for batch_idx, batch in enumerate(data_iter):
            answer = self(batch.word)
            predicted = torch.max(answer, 2)[1].view(-1).data
            correct = batch.label.view(-1).data
            words = batch.word.view(-1).data
            for idx, word in enumerate(words):
                if word not in special_tokens:
                    if predicted[idx] == correct[idx]:
                        n_correct += 1
                    n_total += 1
            loss = loss_fn(answer.view(-1, config.d_out), batch.label.view(-1))
            total_loss += loss.data[0]
        loss_per_batch = total_loss / (batch_idx + 1.0)
        accuracy = 100. * n_correct / n_total
        return accuracy, loss_per_batch
