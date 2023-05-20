#references:
#https://towardsdatascience.com/sarcasm-detection-with-nlp-cbff1723f69a

#Data Collection:
#Provided Training Set: https://github.com/iabufarha/iSarcasmEval 
#Provided PDF: https://aclanthology.org/2022.semeval-1.111.pdf
#Additional Sets:

import torch
import torch.nn as nn

class SarcasmAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_rate):
        super(SarcasmAnalysisModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout_rate, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)