import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [('My grandmother ate the polemta'.split(), ['DET', 'NN', 'V','DET', 'NN']),
                 ('Marina read my book'.split(), ['NN', 'V', 'DET', 'NN'])]

word_index = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_index:
            word_index[word] = len(word_index)

print(word_index)

tag_to_ix = {'DET': 0, 'NN': 1, 'V': 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_index), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        
        sentence_in = prepare_sequence(sentence, word_index)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in)
        
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

        inputs = prepare_sequence(training_data[0][0], word_index)
        tag_scores = model(inputs)

ix_to_tag = {0: 'DET', 1: 'NN', 2: 'V'}

def get_max_prob_result(inp, ix_to_tag):
    idx_max = np.argmax(inp, axis=0)
    return ix_to_tag[idx_max]

test_sentence = training_data[0][0]
inputs = prepare_sequence(test_sentence, word_index)
tag_scores = model(inputs)

for i in range(len(test_sentence)):
    print(test_sentence[i], get_max_prob_result(tag_scores[i].data.numpy, ix_to_tag))    