import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim

print("Is CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("Current GPU: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Read the CSV file into a pandas DataFrame
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index, 0]  # Assuming the features are in columns 1 onwards
        label = self.data.iloc[index,1]  # Assuming the label is in the first column

        # If a transform is provided, apply it to the sample
        if self.transform:
            sample = self.transform(sample)

        return sample, torch.tensor(label, dtype=torch.long)

train_dataset = CustomDataset('D:\\Python_Project\\Deep_Learning_Torch\\IMDB_data\\Train\\train.csv')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')  # Using Spacy tokenizer

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_dataset), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
vocab.set_default_index(vocab["<unk>"])  # Set default index for unknown tokens

def collate_batch(batch):
    label_list, text_list, = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor([vocab[token] for token in tokenizer(_text)], dtype=torch.int64)
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=vocab["<pad>"]), torch.tensor(label_list, dtype=torch.int64)


dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

class LSTMNet(nn.Module):
    def __init__(self, emb_size, hidden_size, n_category, batch_size=1, n_lstm=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_lstm = n_lstm
        self.e = nn.Embedding(emb_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_lstm)
        self.fc2 = nn.Linear(hidden_size, n_category)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        batch_size = x.size()[1]
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        e_out = self.e(x)
        h0 = torch.zeros(self.n_lstm, self.batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.n_lstm, self.batch_size, self.hidden_size, device=x.device)
        rnn_o, _ = self.rnn(e_out, (h0, c0))
        rnn_o = rnn_o[-1]
        fc = F.dropout(self.fc2(rnn_o), p=0.8)
        return self.softmax(fc)
    
emb_size = len(vocab)
hidden_size = 128

model = LSTMNet(emb_size, hidden_size, n_category=3, batch_size=32).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):

    for batch in dataloader:
        texts, labels = batch

        optimizer.zero_grad()

        outputs = model(texts.to(device))
        loss = criterion(outputs, labels.to(device))  # Apply log_softmax here

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels.to(device)).sum().item()
        accuracy = correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')