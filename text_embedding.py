import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim

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


# Assuming 'data.csv' is the name of your CSV file
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
    return pad_sequence(text_list, padding_value=vocab["<pad>"]).T, torch.tensor(label_list, dtype=torch.int64)


dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

class EmbNet(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(emb_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 2)  
    
    def forward(self, x):
        embeds = self.embedding(x)
        embeds = torch.mean(embeds, dim=1)  # Average the embeddings over the sequence length
        out = self.fc(embeds)
        return F.log_softmax(out, dim=-1)
    
emb_size = len(vocab)
hidden_size = 128

model = EmbNet(emb_size, hidden_size)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):

    for batch in dataloader:
        texts, labels = batch

        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)  # Apply log_softmax here

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')
