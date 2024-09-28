# pip install datasets torch transformers
from datasets import load_dataset

# Загрузка параллельного корпуса для перевода с неметцкого на русский
dataset = load_dataset("wmt14", "de-en")

from transformers import AutoTokenizer

# Используем токенизатор BPE (byte-pair encoding) для обоих языков
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_ru = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
print(dataset['train'][0])

# Токенизация данных
def tokenize_function(example):
    source = tokenizer_ru(example['translation'][0]['de'], padding="max_length", truncation=True, max_length=1000)
    target = tokenizer_en(example['translation'][0]['en'], padding="max_length", truncation=True, max_length=1000)
    return {"input_ids": source['input_ids'], "attention_mask": source['attention_mask'], "labels": target['input_ids']}

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Разделение на тренировочный и тестовый наборы
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, d_model=512, n_heads=8, num_encoder_layers=6, num_decoder_layers=6, forward_expansion=4, dropout=0.1, max_len=128):
        super(Transformer, self).__init__()
        
        self.src_word_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_position_embedding = nn.Embedding(max_len, d_model)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.trg_position_embedding = nn.Embedding(max_len, d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=d_model * forward_expansion, dropout=dropout)

        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape
        
        src_positions = torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(src.device)
        trg_positions = torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(trg.device)

        embed_src = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_mask, tgt_mask=trg_mask)
        out = self.fc_out(out)
        return out

# Гиперпараметры
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_vocab_size = tokenizer_ru.vocab_size
trg_vocab_size = tokenizer_en.vocab_size
src_pad_idx = tokenizer_ru.pad_token_id
trg_pad_idx = tokenizer_en.pad_token_id

model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# Тренировочная петля
def train(model, train_dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        src = batch["input_ids"].to(device)
        trg = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_dataloader)

from torch.utils.data import DataLoader

# Создаем DataLoader для тренировочных и тестовых данных
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
