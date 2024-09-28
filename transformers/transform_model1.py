import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import numpy as np
import spacy
import random

spacy_en = spacy.load('en_core_web_sm')
spacy_ru = spacy.load('ru_core_news_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_ru(text):
    return [tok.text for tok in spacy_ru.tokenizer(text)]

SRC = Field(tokenize=tokenize_ru, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.ru', '.en'), fields=(SRC, TRG))
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Создание трансформера Алё
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, d_model=512, n_heads=8, num_encoder_layers=6, num_decoder_layers=6, forward_expansion=4, dropout=0.1, max_len=100):
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

# Определение гиперпараметров и тренировка 
# Гиперпараметры
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_vocab_size = len(SRC.vocab)
trg_vocab_size = len(TRG.vocab)
src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# Тренировка
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:-1])
        
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.to(device)
            trg = batch.trg.to(device)

            output = model(src, trg[:-1])

            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Запуск тренировки
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    print(f'Epoch: {epoch+1}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
