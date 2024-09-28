import torch
import torch.nn as nn
import torch.optim as optim 
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint 
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multy30k 
from torchtext.data import Field, BucketIterator 

spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de')

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenize_ger, init_token='<sos>', eos_token='<eos>', lower=True)
english = Field(tokenize=tokenize_eng, init_token='<sos>', eos_token='<eos>', lower=True)
traind_data, valid_data, test_data = Multy30k.splits(
    exts = ('.de', '.en'), feilds=(german, english)
)
german.build_vocab(traind_data, max_size=10000, min_freq=2)
english.build_vocab(traind_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(
            self,
            embeding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout,
            max_len,
            device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embeding_size)
        self.src_position_embedding = nn.Embedding(max_len, embeding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embeding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embeding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embeding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.dc_out = nn.Linear(embeding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
    
    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask 
    
    def forward(self, src, trg):
        src_seq_length, N = src.shape 
        trg_seq_length, N = trg.shape
        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N)
            .to(self.device)
        )

        embeded_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask = trg_mask
        )

        out = self.fc_out(out)
        return out
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False 
save_model = True 
num_epochs = 5
learning_rate = 3e-4
batch_size = 32
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embeding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi['<pad>']
writer = SummaryWriter('runs/loss_plot')
step = 0
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (traind_data, valid_data, test_data),
    batch_size = batch_size,
    device = device,
    sort_key = lambda x: len(x.src),
    sort_within_batch = True,
)

model = Transformer(
    embeding_size, 
    src_vocab_size, 
    trg_vocab_size, 
    src_pad_idx, 
    num_heads, 
    num_encoder_layers, 
    num_decoder_layers, 
    dropout, 
    max_len, 
    device,
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

sentence = 'Hallo, mein Name ist Igor und ich lebe in Rostow'
for epoch in range(num_epochs):
    print(f'[Epoch {epoch+1}/{num_epochs}]')

    if save_model:
        checkpoint = {
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }

        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length = 100 
    )

    print(f"Translated example sentence \n {translated_sentence}")
    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        output = model(inp_data, target[:-1])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        writer.add_scalar('loss', loss, global_step=step)
        step += 1

score = bleu(test_data, model, german, english, device)
print(f"BLEU score: {score*100:.2f}")