"""
STRIPES MLM Pretraining

Pretrain a Transformer Encoder on STRIPES sequences using Masked Language Modeling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import pickle
from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def tokenize_stripes(stripes_seq):
    """Tokenize a STRIPES sequence into a list of string tokens.

    Format: atoms separated by ';', interactions separated by '.'
    Each atom has 11 features.
    """
    atoms = stripes_seq.split(';')
    tokens = ['<STRIPES>']

    for atom in atoms:
        atom_features = atom.split('.')
        if len(atom_features) >= 11:      
            for i in range(11):
                if i < len(atom_features):   
                    tokens.append(atom_features[i])
                else:
                    tokens.append('<MISSING>')
            tokens.append('<ATOM_SEP>')

    return tokens[:-1] if len(tokens) > 1 else ['<STRIPES>', '<EMPTY>'] 


def build_stripes_vocab(stripes_sequences, min_freq=1, max_len=1200):
    """Build vocabulary from STRIPES sequences only.

    Args:
        min_freq: minimum token frequency to include
        max_len: max token length for filtering during vocab building
    """

    all_tokens = []
    filtered = 0

    for seq in tqdm(stripes_sequences, desc="Tokenizing STRIPES"):
        tokens = tokenize_stripes(seq)
        if len(tokens) <= max_len:
            all_tokens.extend(tokens)
        else:
            filtered += 1

    print(f"Filtered {filtered} sequences exceeding max_len={max_len}")

    token_counter = Counter(all_tokens)

    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
        '<STRIPES>': 3,  
        '<ATOM_SEP>': 4,
        '<EMPTY>': 5,
        '<MISSING>': 6,
    }

    for token, count in token_counter.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    print(f"STRIPES vocabulary size: {len(vocab)}")
    return vocab


class STRIPESPretrainingDataset(Dataset):
    """MLM dataset for STRIPES sequences only."""

    def __init__(self, stripes_sequences, vocab, max_len=1200, mask_prob=0.15):
        self.vocab = vocab
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = vocab['<MASK>']
        self.pad_token_id = vocab['<PAD>']
        self.unk_token_id = vocab['<UNK>']
        self.vocab_size = len(vocab)

        self.sequences = []
        for seq in stripes_sequences:
            tokens = tokenize_stripes(seq)
            if len(tokens) <= max_len:
                self.sequences.append(seq)

        print(f"Dataset: {len(self.sequences)} sequences "
              f"(filtered from {len(stripes_sequences)}, max_len={max_len})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = tokenize_stripes(self.sequences[idx])
        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]

        input_ids = list(ids)
        labels = [-100] * len(input_ids)

        maskable = list(range(1, len(input_ids)))
        if maskable:
            n_mask = max(1, int(len(maskable) * self.mask_prob))
            mask_indices = random.sample(maskable, min(n_mask, len(maskable)))

            for i in mask_indices:
                labels[i] = ids[i]
                r = random.random()
                if r < 0.8:
                    input_ids[i] = self.mask_token_id
                elif r < 0.9:
                    input_ids[i] = random.randint(7, self.vocab_size - 1)

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))


def pretraining_collate_fn(batch):
    """Pad sequences in a batch."""
    input_ids, labels = zip(*batch)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids_padded, labels_padded


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class STRIPESEncoder(nn.Module):
    """Transformer Encoder with MLM head for STRIPES pretraining."""
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=8,
                 dim_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.dropout = nn.Dropout(p=dropout)

        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.mlm_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, attention_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = (src == 0)

        hidden_states = self.encoder(src_emb, src_key_padding_mask=key_padding_mask)
        logits = self.mlm_head(hidden_states)

        return {'logits': logits, 'hidden_states': hidden_states}



class STRIPESPretrainer:
    """Training loop for STRIPES MLM pretraining.

    Uses warmup + linear decay schedule following BERT.
    """

    def __init__(self, model, train_loader, val_loader, vocab,
                 lr=1e-4, warmup_steps=1000, num_epochs=100, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.global_step = 0

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        total_steps = len(train_loader) * num_epochs
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (
                min(1.0, step / max(1, warmup_steps))          
                * max(0.0, 1.0 - step / max(1, total_steps))   
            )
        )
        print(f"Scheduler: warmup={warmup_steps} steps, total={total_steps} steps")

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        n_batches = 0

        for input_ids, labels in tqdm(self.train_loader, desc="Pretraining"):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(input_ids, attention_mask=(input_ids != 0))
            loss = self.criterion(
                output['logits'].view(-1, output['logits'].size(-1)),
                labels.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0

    def validate(self):
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for input_ids, labels in self.val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                output = self.model(input_ids, attention_mask=(input_ids != 0))
                loss = self.criterion(
                    output['logits'].view(-1, output['logits'].size(-1)),
                    labels.view(-1)
                )
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else float('inf')

    def pretrain(self, num_epochs, save_path):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                  f"LR = {lr_now:.2e}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_config': {
                        'vocab_size': len(self.vocab),
                        'd_model': self.model.d_model,
                        'n_heads': self.model.encoder.layers[0].self_attn.num_heads,
                        'n_layers': len(self.model.encoder.layers),
                        'dim_ff': self.model.encoder.layers[0].linear1.out_features,
                        'dropout': self.model.dropout.p
                    },
                    'vocab': self.vocab
                }, save_path)
                print(f"  Saved best model (val_loss={self.best_val_loss:.4f})")

        print("Pretraining completed!")
        return self.train_losses, self.val_losses
