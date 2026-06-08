"""
STRIPES → SMILES Finetuning — Transformer Encoder-Decoder

The encoder is initialised from BERT-style MLM-pretrained weights (pretraining.py).
The decoder is initialised from scratch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import pickle
import logging
import random
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import pandas as pd

from torch.utils.data import Dataset
from pretraining import STRIPESEncoder, PositionalEncoding, tokenize_stripes
from smiles_utils import segment_smiles, sanitize_smiles, is_valid_smiles

logger = logging.getLogger(__name__)

DATASET_MAX_LENGTHS = {
    'AR': 1200,
    'JAK1': 1200,
    'PIM1': 1200,
    'PPAR': 1200,
}



# Model ____________________________________


class STRIPESToSMILESModel(nn.Module):
    """Encoder-decoder for STRIPES → SMILES translation.

    The encoder is loaded from pretrained weights. 
    Only the decoder components are initialised from
    scratch.
    """

    def __init__(self, pretrained_encoder_path, pretrained_vocab_path,
                 smiles_vocab_size, d_model=None, n_heads=8,
                 n_decoder_layers=6, dim_ff=None, dropout=0.1,
                 freeze_encoder_layers=0):
        super().__init__()

        self.pretrained_vocab = self._load_pretrained_encoder(
            pretrained_encoder_path, pretrained_vocab_path
        )

        if d_model is None:
            d_model = self.encoder.d_model
        if dim_ff is None:
            dim_ff = self.encoder.encoder.layers[0].linear1.out_features

        self.d_model = d_model
        self.smiles_vocab_size = smiles_vocab_size

        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)

        self.smiles_embedding = nn.Embedding(smiles_vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_ff, dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, smiles_vocab_size)

        self.pos_encoding = self.encoder.pos_encoding
        self.dropout = nn.Dropout(dropout)

        self._init_decoder_weights()


    def _load_pretrained_encoder(self, model_path, vocab_path):
        logger.info(f"Loading pretrained encoder from {model_path}")

        with open(vocab_path, 'rb') as f:
            pretrained_vocab = pickle.load(f)

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        cfg = checkpoint.get('model_config', {})
        if not cfg:
            state = checkpoint.get('model_state_dict', checkpoint)
            cfg = {
                'vocab_size': len(pretrained_vocab),
                'd_model': state['embedding.weight'].size(1),
                'n_heads': 8, 'n_layers': 8, 'dim_ff': 2048, 'dropout': 0.1,
            }

        self.encoder = STRIPESEncoder(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            dim_ff=cfg['dim_ff'],
            dropout=cfg['dropout'],
        )
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        result = self.encoder.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"Missing keys in pretrained encoder: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys in pretrained encoder: {result.unexpected_keys}")
        logger.info("Pretrained encoder loaded successfully")
        return pretrained_vocab

    def _freeze_encoder_layers(self, n_layers):
        logger.info(f"Freezing first {n_layers} encoder layers + embeddings")
        for p in self.encoder.embedding.parameters():
            p.requires_grad = False
        for i in range(min(n_layers, len(self.encoder.encoder.layers))):
            for p in self.encoder.encoder.layers[i].parameters():
                p.requires_grad = False

    def _init_decoder_weights(self):
        """Initialise ONLY decoder components.  Encoder weights are NOT touched."""
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.output_projection.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.smiles_embedding.weight)


    def encode(self, src):
        """Encode STRIPES using the pretrained encoder (no MLM head)."""
        src_key_padding_mask = (src == 0)
        src_emb = self.encoder.embedding(src) * math.sqrt(self.encoder.d_model)
        src_emb = self.encoder.pos_encoding(src_emb)
        return self.encoder.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.smiles_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = self.layer_norm(output)
        output = self.output_dropout(output)
        return self.output_projection(output)

    def forward(self, src, tgt, src_key_padding_mask=None,
                tgt_key_padding_mask=None, tgt_mask=None):
        memory = self.encode(src)
        memory_key_padding_mask = (src == 0)
        return self.decode(
            tgt, memory, tgt_mask,
            tgt_key_padding_mask, memory_key_padding_mask,
        )

    @staticmethod
    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# Dataset____________________________________


class FinetuningDataset(Dataset):
    """Paired STRIPES → SMILES dataset for finetuning."""

    def __init__(self, stripes_sequences, smiles_sequences,
                 pretrained_vocab, smiles_vocab,
                 dataset_name='default', max_len=None):
        self.stripes_sequences = stripes_sequences
        self.smiles_sequences = smiles_sequences
        self.pretrained_vocab = pretrained_vocab
        self.smiles_vocab = smiles_vocab
        self.max_len = max_len or DATASET_MAX_LENGTHS.get(dataset_name, 700)
        logger.info(f"FinetuningDataset ({dataset_name}): {len(stripes_sequences)} pairs, "
                    f"max_len={self.max_len}")

    def __len__(self):
        return len(self.stripes_sequences)

    def _tokenize_smiles(self, smiles_seq):
        tokens = ['<SOS>']
        try:
            tokens.extend(segment_smiles(smiles_seq))
        except Exception:
            tokens.extend(list(smiles_seq))
        tokens.append('<EOS>')
        return tokens

    def __getitem__(self, idx):
        stripes = self.stripes_sequences[idx]
        smiles = self.smiles_sequences[idx]

        # Encode STRIPES with pretrained vocab
        stripes_tokens = tokenize_stripes(stripes)
        stripes_ids = [self.pretrained_vocab.get(t, self.pretrained_vocab.get('<UNK>', 1))
                       for t in stripes_tokens[:self.max_len]]

        # Encode SMILES with SMILES vocab
        smiles_clean = sanitize_smiles(smiles) or smiles
        smiles_tokens = self._tokenize_smiles(smiles_clean)
        smiles_ids = [self.smiles_vocab.get(t, self.smiles_vocab.get('<UNK>', 1))
                      for t in smiles_tokens[:self.max_len]]

        return (torch.tensor(stripes_ids, dtype=torch.long),
                torch.tensor(smiles_ids, dtype=torch.long))



# Trainer______________________________________

class FinetuningTrainer:

    def __init__(self, model, train_loader, val_loader,
                 smiles_vocab, pretrained_vocab,
                 lr=1e-4, device='cpu',
                 num_epochs=100, warmup_epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.smiles_vocab = smiles_vocab
        self.pretrained_vocab = pretrained_vocab
        self.device = device

        # Differential LR: encoder 10x slower
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': lr * 0.1},
            {'params': decoder_params, 'lr': lr},
        ], weight_decay=1e-3)

        # Warmup + ReduceLROnPlateau 
        warmup_steps = warmup_epochs * len(train_loader)
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_steps,
        )
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.7, min_lr=1e-7,
        )
        self.scheduler = self.plateau_scheduler
        self.warmup_steps = warmup_steps
        self.global_step = 0

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        n_batches = 0

        for stripes_batch, smiles_batch in tqdm(self.train_loader, desc="Fine-tuning"):
            stripes_batch = stripes_batch.to(self.device)
            smiles_batch = smiles_batch.to(self.device)

            decoder_input = smiles_batch[:, :-1]
            target = smiles_batch[:, 1:]

            tgt_mask = self.model.generate_square_subsequent_mask(
                decoder_input.size(1)
            ).to(self.device)
            tgt_key_padding_mask = (decoder_input == 0)

            self.optimizer.zero_grad()
            output = self.model(
                stripes_batch, decoder_input,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask,
            )
            loss = self.criterion(
                output.reshape(-1, output.size(-1)),
                target.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.global_step < self.warmup_steps:
                self.warmup_scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0

    def validate(self):
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for stripes_batch, smiles_batch in tqdm(self.val_loader, desc="Validation"):
                stripes_batch = stripes_batch.to(self.device)
                smiles_batch = smiles_batch.to(self.device)

                decoder_input = smiles_batch[:, :-1]
                target = smiles_batch[:, 1:]

                tgt_mask = self.model.generate_square_subsequent_mask(
                    decoder_input.size(1)
                ).to(self.device)
                tgt_key_padding_mask = (decoder_input == 0)

                output = self.model(
                    stripes_batch, decoder_input,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_mask=tgt_mask,
                )
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    target.reshape(-1),
                )
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else float('inf')

    def fine_tune(self, num_epochs=100, save_path=None, early_stopping_patience=15):
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr_now:.2e}")

            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'smiles_vocab': self.smiles_vocab,
                        'pretrained_vocab': self.pretrained_vocab,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'epoch': epoch,
                        'best_val_loss': self.best_val_loss,
                        'model_config': {
                            'd_model': self.model.d_model,
                            'smiles_vocab_size': self.model.smiles_vocab_size,
                            'n_decoder_layers': len(self.model.decoder.layers),
                            'freeze_encoder_layers': 0,
                            'dropout': self.model.dropout.p
                                       if hasattr(self.model.dropout, 'p') else 0.1,
                        },
                    }, save_path)
                    print(f"  Saved best model (val_loss={self.best_val_loss:.4f}, "
                          f"improvement={improvement:.4f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement for {self.patience_counter} epochs")
                if self.patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
                if epoch > 20 and val_loss > self.best_val_loss * 1.2:
                    print(f"  Val loss diverging. Stopping at epoch {epoch+1}")
                    break



# Translator (beam search — per-sequence uniqueness)_______________

class SMILESTranslator:
    """Generate SMILES from STRIPES using beam search."""

    def __init__(self, model_path, smiles_vocab, pretrained_vocab, device='cpu',
                 pretrained_encoder_path=None, pretrained_vocab_path=None,
                 model_config=None):
        self.smiles_vocab = smiles_vocab
        self.pretrained_vocab = pretrained_vocab
        self.device = device
        self.smiles_inv_vocab = {v: k for k, v in smiles_vocab.items()}

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        cfg = checkpoint.get('model_config', model_config or {})
        if not cfg:
            state = checkpoint.get('model_state_dict', checkpoint)
            max_layer = 0
            for key in state:
                if key.startswith('decoder.layers.'):
                    max_layer = max(max_layer, int(key.split('.')[2]))
            cfg = {
                'smiles_vocab_size': len(smiles_vocab),
                'd_model': 512,
                'freeze_encoder_layers': 0,
                'n_decoder_layers': max_layer + 1,
                'dropout': 0.1,
            }

        self.model = STRIPESToSMILESModel(
            pretrained_encoder_path=pretrained_encoder_path,
            pretrained_vocab_path=pretrained_vocab_path,
            smiles_vocab_size=cfg.get('smiles_vocab_size', len(smiles_vocab)),
            d_model=cfg.get('d_model', 512),
            freeze_encoder_layers=cfg.get('freeze_encoder_layers', 0),
            n_decoder_layers=cfg.get('n_decoder_layers', 6),
            dropout=cfg.get('dropout', 0.1),
        )
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def predict(self, stripes_sequence, max_length=100, beam_size=5,
                n_molecules=5, temperature=1.0):
        """Beam-search prediction with temperature scaling."""
        self.model.eval()

        with torch.no_grad():
            tokens = tokenize_stripes(stripes_sequence)
            ids = [self.pretrained_vocab.get(t, self.pretrained_vocab.get('<UNK>', 1))
                   for t in tokens]
            src = torch.tensor([ids]).to(self.device)

            # Encode source ONCE, reuse for all beam steps
            memory = self.model.encode(src)
            memory_key_padding_mask = (src == 0)

            sos_id = self.smiles_vocab.get('<SOS>', 2)
            eos_id = self.smiles_vocab.get('<EOS>', 3)

            beams = [(0.0, [sos_id])]

            for _step in range(max_length):
                new_beams = []
                for score, seq_ids in beams:
                    if seq_ids[-1] == eos_id:
                        new_beams.append((score, seq_ids))
                        continue

                    tgt = torch.tensor([seq_ids]).to(self.device)
                    tgt_mask = self.model.generate_square_subsequent_mask(
                        tgt.size(1)
                    ).to(self.device)
                    output = self.model.decode(
                        tgt, memory, tgt_mask=tgt_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                    )

                    logits = output[0, -1, :] / temperature
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_idx = torch.topk(probs, beam_size)

                    for prob, idx in zip(top_probs, top_idx):
                        new_seq = seq_ids + [idx.item()]
                        new_score = score + torch.log(prob).item()
                        new_beams.append((new_score, new_seq))

                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[
                    :beam_size * 2
                ]
                if all(seq[-1] == eos_id for _, seq in beams):
                    break

            # Collect unique molecules
            unique = []
            seen = set()
            for score, seq_ids in sorted(beams, key=lambda x: x[0], reverse=True):
                tokens_out = []
                for tok_id in seq_ids[1:]:  # skip <SOS>
                    if tok_id == eos_id:
                        break
                    tok = self.smiles_inv_vocab.get(tok_id, '')
                    if tok and tok not in ('<PAD>', '<UNK>'):
                        tokens_out.append(tok)
                smiles = ''.join(tokens_out)
                if smiles and smiles not in seen:
                    unique.append((smiles, score))
                    seen.add(smiles)
                    if len(unique) >= n_molecules:
                        break

            while len(unique) < n_molecules:
                unique.append(("INVALID", float('-inf')))
            return unique[:n_molecules]

    def translate_batch(self, stripes_sequences, beam_size=5, n_molecules=5,
                        max_attempts=2, initial_temperature=1.0,
                        temperature_increment=0.4):
        """Translate a batch of STRIPES → SMILES.

        Uniqueness is enforced **per-sequence** .
        If the first attempt doesn't yield enough valid unique molecules,
        subsequent attempts increase temperature to explore more.
        """
        all_results = []

        for seq_idx, stripes_seq in enumerate(
            tqdm(stripes_sequences, desc="Translating")
        ):
            valid_unique = []  # per-sequence set

            for attempt in range(max_attempts):
                temp = initial_temperature + attempt * temperature_increment

                preds = self.predict(
                    stripes_seq,
                    beam_size=max(beam_size, 5),
                    n_molecules=max(beam_size, 5),
                    temperature=temp,
                )

                seen_this_seq = {s for s, _ in valid_unique}
                for smiles, score in preds:
                    if not is_valid_smiles(smiles):
                        continue
                    if smiles in seen_this_seq:
                        continue
                    valid_unique.append((smiles, score))
                    seen_this_seq.add(smiles)
                    if len(valid_unique) >= n_molecules:
                        break

                if len(valid_unique) >= n_molecules:
                    break

            final = [s for s, _ in sorted(valid_unique, key=lambda x: x[1],
                                          reverse=True)[:n_molecules]]
            while len(final) < n_molecules:
                final.append("INVALID")
            all_results.append(final)

            if (seq_idx + 1) % 10 == 0:
                logger.info(f"Translated {seq_idx+1}/{len(stripes_sequences)}")

        return all_results



# Evaluation_______________________

def calculate_molecular_similarity(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)
        return TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def evaluate_predictions(predictions, test_smiles, smiles_vocab):
    """Evaluate generated SMILES against ground truth."""
    results = []
    valid_preds = 0
    exact_matches = 0
    similarities = []
    total_preds = 0
    total_samples = len(test_smiles)

    for i, (pred_mols, gt_smiles) in enumerate(zip(predictions, test_smiles)):
        for mol_idx, pred_smi in enumerate(pred_mols):
            pred_smi = pred_smi.strip()
            if not pred_smi:
                pred_smi = "INVALID"

            valid = is_valid_smiles(pred_smi)
            if valid:
                valid_preds += 1
                canonical_pred = sanitize_smiles(pred_smi, to_canonical=True) or pred_smi
            else:
                pred_smi = "INVALID"
                canonical_pred = "INVALID"

            sim = 0.0
            if valid and pred_smi != "INVALID":
                sim = calculate_molecular_similarity(gt_smiles, canonical_pred)
                similarities.append(sim)
                if canonical_pred == gt_smiles:
                    exact_matches += 1

            total_preds += 1
            results.append({
                'sample_id': i,
                'molecule_id': mol_idx,
                'can_smiles': gt_smiles,
                'predicted_smiles': pred_smi,
                'is_valid': valid,
                'similarity': sim,
                'exact_match': canonical_pred == gt_smiles,
            })

    # Best per sample
    best_per_sample = []
    for sid in range(total_samples):
        sample_res = [r for r in results if r['sample_id'] == sid]
        valid_res = [r for r in sample_res if r['is_valid']]
        if valid_res:
            best_per_sample.append(max(valid_res, key=lambda x: x['similarity']))

    best_sims = [r['similarity'] for r in best_per_sample]
    best_exact = sum(1 for r in best_per_sample if r['exact_match'])

    metrics = {
        'total_samples': total_samples,
        'total_predictions': total_preds,
        'valid_predictions': valid_preds,
        'validity_rate': valid_preds / total_preds if total_preds else 0,
        'exact_matches': exact_matches,
        'exact_match_rate': exact_matches / total_preds if total_preds else 0,
        'avg_similarity': float(np.mean(similarities)) if similarities else 0,
        'median_similarity': float(np.median(similarities)) if similarities else 0,
        'best_per_sample_validity_rate': len(best_per_sample) / total_samples,
        'best_per_sample_exact_match_rate': best_exact / total_samples,
        'best_per_sample_avg_similarity': float(np.mean(best_sims)) if best_sims else 0,
    }

    return pd.DataFrame(results), metrics
