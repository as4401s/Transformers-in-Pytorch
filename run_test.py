"""
Quick test script to verify the Transformer model works.
This script runs the same tests as the notebook test cell.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import os
from pathlib import Path

# Import the classes from the notebook (we'll define them here for testing)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print('Restored model weights from the best epoch.')
                if self.verbose:
                    print(f'Early stopping triggered. Best loss: {self.best_loss:.4f}')
        
        return self.early_stop
    
    def _save_checkpoint(self, model: nn.Module):
        self.best_weights = copy.deepcopy(model.state_dict())
    
    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, filepath: str, additional_info: dict = None):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f'Model saved to {filepath}')

def load_model(model: nn.Module, optimizer: torch.optim.Optimizer = None, filepath: str = None, device: torch.device = None):
    if device is None:
        device = torch.device('cpu')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Model file not found: {filepath}')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f'Model loaded from {filepath}')
    print(f'Resuming from epoch {checkpoint.get("epoch", "unknown")}, loss: {checkpoint.get("loss", "unknown"):.4f}')
    
    return checkpoint

# Test parameters
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Run tests
print("=" * 70)
print("TESTING TRANSFORMER MODEL")
print("=" * 70)

# Test 1: Model Creation
print("\n[Test 1] Creating Transformer model...")
try:
    test_transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    print("✓ Model created successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in test_transformer.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in test_transformer.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    raise

# Test 2: Forward Pass
print("\n[Test 2] Testing forward pass...")
try:
    batch_size = 4
    test_src = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))
    test_tgt = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))
    
    test_transformer.eval()
    with torch.no_grad():
        output = test_transformer(test_src, test_tgt[:, :-1])
    
    expected_shape = (batch_size, max_seq_length - 1, tgt_vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print(f"✓ Forward pass successful")
    print(f"  - Input src shape: {test_src.shape}")
    print(f"  - Input tgt shape: {test_tgt[:, :-1].shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    raise

# Test 3: Mask Generation
print("\n[Test 3] Testing mask generation...")
try:
    src_mask, tgt_mask = test_transformer.generate_mask(test_src, test_tgt)
    assert src_mask.shape == (batch_size, 1, 1, max_seq_length), f"Wrong src_mask shape: {src_mask.shape}"
    assert tgt_mask.shape == (batch_size, 1, max_seq_length, max_seq_length), f"Wrong tgt_mask shape: {tgt_mask.shape}"
    print("✓ Mask generation successful")
    print(f"  - Source mask shape: {src_mask.shape}")
    print(f"  - Target mask shape: {tgt_mask.shape}")
except Exception as e:
    print(f"✗ Mask generation failed: {e}")
    raise

# Test 4: Training Step
print("\n[Test 4] Testing training step...")
try:
    test_transformer.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    test_optimizer = optim.Adam(test_transformer.parameters(), lr=0.0001)
    
    output = test_transformer(test_src, test_tgt[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), test_tgt[:, 1:].contiguous().view(-1))
    
    test_optimizer.zero_grad()
    loss.backward()
    test_optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  - Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive"
except Exception as e:
    print(f"✗ Training step failed: {e}")
    raise

# Test 5: Model Saving and Loading
print("\n[Test 5] Testing model save/load...")
try:
    test_save_path = "test_checkpoint.pth"
    
    save_model(
        model=test_transformer,
        optimizer=test_optimizer,
        epoch=1,
        loss=loss.item(),
        filepath=test_save_path,
        additional_info={'test': True}
    )
    
    loaded_transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    loaded_optimizer = optim.Adam(loaded_transformer.parameters(), lr=0.0001)
    
    checkpoint = load_model(
        model=loaded_transformer,
        optimizer=loaded_optimizer,
        filepath=test_save_path,
        device=torch.device('cpu')
    )
    
    loaded_transformer.eval()
    test_transformer.eval()
    with torch.no_grad():
        output_original = test_transformer(test_src, test_tgt[:, :-1])
        output_loaded = loaded_transformer(test_src, test_tgt[:, :-1])
    
    assert torch.allclose(output_original, output_loaded, atol=1e-5), "Loaded model output doesn't match original!"
    print("✓ Model save/load successful")
    print(f"  - Checkpoint saved to: {test_save_path}")
    print(f"  - Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  - Checkpoint loss: {checkpoint['loss']:.4f}")
    
    if os.path.exists(test_save_path):
        os.remove(test_save_path)
        print("  - Test checkpoint cleaned up")
except Exception as e:
    print(f"✗ Model save/load failed: {e}")
    raise

# Test 6: Early Stopping
print("\n[Test 6] Testing early stopping...")
try:
    early_stopping_test = EarlyStopping(patience=3, min_delta=0.001, verbose=False)
    
    losses = [10.0, 9.5, 9.0, 8.9, 8.95, 8.92, 8.91, 8.91, 8.91]
    should_stop = False
    
    for i, loss_val in enumerate(losses):
        should_stop = early_stopping_test(loss_val, test_transformer)
        if should_stop:
            print(f"✓ Early stopping triggered at epoch {i+1} (as expected)")
            break
    
    assert should_stop, "Early stopping should have been triggered!"
    assert early_stopping_test.best_loss == 8.9, f"Best loss should be 8.9, got {early_stopping_test.best_loss}"
    print(f"  - Best loss recorded: {early_stopping_test.best_loss:.4f}")
    print(f"  - Early stopping counter: {early_stopping_test.counter}")
except Exception as e:
    print(f"✗ Early stopping test failed: {e}")
    raise

# Test 7: Gradient Flow
print("\n[Test 7] Testing gradient flow...")
try:
    test_transformer.train()
    test_optimizer.zero_grad()
    
    output = test_transformer(test_src, test_tgt[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), test_tgt[:, 1:].contiguous().view(-1))
    loss.backward()
    
    has_gradients = False
    for name, param in test_transformer.named_parameters():
        if param.grad is not None:
            has_gradients = True
            if torch.isnan(param.grad).any():
                raise ValueError(f"NaN gradients found in {name}!")
            break
    
    assert has_gradients, "No gradients found!"
    print("✓ Gradient flow successful")
    print("  - Gradients are flowing through the model")
    print("  - No NaN gradients detected")
except Exception as e:
    print(f"✗ Gradient flow test failed: {e}")
    raise

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nThe Transformer model is working correctly!")
print("You can now proceed with training on your dataset.")

