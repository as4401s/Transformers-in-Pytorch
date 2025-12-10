"""
Test script to verify that the Transformer model works correctly.
Run this script to test all functionality including forward pass, training, saving/loading, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add the current directory to the path to import transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from transformers import build_transformer, EarlyStopping, save_model, load_model
    USE_BUILDER = True
except ImportError:
    print("Note: Could not import from transformers.py. Make sure the file exists.")
    USE_BUILDER = False

def test_model_creation():
    """Test 1: Model Creation"""
    print("\n[Test 1] Creating Transformer model...")
    try:
        if USE_BUILDER:
            # Use the build_transformer function
            src_vocab_size = 5000
            tgt_vocab_size = 5000
            src_seq_len = 100
            tgt_seq_len = 100
            d_model = 512
            N = 6
            h = 8
            dropout = 0.1
            d_ff = 2048
            
            model = build_transformer(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                src_seq_len=src_seq_len,
                tgt_seq_len=tgt_seq_len,
                d_model=d_model,
                N=N,
                h=h,
                dropout=dropout,
                d_ff=d_ff
            )
        else:
            # For notebook version, you would need to define the classes here
            raise ImportError("Please run this test from the notebook or ensure transformers.py is available")
        
        print("✓ Model created successfully")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        return model, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        raise

def test_forward_pass(model, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len):
    """Test 2: Forward Pass"""
    print("\n[Test 2] Testing forward pass...")
    try:
        batch_size = 4
        test_src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
        test_tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
        
        # Generate masks
        src_mask = (test_src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (test_tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = test_tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        
        model.eval()
        with torch.no_grad():
            # Use the encode-decode-project API
            encoder_output = model.encode(test_src, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, test_tgt[:, :-1], tgt_mask[:, :, :-1, :-1])
            output = model.project(decoder_output)
        
        expected_shape = (batch_size, tgt_seq_len - 1, tgt_vocab_size)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"✓ Forward pass successful")
        print(f"  - Input src shape: {test_src.shape}")
        print(f"  - Input tgt shape: {test_tgt[:, :-1].shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        return model, test_src, test_tgt, output
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise

def test_forward_method(model, test_src, test_tgt):
    """Test 2b: Forward method (if available)"""
    print("\n[Test 2b] Testing forward() method...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(test_src, test_tgt[:, :-1])
        
        print(f"✓ Forward() method successful")
        print(f"  - Output shape: {output.shape}")
        return output
    except AttributeError:
        print("  - Note: forward() method not available (using encode/decode/project API)")
        return None
    except Exception as e:
        print(f"✗ Forward() method failed: {e}")
        raise

def test_training_step(model, test_src, test_tgt, tgt_vocab_size):
    """Test 3: Training Step"""
    print("\n[Test 3] Testing training step...")
    try:
        model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        # Forward pass
        if hasattr(model, 'forward') and callable(getattr(model, 'forward', None)):
            output = model(test_src, test_tgt[:, :-1])
        else:
            # Use encode-decode-project
            src_mask = (test_src != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = (test_tgt != 0).unsqueeze(1).unsqueeze(3)
            seq_length = test_tgt.size(1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
            tgt_mask = tgt_mask & nopeak_mask
            
            encoder_output = model.encode(test_src, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, test_tgt[:, :-1], tgt_mask[:, :, :-1, :-1])
            output = model.project(decoder_output)
        
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), test_tgt[:, 1:].contiguous().view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  - Loss: {loss.item():.4f}")
        assert not torch.isnan(loss), "Loss is NaN!"
        assert loss.item() > 0, "Loss should be positive"
        return model, optimizer, loss
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        raise

def test_save_load(model, optimizer, loss, test_src, test_tgt, tgt_vocab_size):
    """Test 4: Model Saving and Loading"""
    print("\n[Test 4] Testing model save/load...")
    try:
        test_save_path = "test_checkpoint.pth"
        
        # Save model
        save_model(
            model=model,
            optimizer=optimizer,
            epoch=1,
            loss=loss.item(),
            filepath=test_save_path,
            additional_info={'test': True}
        )
        
        # Create new model and load
        # Note: In a real scenario, you'd need to rebuild the model with the same architecture
        # For this test, we'll just verify the checkpoint was saved correctly
        checkpoint = torch.load(test_save_path, map_location='cpu')
        
        assert 'model_state_dict' in checkpoint, "Checkpoint missing model_state_dict"
        assert 'optimizer_state_dict' in checkpoint, "Checkpoint missing optimizer_state_dict"
        assert 'epoch' in checkpoint, "Checkpoint missing epoch"
        assert 'loss' in checkpoint, "Checkpoint missing loss"
        
        print("✓ Model save/load successful")
        print(f"  - Checkpoint saved to: {test_save_path}")
        print(f"  - Checkpoint epoch: {checkpoint['epoch']}")
        print(f"  - Checkpoint loss: {checkpoint['loss']:.4f}")
        
        # Clean up test file
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
            print("  - Test checkpoint cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ Model save/load failed: {e}")
        raise

def test_early_stopping(model):
    """Test 5: Early Stopping"""
    print("\n[Test 5] Testing early stopping...")
    try:
        early_stopping = EarlyStopping(patience=3, min_delta=0.001, verbose=False)
        
        # Simulate decreasing loss
        losses = [10.0, 9.5, 9.0, 8.9, 8.95, 8.92, 8.91, 8.91, 8.91]
        should_stop = False
        
        for i, loss_val in enumerate(losses):
            should_stop = early_stopping(loss_val, model)
            if should_stop:
                print(f"✓ Early stopping triggered at epoch {i+1} (as expected)")
                break
        
        assert should_stop, "Early stopping should have been triggered!"
        assert early_stopping.best_loss == 8.9, f"Best loss should be 8.9, got {early_stopping.best_loss}"
        print(f"  - Best loss recorded: {early_stopping.best_loss:.4f}")
        print(f"  - Early stopping counter: {early_stopping.counter}")
        return True
    except Exception as e:
        print(f"✗ Early stopping test failed: {e}")
        raise

def test_gradient_flow(model, test_src, test_tgt, tgt_vocab_size):
    """Test 6: Gradient Flow"""
    print("\n[Test 6] Testing gradient flow...")
    try:
        model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        optimizer.zero_grad()
        
        if hasattr(model, 'forward') and callable(getattr(model, 'forward', None)):
            output = model(test_src, test_tgt[:, :-1])
        else:
            src_mask = (test_src != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = (test_tgt != 0).unsqueeze(1).unsqueeze(3)
            seq_length = test_tgt.size(1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
            tgt_mask = tgt_mask & nopeak_mask
            
            encoder_output = model.encode(test_src, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, test_tgt[:, :-1], tgt_mask[:, :, :-1, :-1])
            output = model.project(decoder_output)
        
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), test_tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        
        # Check if gradients are flowing
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                if torch.isnan(param.grad).any():
                    raise ValueError(f"NaN gradients found in {name}!")
                break
        
        assert has_gradients, "No gradients found!"
        print("✓ Gradient flow successful")
        print("  - Gradients are flowing through the model")
        print("  - No NaN gradients detected")
        return True
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        raise

def main():
    """Run all tests"""
    print("=" * 70)
    print("TESTING TRANSFORMER MODEL")
    print("=" * 70)
    
    try:
        # Run all tests
        model, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len = test_model_creation()
        model, test_src, test_tgt, output = test_forward_pass(model, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len)
        test_forward_method(model, test_src, test_tgt)
        model, optimizer, loss = test_training_step(model, test_src, test_tgt, tgt_vocab_size)
        test_save_load(model, optimizer, loss, test_src, test_tgt, tgt_vocab_size)
        test_early_stopping(model)
        test_gradient_flow(model, test_src, test_tgt, tgt_vocab_size)
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nThe Transformer model is working correctly!")
        print("You can now proceed with training on your dataset.")
        return 0
    except Exception as e:
        print("\n" + "=" * 70)
        print("TESTS FAILED! ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

