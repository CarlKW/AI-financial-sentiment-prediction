# How to Create the Optimized Training Notebook

## Quick Summary

You already have `sentiment-analysis-optimized-v2.ipynb` - just need to update a few cells.

---

## Changes Needed

### ✅ Already Done:

- Cell 4: Removed augmentation setup
- Cell 5: Added training config constants
- Cell 9: Removed augmentation code, using original data
- Cell 10: Using original data for datasets

### 🔧 Still Need to Update:

#### Cell 12 (Model Setup):

Replace the entire cell with:

```python
from transformers import AutoTokenizer

# Model and training configuration
modelName = "ProsusAI/finbert"
MAX_LEN = 128
BATCH_SIZE = 32
DROPOUT = 0.3
WEIGHT_DECAY = 0.01

# Phase 1: Warmup (frozen backbone)
WARMUP_EPOCHS = 3
WARMUP_LR = 2e-4

# Phase 2: Fine-tuning (unfrozen backbone)
FINETUNE_EPOCHS = 15
FINETUNE_LR = 2e-5  # 10x lower
PATIENCE = 3  # Early stopping

tokenizer = AutoTokenizer.from_pretrained(modelName)

# Create data loaders with ORIGINAL data
trainLoader, valLoader = getLoaders(train_texts, train_labels, tokenizer,
                                     maxLen=MAX_LEN, batchSize=BATCH_SIZE)

# Calculate class weights
pos_weight = torch.tensor([negative_count / positive_count])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = pos_weight.to(device)

print(f"{'='*80}")
print(f"TRAINING CONFIGURATION")
print(f"{'='*80}")
print(f"Device: {device}")
print(f"Batches per epoch: {len(trainLoader)}")
print(f"\nPhase 1 (Warmup): {WARMUP_EPOCHS} epochs at LR={WARMUP_LR}")
print(f"Phase 2 (Fine-tune): Up to {FINETUNE_EPOCHS} epochs at LR={FINETUNE_LR}")
print(f"\nClass weight (negative): {pos_weight.item():.3f}")
print(f"{'='*80}\n")

# Initialize model
model = FinbertBinaryClf(modelName, pDrop=DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}\n")
```

#### Cell 13 (Training) - REPLACE ENTIRELY:

```python
# ============================================================================
# PHASE 1: HEAD WARMUP (Frozen Backbone)
# ============================================================================

# Freeze backbone
for p in model.backbone.parameters():
    p.requires_grad = False

trainable_warmup = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("="*80)
print("PHASE 1: HEAD WARMUP (Backbone Frozen)")
print("="*80)
print(f"Trainable parameters: {trainable_warmup:,}")
print(f"Data-to-parameter ratio: {len(train_texts) / trainable_warmup:.2f}:1")
print(f"\nTraining head for {WARMUP_EPOCHS} epochs...\n")

# Optimizer for warmup
optimizer_warmup = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=WARMUP_LR,
    weight_decay=WEIGHT_DECAY
)

# Warmup training
warmup_history = {
    'train_loss': [], 'train_acc': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_f1': []
}

for epoch in range(WARMUP_EPOCHS):
    trLoss, trAcc, trF1 = runEpoch(model, trainLoader, device, train=True,
                                    optimizer=optimizer_warmup, posWeight=pos_weight)
    vaLoss, vaAcc, vaF1 = runEpoch(model, valLoader, device, train=False)

    warmup_history['train_loss'].append(trLoss)
    warmup_history['train_acc'].append(trAcc)
    warmup_history['train_f1'].append(trF1)
    warmup_history['val_loss'].append(vaLoss)
    warmup_history['val_acc'].append(vaAcc)
    warmup_history['val_f1'].append(vaF1)

    print(f"[Warmup] Epoch {epoch+1}/{WARMUP_EPOCHS}:")
    print(f"  Train - Loss: {trLoss:.4f}, Acc: {trAcc:.3f}, F1: {trF1:.3f}")
    print(f"  Val   - Loss: {vaLoss:.4f}, Acc: {vaAcc:.3f}, F1: {vaF1:.3f}")
    print()

print("✓ Warmup phase complete!\n")

# ============================================================================
# PHASE 2: FINE-TUNING (Unfrozen Backbone + Early Stopping)
# ============================================================================

# Unfreeze backbone
for p in model.backbone.parameters():
    p.requires_grad = True

trainable_finetune = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("="*80)
print("PHASE 2: FINE-TUNING (Backbone Unfrozen + Early Stopping)")
print("="*80)
print(f"Trainable parameters: {trainable_finetune:,}")
print(f"Data-to-parameter ratio: {len(train_texts) / trainable_finetune:.6f}:1")
print(f"Learning rate: {FINETUNE_LR} (10x lower than warmup)")
print(f"Early stopping patience: {PATIENCE} epochs")
print(f"\nFine-tuning for up to {FINETUNE_EPOCHS} epochs...\n")

# Optimizer for fine-tuning (LOWER LR!)
optimizer_finetune = AdamW(
    model.parameters(),
    lr=FINETUNE_LR,
    weight_decay=WEIGHT_DECAY
)

# Fine-tuning with early stopping
best_val_loss = float('inf')
best_val_f1 = 0.0
patience_counter = 0
best_epoch = 0

finetune_history = {
    'train_loss': [], 'train_acc': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_f1': []
}

for epoch in range(FINETUNE_EPOCHS):
    trLoss, trAcc, trF1 = runEpoch(model, trainLoader, device, train=True,
                                    optimizer=optimizer_finetune, posWeight=pos_weight)
    vaLoss, vaAcc, vaF1 = runEpoch(model, valLoader, device, train=False)

    finetune_history['train_loss'].append(trLoss)
    finetune_history['train_acc'].append(trAcc)
    finetune_history['train_f1'].append(trF1)
    finetune_history['val_loss'].append(vaLoss)
    finetune_history['val_acc'].append(vaAcc)
    finetune_history['val_f1'].append(vaF1)

    print(f"[Fine-tune] Epoch {epoch+1}/{FINETUNE_EPOCHS}:")
    print(f"  Train - Loss: {trLoss:.4f}, Acc: {trAcc:.3f}, F1: {trF1:.3f}")
    print(f"  Val   - Loss: {vaLoss:.4f}, Acc: {vaAcc:.3f}, F1: {vaF1:.3f}")

    # Check for improvement
    if vaLoss < best_val_loss:
        best_val_loss = vaLoss
        best_val_f1 = vaF1
        best_epoch = epoch
        patience_counter = 0

        # Save best model
        torch.save({
            'epoch': len(warmup_history['train_loss']) + epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_finetune.state_dict(),
            'val_loss': vaLoss,
            'val_f1': vaF1,
        }, "finbert_finetuned.pt")
        print(f"  ✓ New best model saved! (Val Loss: {vaLoss:.4f}, Val F1: {vaF1:.3f})")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")

    print()

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"⚠ Early stopping triggered at epoch {epoch+1}")
        print(f"Best model from fine-tune epoch {best_epoch+1}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation F1: {best_val_f1:.3f}")
        break

print("\n" + "="*80)
print("✓ Training complete!")
print(f"Best model saved to: finbert_finetuned.pt")
print("="*80)
```

#### Cell 14 (Plots) - ADD THIS LINE:

At the top of the plotting cell, add:

```python
import matplotlib.pyplot as plt

# Combine warmup and fine-tune history
full_history = {
    'train_loss': warmup_history['train_loss'] + finetune_history['train_loss'],
    'train_acc': warmup_history['train_acc'] + finetune_history['train_acc'],
    'train_f1': warmup_history['train_f1'] + finetune_history['train_f1'],
    'val_loss': warmup_history['val_loss'] + finetune_history['val_loss'],
    'val_acc': warmup_history['val_acc'] + finetune_history['val_acc'],
    'val_f1': warmup_history['val_f1'] + finetune_history['val_f1']
}

# Then use full_history instead of history in plots
# Also add vertical line to show where fine-tuning starts:
warmup_end = len(warmup_history['train_loss'])
axes[0].axvline(x=warmup_end-0.5, color='red', linestyle='--', alpha=0.5, label='Fine-tune starts')
```

#### Cell 15 (Model Summary):

Change:

```python
# FROM:
checkpoint = torch.load("finbert_custom_head.pt")

# TO:
checkpoint = torch.load("finbert_finetuned.pt")
```

And update the summary text to mention 2-phase training.

---

## Expected Results

```
PHASE 1: HEAD WARMUP
Epoch 1: Val F1 ~0.80-0.82
Epoch 2: Val F1 ~0.81-0.83
Epoch 3: Val F1 ~0.82-0.84

PHASE 2: FINE-TUNING
Epoch 1: Val F1 ~0.84-0.85 ✓
Epoch 2: Val F1 ~0.84-0.85
Epoch 3: Val F1 ~0.83-0.84 (might start to overfit)
Early stopping: Saves best epoch!

Final F1: 0.84-0.85 WITHOUT overfitting!
```

---

## Why This Works

1. **Warmup**: Trains head with frozen backbone (safe, fast convergence)
2. **Lower LR**: Fine-tuning uses 10x lower learning rate (prevents destroying pre-trained weights)
3. **Early Stopping**: Stops BEFORE overfitting happens
4. **Class Weights**: Handles imbalance without augmentation noise

This combines the best of both worlds:

- High F1 from fine-tuning (0.84-0.85)
- No overfitting from early stopping
- Stable generalization from 2-phase approach

---

## Alternative: Simpler Approach

If the above is too complex, just use this in Cell 13:

```python
# SIMPLE: Train head only for more epochs
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=2e-4, weight_decay=0.01)

# Train for 10-15 epochs with early stopping
# This alone should get you F1 ~0.82-0.84
```

This might actually work better than full fine-tuning given your small dataset!
