#!/usr/bin/env python3
"""
CTC Loss と backward pass の検証スクリプト
"""
import sys
import torch
import torch.nn as nn
from dataset import MRZDataset, collate_fn, NUM_CLASSES
from model import CRNN
from torch.utils.data import DataLoader
from pathlib import Path
import time

# 出力をバッファリングしない
sys.stdout.reconfigure(line_buffering=True)

print("=== CTC Loss / Backward Pass 検証 ===", flush=True)

# Setup
print("データ読み込み中...", flush=True)
train_dataset = MRZDataset(Path(__file__).parent.parent / "data/train", max_width=280, augment=False)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

print("モデル作成中...", flush=True)
model = CRNN(num_classes=NUM_CLASSES)
criterion = nn.CTCLoss(blank=NUM_CLASSES - 1, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Test 3 batches
print("\n3バッチをテスト:", flush=True)
for i, batch in enumerate(train_loader):
    if i >= 3:
        break

    images = batch["images"]
    labels = batch["labels"]
    label_lengths = batch["label_lengths"]

    print(f"\nBatch {i}:", flush=True)
    print(f"  images: {images.shape}", flush=True)
    print(f"  labels: {labels.shape}, lengths: {label_lengths.tolist()}", flush=True)

    # Forward
    t0 = time.time()
    outputs = model(images)
    t1 = time.time()
    print(f"  outputs: {outputs.shape} (fwd: {t1-t0:.3f}s)", flush=True)

    T, B, C = outputs.shape
    input_lengths = torch.full((B,), T, dtype=torch.long)
    print(f"  input_lengths: {input_lengths.tolist()}", flush=True)

    # Check: T > max(label_lengths)
    max_label = max(label_lengths.tolist())
    print(f"  T={T} > max_label={max_label}? {T > max_label}", flush=True)

    if T <= max_label:
        print("  ERROR: シーケンス長がラベル長より短い！", flush=True)
        sys.exit(1)

    # CTC Loss
    t2 = time.time()
    loss = criterion(outputs, labels, input_lengths, label_lengths)
    t3 = time.time()
    print(f"  loss: {loss.item():.4f} (ctc: {t3-t2:.3f}s)", flush=True)

    # Backward
    optimizer.zero_grad()
    t4 = time.time()
    loss.backward()
    t5 = time.time()
    print(f"  backward: {t5-t4:.3f}s", flush=True)

    optimizer.step()

print("\n✅ CTC Loss と backward pass は正常に動作", flush=True)
