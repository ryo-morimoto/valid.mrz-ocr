#!/usr/bin/env python3
"""
backward ハングの原因切り分け
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import time

print("=== Backward ハング原因調査 ===", flush=True)

# Test 1: 単純なテンソルの backward
print("\n[Test 1] 単純なテンソル backward", flush=True)
x = torch.randn(100, 100, requires_grad=True)
y = (x ** 2).sum()
t0 = time.time()
y.backward()
print(f"  完了: {time.time()-t0:.3f}s", flush=True)

# Test 2: Linear層の backward
print("\n[Test 2] Linear層 backward", flush=True)
linear = nn.Linear(256, 38)
x = torch.randn(70, 4, 256)
y = linear(x).sum()
t0 = time.time()
y.backward()
print(f"  完了: {time.time()-t0:.3f}s", flush=True)

# Test 3: LSTM の backward
print("\n[Test 3] BiLSTM backward", flush=True)
lstm = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=False)
x = torch.randn(70, 4, 256)
out, _ = lstm(x)
loss = out.sum()
t0 = time.time()
loss.backward()
print(f"  完了: {time.time()-t0:.3f}s", flush=True)

# Test 4: CTC Loss の backward
print("\n[Test 4] CTC Loss backward", flush=True)
ctc = nn.CTCLoss(blank=37, zero_infinity=True)
log_probs = torch.randn(70, 4, 38).log_softmax(dim=2)
targets = torch.randint(0, 37, (4 * 44,))
input_lengths = torch.full((4,), 70, dtype=torch.long)
target_lengths = torch.full((4,), 44, dtype=torch.long)

log_probs.requires_grad = True
loss = ctc(log_probs, targets, input_lengths, target_lengths)
print(f"  loss: {loss.item():.4f}", flush=True)
t0 = time.time()
loss.backward()
print(f"  完了: {time.time()-t0:.3f}s", flush=True)

# Test 5: CNN + LSTM 組み合わせ
print("\n[Test 5] CNN + BiLSTM backward", flush=True)
from model import CRNN
from dataset import NUM_CLASSES

model = CRNN(num_classes=NUM_CLASSES)
x = torch.randn(4, 1, 32, 280)
out = model(x)
loss = out.sum()
t0 = time.time()
loss.backward()
print(f"  完了: {time.time()-t0:.3f}s", flush=True)

# Test 6: フルパイプライン（CTC含む）
print("\n[Test 6] フルパイプライン (CRNN + CTC Loss)", flush=True)
model = CRNN(num_classes=NUM_CLASSES)
ctc = nn.CTCLoss(blank=NUM_CLASSES-1, zero_infinity=True)
x = torch.randn(4, 1, 32, 280)
out = model(x)  # (70, 4, 38)
T, B, C = out.shape

targets = torch.randint(0, NUM_CLASSES-1, (B * 44,))
input_lengths = torch.full((B,), T, dtype=torch.long)
target_lengths = torch.full((B,), 44, dtype=torch.long)

loss = ctc(out, targets, input_lengths, target_lengths)
print(f"  loss: {loss.item():.4f}", flush=True)
t0 = time.time()
loss.backward()
print(f"  完了: {time.time()-t0:.3f}s", flush=True)

print("\n✅ 全テスト完了", flush=True)
