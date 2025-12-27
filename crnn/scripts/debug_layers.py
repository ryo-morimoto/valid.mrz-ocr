#!/usr/bin/env python3
"""
各レイヤーの backward 時間を測定
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import time

print("=== レイヤー別 Backward 時間測定 ===\n", flush=True)

# CNN 各ブロック単体テスト
print("[1] CNN Block 1 (Conv+BN+ReLU+Pool)", flush=True)
block1 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
)
x = torch.randn(4, 1, 32, 280, requires_grad=True)
out = block1(x)
print(f"  out: {out.shape}", flush=True)
t0 = time.time()
out.sum().backward()
print(f"  backward: {time.time()-t0:.3f}s\n", flush=True)

# 全CNN
print("[2] Full CNN (5 blocks)", flush=True)
cnn = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((2, 1)),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((2, 1)),
    nn.Conv2d(256, 256, kernel_size=(2, 1)),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
)
x = torch.randn(4, 1, 32, 280, requires_grad=True)
out = cnn(x)
print(f"  out: {out.shape}", flush=True)
t0 = time.time()
out.sum().backward()
print(f"  backward: {time.time()-t0:.3f}s\n", flush=True)

# CNN + reshape
print("[3] CNN + reshape", flush=True)
x = torch.randn(4, 1, 32, 280, requires_grad=True)
out = cnn(x)
out = out.squeeze(2).permute(2, 0, 1)  # (W', B, C)
print(f"  out: {out.shape}", flush=True)
t0 = time.time()
out.sum().backward()
print(f"  backward: {time.time()-t0:.3f}s\n", flush=True)

# CNN + LSTM (without dropout)
print("[4] CNN + LSTM (no dropout)", flush=True)
lstm = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=False, dropout=0.0)
x = torch.randn(4, 1, 32, 280, requires_grad=True)
features = cnn(x).squeeze(2).permute(2, 0, 1)
lstm_out, _ = lstm(features)
print(f"  lstm_out: {lstm_out.shape}", flush=True)
t0 = time.time()
lstm_out.sum().backward()
print(f"  backward: {time.time()-t0:.3f}s\n", flush=True)

# CNN + LSTM (with dropout)
print("[5] CNN + LSTM (dropout=0.1)", flush=True)
lstm_drop = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=False, dropout=0.1)
x = torch.randn(4, 1, 32, 280, requires_grad=True)
features = cnn(x).squeeze(2).permute(2, 0, 1)
lstm_out, _ = lstm_drop(features)
print(f"  lstm_out: {lstm_out.shape}", flush=True)
t0 = time.time()
lstm_out.sum().backward()
print(f"  backward: {time.time()-t0:.3f}s\n", flush=True)

print("✅ 完了", flush=True)
