import os
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import trange, tqdm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_has_sequence(length, chunk_period=10, chunk_size_mean=50000, jitter=0.2):
    seq = np.zeros(length, dtype=np.float32)
    t = np.arange(length)
    for start in range(0, length, chunk_period):
        size = max(100, int(np.random.normal(chunk_size_mean, chunk_size_mean*0.3)))
        end = min(length, start + 1 + int(np.random.normal(chunk_period*0.6, chunk_period*0.2)))
        if end<=start:
            end = min(length, start+1)
        distribution = np.abs(np.random.normal(1.0, jitter, end-start))
        distribution = distribution/np.sum(distribution)
        seq[start:end] += distribution * size
    noise = np.random.poisson(20, size=length).astype(np.float32)
    seq += noise
    return seq

def generate_other_sequence(length):
    seq = np.random.poisson(5, size=length).astype(np.float32)
    bursts = random.randint(1, max(1,length//30))
    for _ in range(bursts):
        center = random.randint(0, length-1)
        width = random.randint(1, min(30, length//10))
        amplitude = random.randint(500, 20000)
        start = max(0, center - width//2)
        end = min(length, center + width//2 + 1)
        window = np.hanning(end-start)
        seq[start:end] += amplitude * window
    return seq

def generate_start_screen(length):
    base = np.random.poisson(2, size=length).astype(np.float32)
    background = np.random.choice([0,1], p=[0.98,0.02], size=length).astype(np.float32) * np.random.poisson(50, size=length)
    return base + background

def make_sample(label, length):
    if label==0:
        d = generate_has_sequence(length)
        u = np.random.poisson(2, size=length).astype(np.float32)
    elif label==1:
        d = generate_other_sequence(length)
        u = np.random.poisson(3, size=length).astype(np.float32)
    else:
        d = generate_start_screen(length)
        u = np.random.poisson(1, size=length).astype(np.float32)
    return d, u

def create_dataset(num_per_class, length_seconds):
    X = []
    Y = []
    for cls in range(3):
        for _ in range(num_per_class):
            d,u = make_sample(cls, length_seconds)
            X.append(np.stack([d,u], axis=0))
            Y.append(cls)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    idx = np.arange(len(Y))
    np.random.shuffle(idx)
    return X[idx], Y[idx]

class BitstreamDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.Y[idx], dtype=torch.long)

class ConvNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels,32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def forward(self,x):
        return self.net(x)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb,yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        _, pcls = pred.max(dim=1)
        correct += (pcls==yb).sum().item()
        total += xb.size(0)
    return total_loss/total, correct/total

def eval_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
            _, pcls = pred.max(dim=1)
            preds.append(pcls.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = (preds==trues).mean()
    return total_loss/len(trues), acc, preds, trues

def plot_sample_sequences(X, Y, outdir, n=6):
    os.makedirs(outdir, exist_ok=True)
    ids = np.random.choice(len(Y), n, replace=False)
    fig, axs = plt.subplots(n,1, figsize=(8, 2*n), constrained_layout=True)
    for i,idx in enumerate(ids):
        seq = X[idx]
        axs[i].plot(seq[0], label='downlink')
        axs[i].plot(seq[1], label='uplink')
        axs[i].set_title(f'label={Y[idx]}')
        axs[i].legend()
    plt.savefig(os.path.join(outdir, 'sample_bitstreams.png'))
    plt.close(fig)

def compute_stats(X):
    down_mean = np.log1p(X[:,0,:].mean(axis=1))
    up_mean = np.log1p(X[:,1,:].mean(axis=1))
    down_std = np.log1p(X[:,0,:].std(axis=1))
    up_std = np.log1p(X[:,1,:].std(axis=1))
    return down_mean, up_mean, down_std, up_std

def plot_kde_stats(X, Y, outdir):
    os.makedirs(outdir, exist_ok=True)
    import seaborn as sns
    dm, um, ds, us = compute_stats(X)
    labels = Y
    fig, axs = plt.subplots(2,2, figsize=(10,8), constrained_layout=True)
    sns.kdeplot(dm[labels==0], ax=axs[0,0], label='video')
    sns.kdeplot(dm[labels!=0], ax=axs[0,0], label='other')
    axs[0,0].set_title('Downlink mean log')
    sns.kdeplot(ds[labels==0], ax=axs[0,1], label='video')
    sns.kdeplot(ds[labels!=0], ax=axs[0,1], label='other')
    axs[0,1].set_title('Downlink std log')
    sns.kdeplot(um[labels==0], ax=axs[1,0], label='video')
    sns.kdeplot(um[labels!=0], ax=axs[1,0], label='other')
    axs[1,0].set_title('Uplink mean log')
    sns.kdeplot(us[labels==0], ax=axs[1,1], label='video')
    sns.kdeplot(us[labels!=0], ax=axs[1,1], label='other')
    axs[1,1].set_title('Uplink std log')
    plt.savefig(os.path.join(outdir, 'kde_stats.png'))
    plt.close(fig)

def fundamental_frequency_cdf(X, outdir, sample_count=500):
    os.makedirs(outdir, exist_ok=True)
    lengths = []
    for i in range(min(len(X), sample_count)):
        seq = X[i,0,:] - X[i,0,:].mean()
        N = len(seq)
        yf = np.abs(rfft(seq))
        xf = rfftfreq(N, 1.0)
        peak = xf[np.argmax(yf[1:])+1]
        lengths.append(1.0/peak if peak>0 else 0)
    lengths = np.array(lengths)
    lengths = lengths[lengths>0]
    lengths_sorted = np.sort(lengths)
    cdf = np.arange(1,len(lengths_sorted)+1)/len(lengths_sorted)
    plt.figure(figsize=(6,4))
    plt.plot(lengths_sorted, cdf)
    plt.xlabel('Fundamental period (s)')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'fundamental_cdf.png'))
    plt.close()

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    X60,Y60 = create_dataset(num_per_class=1000, length_seconds=60)
    X300,Y300 = create_dataset(num_per_class=400, length_seconds=300)
    X600,Y600 = create_dataset(num_per_class=200, length_seconds=600)
    plot_sample_sequences(X300, Y300, args.outdir, n=6)
    plot_kde_stats(X300, Y300, args.outdir)
    fundamental_frequency_cdf(X300, args.outdir)
    X = X300
    Y = Y300
    split = int(0.8*len(Y))
    Xtr, Ytr = X[:split], Y[:split]
    Xte, Yte = X[split:], Y[split:]
    train_ds = BitstreamDataset(Xtr, Ytr)
    val_ds = BitstreamDataset(Xte, Yte)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet(in_channels=2, num_classes=3).to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_acc, preds, trues = eval_model(model, val_loader, loss_fn, device)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        if va_acc>best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best_model.pth'))
    plt.figure(figsize=(6,4))
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(args.outdir,'accuracy_curve.png'))
    plt.close()
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.savefig(os.path.join(args.outdir,'confusion_matrix.png'))
    plt.close()
    report = classification_report(trues, preds, target_names=['video','other','start'])
    with open(os.path.join(args.outdir,'classification_report.txt'),'w') as f:
        f.write(report)
    np.save(os.path.join(args.outdir,'history.npy'), history)
    np.save(os.path.join(args.outdir,'X300.npy'), X300)
    np.save(os.path.join(args.outdir,'Y300.npy'), Y300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results', help='output dir')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
