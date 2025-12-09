import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


###########################################
# 1) Load Windows Created in Stage 1
###########################################

X60  = np.load("dataset_generated/X60.npy")
Y60  = np.load("dataset_generated/Y60.npy")

X300 = np.load("dataset_generated/X300.npy")
Y300 = np.load("dataset_generated/Y300.npy")

X600 = np.load("dataset_generated/X600.npy")
Y600 = np.load("dataset_generated/Y600.npy")


###########################################
# 2) PyTorch Dataset Definition
###########################################

class BitstreamDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.int64)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])     # shape [2,window]
        y = torch.tensor(self.Y[idx])
        return x, y



###########################################
# 3) Create Train / Validation Loaders
###########################################

def make_loaders(X,Y,batch=32,ratio=0.8):
    n = int(len(X) * ratio)
    X_train, Y_train = X[:n], Y[:n]
    X_val  , Y_val  = X[n:], Y[n:]

    train = BitstreamDataset(X_train,Y_train)
    val   = BitstreamDataset(X_val,Y_val)

    train_loader = DataLoader(train, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val  , batch_size=batch, shuffle=False)

    return train_loader, val_loader



###########################################
# 4) CNN Model (Two Input Channels)
###########################################

class CNNStream(nn.Module):
    def __init__(self, classes=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32,64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv1d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, classes)

    def forward(self,x):
        z = self.conv(x)
        z = z.squeeze(-1)
        return self.fc(z)



###########################################
# 5) Instantiate Model + Loaders
###########################################

train_loader_300 , val_loader_300  = make_loaders(X300,Y300,batch=64)
train_loader_60  , val_loader_60   = make_loaders(X60 ,Y60 ,batch=64)
train_loader_600 , val_loader_600  = make_loaders(X600,Y600,batch=64)

model = CNNStream(classes=3)
print("Model Ready ✔")
print(model)
