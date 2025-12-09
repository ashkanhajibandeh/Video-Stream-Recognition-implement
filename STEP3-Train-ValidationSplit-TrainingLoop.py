import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


######################################
# load data outputs from stage1
######################################

X = np.load("dataset_generated/X300.npy")
Y = np.load("dataset_generated/Y300.npy")

split = int(len(X)*0.8)
Xtr,Ytr = X[:split],Y[:split]
Xvl,Yvl = X[split:],Y[split:]


######################################
# dataset + loader
######################################

class BitstreamDataset(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        self.X=X.astype(np.float32)
        self.Y=Y.astype(np.int64)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self,i):
        return torch.tensor(self.X[i]),torch.tensor(self.Y[i])

train = BitstreamDataset(Xtr,Ytr)
val   = BitstreamDataset(Xvl,Yvl)

train_loader = DataLoader(train,batch_size=64,shuffle=True)
val_loader   = DataLoader(val,batch_size=64,shuffle=False)


######################################
# model (same as phase2)
######################################

class CNNStream(nn.Module):
    def __init__(self,cls=3):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(2,32,7,padding=3),nn.ReLU(),
            nn.Conv1d(32,64,5,padding=2),nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128,cls)
    def forward(self,x):
        z=self.f(x).squeeze(-1)
        return self.fc(z)

model = CNNStream().to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device


######################################
# training configuration
######################################

opt  = Adam(model.parameters(),lr=1e-3)
loss = nn.CrossEntropyLoss()

epochs=15

history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}

best=0.0


######################################
# training loop
######################################

for ep in trange(epochs):
    model.train()
    t_loss=0;correct=0;total=0
    for xb,yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred  = model(xb)
        l = loss(pred,yb)
        opt.zero_grad(); l.backward(); opt.step()
        t_loss+=l.item()*xb.size(0)
        _,p = pred.max(1)
        correct += (p==yb).sum().item()
        total   += xb.size(0)
    tr_acc=correct/total
    tr_loss=t_loss/total
    
    model.eval()
    v_loss=0;vc=0;vt=0;allp=[];allt=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb = xb.to(device),yb.to(device)
            o=model(xb)
            l=loss(o,yb)
            v_loss+=l.item()*xb.size(0)
            _,p=o.max(1)
            vc+=(p==yb).sum().item()
            vt+=xb.size(0)
            allp.append(p.cpu().numpy())
            allt.append(yb.cpu().numpy())
    va=vc/vt
    v_loss/=vt
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(va)
    
    if va>best:
        best=va
        torch.save(model.state_dict(),"best_model.pth")

preds=np.concatenate(allp)
trues=np.concatenate(allt)


######################################
# save results
######################################

np.save("history.npy",history)
np.save("val_preds.npy",preds)
np.save("val_true.npy",trues)

plt.figure(figsize=(7,4))
plt.plot(history["train_acc"])
plt.plot(history["val_acc"])
plt.savefig("accuracy_curve.png")
plt.close()

with open("report.txt","w") as f:
    f.write(classification_report(trues,preds))
