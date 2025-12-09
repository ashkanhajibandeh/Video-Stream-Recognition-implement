import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# =============================
# 1)  Synthetic Traffic Generators
# =============================

def gen_video_has(length, chunk_period=10, avg=60000):
    x = np.zeros(length, dtype=np.float32)
    t = np.arange(length)

    for start in range(0, length, chunk_period):
        chunk = np.random.normal(avg, avg*0.35)
        if chunk < 500: chunk = 500
        
        end = start + np.random.randint(int(chunk_period*0.6), int(chunk_period*1.4))
        end = min(end, length)

        dist = np.abs(np.random.normal(1, 0.2, end-start))
        dist = dist / dist.sum()
        x[start:end] = dist * chunk

    x += np.random.poisson(20, len(x))  
    return x


def gen_other_app(length):
    x = np.random.poisson(8, length).astype(np.float32)
    for _ in range(np.random.randint(1,4)):
        c = np.random.randint(0, length-1)
        w = np.random.randint(3,25)
        amp = np.random.randint(300,20000)
        s = max(0, c-w//2)
        e = min(length, c+w//2)
        x[s:e] += np.hanning(e-s) * amp
    return x


def gen_background(length):
    base = np.random.poisson(2, length).astype(np.float32)
    occasional = np.random.choice([0,1], p=[0.98,0.02], size=length) * np.random.poisson(50, length)
    return base + occasional


# =============================
# 2) window creator
# =============================

def make_sample(label, length):
    if label==0:
        d = gen_video_has(length)
        u = np.random.poisson(3, length)
    elif label==1:
        d = gen_other_app(length)
        u = np.random.poisson(4, length)
    else:
        d = gen_background(length)
        u = np.random.poisson(1, length)

    return np.stack([d,u], axis=0)  # shape: [2,length]


# =============================
# 3) Multi-app generation (for removal like paper)
# =============================

def is_multi_app(signal, threshold_ratio=0.65):
    d = signal[0]
    peaks = (d > d.mean()*3).sum()
    return peaks/len(d) < threshold_ratio   # اگر رفتار چنک واضح نباشد → حذف


# =============================
# 4) Generate windows dataset
# =============================

def build_dataset(count_per_class=800, win=300):
    X, Y = [], []
    
    for label in [0,1,2]:
        for _ in range(count_per_class):
            x = make_sample(label, win)
            
            if is_multi_app(x): 
                continue   # حذف مثل مقاله
            
            X.append(x)
            Y.append(label)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    return X,Y


# =============================
# 5) Run Stage-1
# =============================

if __name__ == "__main__":
    
    os.makedirs("dataset_generated", exist_ok=True)

    X60 ,Y60  = build_dataset(1200, 60)
    X300,Y300 = build_dataset(600 ,300)
    X600,Y600 = build_dataset(300 ,600)

    np.save("dataset_generated/X60.npy",X60)
    np.save("dataset_generated/Y60.npy",Y60)
    np.save("dataset_generated/X300.npy",X300)
    np.save("dataset_generated/Y300.npy",Y300)
    np.save("dataset_generated/X600.npy",X600)
    np.save("dataset_generated/Y600.npy",Y600)

    print("Done — Synthetic dataset ready ✔")
    print("Shapes →", X60.shape, X300.shape, X600.shape)
