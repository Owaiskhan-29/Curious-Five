%pip install mne optuna torch torchvision torchaudio scikit-learn tqdm

import os
import torch
import numpy as np
import mne
import optuna

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

DATA_ROOT = "C:/Users/manas/Downloads/Hackathon"   # change to your dataset path

EEG_DIR = os.path.join(DATA_ROOT, "EEG")

files = sorted([f for f in os.listdir(EEG_DIR) if f.endswith(".set")])

print("Total EEG files:", len(files))

def load_eeg(file_path):

    epochs = mne.read_epochs_eeglab(file_path, verbose=False)

    data = epochs.get_data()        # (trials, channels, time)
    events = epochs.events[:, -1]   # event ids 1–10

    labels = np.array([label_map[int(e)] for e in events])

    return data, labels
	
label_map = {
    1:0,   # Smooth

    2:1, 3:1,   # Acceleration

    4:2, 5:2, 6:2,   # Deceleration

    7:3, 8:3,   # Lane change

    9:4, 10:4   # Turning
}

X = []
y = []

for f in tqdm(files):

    path = os.path.join(EEG_DIR, f)

    data, labels = load_eeg(path)

    X.append(data)
    y.append(labels)

X = np.concatenate(X)
y = np.concatenate(y)

print("Dataset shape:", X.shape)

from collections import Counter

print("Class distribution:", Counter(y))

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y
)

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

print(class_weights)

X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-6)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(X_train.shape, X_test.shape)

class EEGDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
		
train_loader = DataLoader(
    EEGDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    EEGDataset(X_test, y_test),
    batch_size=64,
    shuffle=False,
    num_workers=0
)

torch.backends.cudnn.benchmark = True

class NASModel(nn.Module):

    def __init__(self, trial):

        super().__init__()

        layers = []

        in_channels = 59

        n_layers = trial.suggest_int("n_layers", 2, 5)

        for i in range(n_layers):

            out_channels = trial.suggest_categorical(
                f"channels_{i}", [16, 32, 64]
            )

            kernel = trial.suggest_categorical(
                f"kernel_{i}", [3,5,7]
            )

            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel, padding=kernel//2)
            )

            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())

            if trial.suggest_categorical(f"pool_{i}", [True, False]):
                layers.append(nn.MaxPool1d(2))

            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        hidden = trial.suggest_categorical("hidden", [64,128,256])

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(trial.suggest_float("dropout",0.1,0.5)),
            nn.Linear(hidden, 5)
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        return self.fc(x)
		
scaler = torch.cuda.amp.GradScaler()

def train_epoch(model, loader, optimizer, criterion):

    model.train()
    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)
	
def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y in loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            pred = out.argmax(1)

            correct += (pred == y).sum().item()

            total += y.size(0)

    return correct / total
	
def objective(trial):

    model = NASModel(trial).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(5):

        train_epoch(model, train_loader, optimizer, criterion)

    acc = evaluate(model, test_loader)

    return acc
	
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best accuracy:", study.best_value)
print("Best parameters:", study.best_params)

best_trial = study.best_trial

model = NASModel(best_trial).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(40):

    loss = train_epoch(model, train_loader, optimizer, criterion)

    acc = evaluate(model, test_loader)

    print(f"Epoch {epoch}: loss={loss:.4f} acc={acc:.4f}")
    

# ---- Improved EEG-specific NAS architecture (EEGNet-style) ----

class EEGNASModel(nn.Module):

    def __init__(self, trial):
        super().__init__()

        # temporal filter length
        temporal_kernel = trial.suggest_categorical("temporal_kernel", [16,32,64])

        # number of temporal filters
        F1 = trial.suggest_categorical("F1", [8,16,32])

        # depth multiplier (spatial filters)
        D = trial.suggest_categorical("depth_multiplier", [1,2,3])

        # separable conv filters
        F2 = trial.suggest_categorical("F2", [16,32,64])

        dropout = trial.suggest_float("dropout", 0.2, 0.5)

        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, temporal_kernel), padding=(0, temporal_kernel//2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1*D, (59,1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(dropout)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1,16), padding=(0,8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(F2*62, 5)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.temporal(x)

        x = self.spatial(x)

        x = self.separable(x)

        x = torch.flatten(x, start_dim=1)

        return self.classifier(x)


# ---- NAS objective for improved architecture ----

def eeg_objective(trial):

    model = EEGNASModel(trial).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(10):
        train_epoch(model, train_loader, optimizer, criterion)

    acc = evaluate(model, test_loader)

    return acc


# ---- run improved NAS search ----

eeg_study = optuna.create_study(direction="maximize")

eeg_study.optimize(eeg_objective, n_trials=100)

print("Best EEGNAS accuracy:", eeg_study.best_value)
print("Best EEGNAS parameters:", eeg_study.best_params)


# ---- train best EEGNAS model longer ----

best_trial = eeg_study.best_trial

model = EEGNASModel(best_trial).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(50):

    loss = train_epoch(model, train_loader, optimizer, criterion)

    acc = evaluate(model, test_loader)

    print(f"Epoch {epoch}: loss={loss:.4f} acc={acc:.4f}")