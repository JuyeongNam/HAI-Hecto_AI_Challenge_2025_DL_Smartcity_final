import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss

# 하이퍼파라미터
num_classes = 396
batch_size = 16
epochs = 10
lr = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 설정
base_path = "C:/Users/fbwod/Desktop/DL_Oldcar"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")

# ✅ val 폴더 생성 (한 번만 실행됨)
if not os.path.exists(val_path):
    print("🔧 val/ 폴더 생성 중 (train에서 10% 분리)...")
    class_names = os.listdir(train_path)
    image_paths, labels = [], []
    for cls in class_names:
        cls_path = os.path.join(train_path, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))]
        for f in files:
            image_paths.append(os.path.join(cls_path, f))
            labels.append(cls)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    _, val_idx = next(splitter.split(image_paths, labels))

    for i in val_idx:
        src = image_paths[i]
        cls = labels[i]
        dst_dir = os.path.join(val_path, cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))
    print("✅ val 데이터 분할 완료.")

# 전처리 정의
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 데이터셋 로딩
train_dataset = ImageFolder(train_path, transform=train_transform)
val_dataset = ImageFolder(val_path, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의 (ResNet18)
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# 손실함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 학습 루프
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_acc = correct / total

    # 🔍 Validation Accuracy + Log Loss
    model.eval()
    val_correct = 0
    val_total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    val_logloss = log_loss(all_labels, all_probs, labels=np.arange(num_classes))

    print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val LogLoss: {val_logloss:.4f}, Loss: {running_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), "resnet18_finetuned.pth")
print("✅ 모델 저장 완료: resnet18_finetuned.pth")
