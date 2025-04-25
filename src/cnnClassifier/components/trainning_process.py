import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request as request

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from cnnClassifier.entity.config_entity import TrainingConfig



class Training:
    def __init__(self, config: TrainingConfig, device: Optional[torch.device] = None):
        self.config = config
        self.num_classes = config.params_num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_valid_generator(self):
        val_split = 0.2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transforms = [transforms.Resize(self.config.params_image_size[:2]),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20)] if self.config.params_is_augmentation else []
        train_transforms += [transforms.ToTensor(), normalize]
        valid_transforms = [transforms.Resize(self.config.params_image_size[:2]),
                            transforms.ToTensor(), normalize]

        full_dataset = datasets.ImageFolder(self.config.training_data,
                                           transform=transforms.Compose(train_transforms))
        total_size = len(full_dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        train_ds, valid_ds = random_split(full_dataset, [train_size, val_size])

        # Apply valid transforms to validation split
        valid_ds.dataset.transform = transforms.Compose(valid_transforms)

        self.train_loader = DataLoader(train_ds,
                                       batch_size=self.config.params_batch_size,
                                       shuffle=True,
                                       num_workers=4)
        self.valid_loader = DataLoader(valid_ds,
                                       batch_size=self.config.params_batch_size,
                                       shuffle=False,
                                       num_workers=4)
        print(f"Train dataset size: {len(train_ds)}")

    def get_base_model(self):
        model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        model.load_state_dict(torch.load(self.config.updated_base_model_path,
                                         map_location=self.device))
        model.to(self.device)
        self.model = model

    def train(self):
        self.train_valid_generator()
        self.get_base_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.config.params_learning_rate,
                              momentum=0.9,
                              weight_decay=1e-4)
        best_acc = 0.0

        # 外层用进度条显示 epoch 进度
        for epoch in tqdm(range(self.config.params_epochs), desc="Epochs"):
            self.model.train()
            running_loss, running_corrects = 0.0, 0

            # 内层用进度条显示 batch 进度
            for inputs, labels in tqdm(self.train_loader,
                                       desc=f"Train Epoch {epoch+1}",
                                       leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (outputs.argmax(1) == labels).sum().item()

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects / len(self.train_loader.dataset)

            # 验证阶段也可以加一个进度条
            self.model.eval()
            val_loss, val_corrects = 0.0, 0
            for inputs, labels in tqdm(self.valid_loader,
                                       desc=f"Valid Epoch {epoch+1}",
                                       leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()

            val_loss /= len(self.valid_loader.dataset)
            val_acc = val_corrects / len(self.valid_loader.dataset)

            # 打印每个 epoch 的汇总信息
            print(f"[Epoch {epoch+1}/{self.config.params_epochs}] "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} │ "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save final trained model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(path=self.config.trained_model_path, model=self.model)
                print(f"--> New best model saved (Val Acc: {best_acc:.4f})")
        
        print(f"Training complete. Best Val Acc: {best_acc:.4f}")

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Trained model saved to {path}")