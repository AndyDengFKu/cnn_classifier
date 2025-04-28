import torch
import dagshub
import mlflow
import mlflow.pytorch
import torch.nn.functional as F
import torchvision.models as models

from torch import nn
from pathlib import Path
from urllib.parse import urlparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        # 1. 构建验证集 DataLoader
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                         std=[0.229,0.224,0.225])
        valid_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.ToTensor(),
            normalize
        ])
        full_ds = datasets.ImageFolder(self.config.training_data,
                                       transform=valid_transform)
        val_size = int(0.2 * len(full_ds))
        _, valid_ds = random_split(full_ds,
                                   [len(full_ds)-val_size, val_size])
        self.valid_loader = DataLoader(valid_ds,
                                       batch_size=self.config.params_batch_size,
                                       shuffle=False,
                                       num_workers=4)

    @staticmethod
    def load_model(path: Path, num_classes: int, device):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        return model.to(device)

    def evaluate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(self.config.path_of_model,
                                     self.config.params_num_classes,
                                     device)
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in self.valid_loader:
                X, y = X.to(device), y.to(device)
                logits = self.model(X)
                loss = F.cross_entropy(logits, y, reduction='sum')
                total_loss += loss.item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        self.val_loss = total_loss / total
        self.val_acc = correct / total

    def save_score(self):
        scores = {"loss": self.val_loss, "accuracy": self.val_acc}
        save_json(path=Path("scores.json"), data=scores)

    def evaluation(self, upload_to_mlflow=True):
        self.evaluate()
        self.save_score()

        if not upload_to_mlflow:
            # 🌟 Local evaluation: print a clean result summary
            print("\n" + "="*30)
            print("📊 Local Evaluation Completed")
            print(f"✅ Validation Loss     : {self.val_loss:.4f}")
            print(f"✅ Validation Accuracy : {self.val_acc*100:.2f}%")
            print("="*30 + "\n")
            return
        
        dagshub.init(
            repo_owner='ZiqiDengZs',
            repo_name='cnn_classifier',
            mlflow=True
        )
        
        tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme



        with mlflow.start_run():
                self.evaluate()
                self.save_score()

                mlflow.log_params(self.config.all_params)
                mlflow.log_metric('val_loss', self.val_loss)
                mlflow.log_metric('val_accuracy', self.val_acc)

                mlflow.log_artifact("scores.json")

                if tracking_scheme != "file":
                    mlflow.pytorch.log_model(
                        self.model,
                        artifact_path="models",
                        registered_model_name="ResNet18Model"
                    )
                else:
                    mlflow.pytorch.log_model(
                        self.model,
                        artifact_path="models"
                    )