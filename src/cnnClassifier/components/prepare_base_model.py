import os
from pathlib import Path
from zipfile import ZipFile
import urllib.request as request

import torch
import torch.nn as nn
import torchvision.models as models
from cnnClassifier.utils.common import create_directories
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        create_directories([self.config.root_dir])

    def get_base_model(self):
        # 根据配置加载 ResNet-18，预训练权重（ImageNet）或随机初始化
        weights = None
        if self.config.params_weights.lower() == "imagenet":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)

        # 若不包含顶层，全连接层替换为空
        if not self.config.params_include_top:
            model.fc = nn.Identity()

        self.base_model = model
        # 保存基础模型的 state_dict
        self.save_model(self.config.base_model_path, model)

    def update_base_model(self):
        model = self.base_model
        # 冻结所有层参数
        for param in model.parameters():
            param.requires_grad = False

        # 构建新的全连接分类层
        # ResNet-18 默认 fc.in_features 为 512
        in_features = model.fc.in_features if hasattr(model.fc, "in_features") else 512
        model.fc = nn.Linear(in_features, self.config.params_classes)

        # 仅分类层参数可训练，并初始化优化器
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=self.config.params_learning_rate)

        self.full_model = model
        self.optimizer = optimizer

        # 保存更新后的模型 state_dict
        self.save_model(self.config.updated_base_model_path, model)

    @staticmethod
    def save_model(path: Path, model: torch.nn.Module):
        torch.save(model.state_dict(), path)


