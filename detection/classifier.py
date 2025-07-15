# =============================================================================
# detection/classifier.py  检测分类器
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, grid_size):
        super().__init__()

        # 更简单但更有效的网络结构
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)  # 自适应池化到4x4
        )

        # 计算特征大小
        feature_size = 64 * 4 * 4

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class PDClassifier:
    def __init__(self, config):
        self.config = config
        self.model = SimpleCNN(len(config.selected_layers), config.grid_size)
        self.model.to(config.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.detector_lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train_classifier(self, benign_diagrams, malicious_diagrams):
        print(f"训练数据: {len(benign_diagrams)}个良性, {len(malicious_diagrams)}个恶意")

        # 检查数据形状
        print(f"良性数据形状: {benign_diagrams.shape}")
        print(f"恶意数据形状: {malicious_diagrams.shape}")

        # 准备数据
        X = np.concatenate([benign_diagrams, malicious_diagrams], axis=0)
        y = np.concatenate([np.zeros(len(benign_diagrams)), np.ones(len(malicious_diagrams))])

        # 数据增强 - 添加噪声
        noise_std = 0.01
        X_noise = X + np.random.normal(0, noise_std, X.shape)
        X = np.concatenate([X, X_noise], axis=0)
        y = np.concatenate([y, y], axis=0)

        print(f"增强后数据量: {len(X)}")

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.config.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.config.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.config.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.config.device)

        # 训练
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.detector_batch_size, shuffle=True)

        self.model.train()
        best_test_acc = 0

        for epoch in range(self.config.detector_epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step()

            if (epoch + 1) % 20 == 0:
                train_acc = self._evaluate(X_train_tensor, y_train_tensor)
                test_acc = self._evaluate(X_test_tensor, y_test_tensor)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        # 最终评估
        final_metrics = self._detailed_evaluation(X_test_tensor, y_test_tensor)
        print(f"最佳测试准确率: {best_test_acc:.4f}")
        return final_metrics

    def _evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean().item()
        self.model.train()
        return accuracy

    def _detailed_evaluation(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)

            y_true = y.cpu().numpy()
            y_pred = predicted.cpu().numpy()

            # 打印混淆矩阵信息
            tn = np.sum((y_true == 0) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))

            print(f"混淆矩阵: TN={tn}, TP={tp}, FN={fn}, FP={fp}")

            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }

        self.model.train()
        return metrics
