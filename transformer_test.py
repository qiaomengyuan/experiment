import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SimpleTransformerDetector(nn.Module):
    """简化的Transformer序列检测器"""

    def __init__(self,
                 seq_length=100,  # 序列长度
                 input_dim=64,  # 输入特征维度
                 d_model=128,  # Transformer隐藏维度
                 nhead=4,  # 注意力头数
                 num_layers=2,  # Transformer层数
                 dim_feedforward=256,  # 前馈网络维度
                 dropout=0.1):
        super().__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.d_model = d_model

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, seq_length)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 二分类：正常/异常
        )

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_dim) 序列数据
        Returns:
            logits: (batch_size, 2) 分类结果
        """
        batch_size, seq_len, _ = x.shape

        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer编码
        x = self.transformer(x)  # (batch_size, seq_length, d_model)

        # 全局平均池化
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_length)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)

        # 分类
        logits = self.classifier(x)  # (batch_size, 2)

        return logits


class ParameterSequenceDataset(Dataset):
    """模型参数序列数据集"""

    def __init__(self, sequences, labels):
        """
        Args:
            sequences: numpy array, shape (num_samples, seq_length, feature_dim)
            labels: numpy array, shape (num_samples,) 0=正常, 1=异常
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_detector(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda'):
    """训练检测器"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_val_acc = 0
    train_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                logits = model(sequences)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_detector.pth')

        scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Acc: {val_acc:.4f}')
            print(f'  Best Val Acc: {best_val_acc:.4f}')

    return train_losses, val_accs


def evaluate_detector(model, test_loader, device='cuda'):
    """评估检测器"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences)

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions,
                                   target_names=['Normal', 'Malicious'])

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{cm}')
    print(f'Classification Report:\n{report}')

    return accuracy, cm, report


# 使用示例
if __name__ == "__main__":
    # 模拟数据生成
    def generate_sample_data():
        """生成示例数据"""
        seq_length = 100
        feature_dim = 64
        num_normal = 1000
        num_malicious = 200

        # 正常序列：参数变化相对平稳
        normal_sequences = []
        for _ in range(num_normal):
            base = np.random.randn(seq_length, feature_dim) * 0.1
            # 添加一些渐变趋势
            trend = np.linspace(0, 0.05, seq_length).reshape(-1, 1)
            sequence = base + trend
            normal_sequences.append(sequence)

        # 恶意序列：包含异常模式
        malicious_sequences = []
        for _ in range(num_malicious):
            base = np.random.randn(seq_length, feature_dim) * 0.1
            # 在随机位置插入异常峰值
            anomaly_pos = np.random.randint(10, seq_length - 10)
            base[anomaly_pos:anomaly_pos + 5] += np.random.randn(5, feature_dim) * 0.5
            malicious_sequences.append(base)

        # 合并数据
        all_sequences = np.array(normal_sequences + malicious_sequences)
        all_labels = np.array([0] * num_normal + [1] * num_malicious)

        # 打乱数据
        indices = np.random.permutation(len(all_sequences))
        return all_sequences[indices], all_labels[indices]


    # 生成数据
    sequences, labels = generate_sample_data()
    print(f"数据形状: {sequences.shape}")
    print(f"标签分布: 正常={np.sum(labels == 0)}, 恶意={np.sum(labels == 1)}")

    # 数据分割
    train_size = int(0.7 * len(sequences))
    val_size = int(0.15 * len(sequences))

    train_sequences = sequences[:train_size]
    train_labels = labels[:train_size]

    val_sequences = sequences[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]

    test_sequences = sequences[train_size + val_size:]
    test_labels = labels[train_size + val_size:]

    # 创建数据集和数据加载器
    train_dataset = ParameterSequenceDataset(train_sequences, train_labels)
    val_dataset = ParameterSequenceDataset(val_sequences, val_labels)
    test_dataset = ParameterSequenceDataset(test_sequences, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleTransformerDetector(
        seq_length=100,
        input_dim=64,
        d_model=128,
        nhead=4,
        num_layers=2
    )

    print(f"使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("\n开始训练...")
    train_losses, val_accs = train_detector(
        model, train_loader, val_loader,
        num_epochs=30, lr=1e-3, device=device
    )

    # 加载最佳模型并测试
    model.load_state_dict(torch.load('best_detector.pth'))
    model = model.to(device)

    print("\n最终测试结果:")
    accuracy, cm, report = evaluate_detector(model, test_loader, device)
