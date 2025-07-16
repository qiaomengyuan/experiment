# =============================================================================
# detection/transformer_detector.py - Transformer异常检测器
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import math


class PositionalEncoding(nn.Module):
    """位置编码：表示网络层次关系"""
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ActivationTransformer(nn.Module):
    """基于激活序列的Transformer异常检测器"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.transformer_dim
        self.num_layers = len(config.selected_layers)

        # 特征投影层：将原始激活特征投影到统一维度
        self.feature_projection = nn.Linear(config.raw_feature_dim, self.feature_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(self.feature_dim, self.num_layers)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.transformer_dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )

        # 异常检测头
        self.anomaly_detector = nn.Sequential(
            nn.Linear(self.feature_dim, config.transformer_ff_dim),
            nn.GELU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(config.transformer_ff_dim, config.transformer_ff_dim // 2),
            nn.GELU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(config.transformer_ff_dim // 2, 1),
            nn.Sigmoid()
        )

        # 重构头（用于自监督预训练）
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.feature_dim, config.transformer_ff_dim),
            nn.GELU(),
            nn.Linear(config.transformer_ff_dim, config.raw_feature_dim)
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(self.feature_dim)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: [batch_size, num_layers, raw_feature_dim] 激活序列
        Returns:
            anomaly_scores: [batch_size, 1] 异常分数
        """
        batch_size, seq_len, _ = x.shape

        # 特征投影
        x = self.feature_projection(x)  # [batch, seq_len, feature_dim]
        x = self.layer_norm(x)

        # 添加位置编码
        x = x.transpose(0, 1)  # [seq_len, batch, feature_dim]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, feature_dim]

        # Transformer编码
        if return_attention:
            # 如果需要返回注意力权重（用于可视化分析）
            encoded, attention_weights = self.transformer(x, return_attention=True)
        else:
            encoded = self.transformer(x)  # [batch, seq_len, feature_dim]

        # 全局特征聚合（取平均）
        global_features = torch.mean(encoded, dim=1)  # [batch, feature_dim]

        # 异常检测
        anomaly_scores = self.anomaly_detector(global_features)  # [batch, 1]

        if return_attention:
            return anomaly_scores, attention_weights
        return anomaly_scores

    def reconstruct(self, x, mask_ratio=0.15):
        """自监督重构任务"""
        batch_size, seq_len, _ = x.shape

        # 随机掩码
        mask = torch.rand(batch_size, seq_len) < mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0  # 掩盖部分特征

        # 编码
        x_proj = self.feature_projection(x_masked)
        x_proj = self.layer_norm(x_proj)
        x_proj = x_proj.transpose(0, 1)
        x_proj = self.pos_encoding(x_proj)
        x_proj = x_proj.transpose(0, 1)

        encoded = self.transformer(x_proj)

        # 重构
        reconstructed = self.reconstruction_head(encoded)

        return reconstructed, mask


class UnsupervisedTransformerDetector:
    """无监督Transformer检测器训练和推理"""
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 初始化模型
        self.model = ActivationTransformer(config).to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.transformer_lr,
            weight_decay=config.transformer_weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.transformer_epochs
        )

        # 损失函数
        self.reconstruction_loss = nn.MSELoss()

        # 异常检测阈值（训练后确定）
        self.anomaly_threshold = None

    def pretrain_reconstruction(self, activation_sequences):
        """自监督预训练：重构任务"""
        print("🎯 开始自监督预训练...")

        # 准备数据（只使用良性样本进行预训练）
        X_tensor = torch.FloatTensor(activation_sequences).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.transformer_batch_size,
            shuffle=True
        )

        self.model.train()

        for epoch in range(self.config.pretrain_epochs):
            epoch_loss = 0

            for batch_x, in dataloader:
                self.optimizer.zero_grad()

                # 重构任务
                reconstructed, mask = self.model.reconstruct(
                    batch_x, self.config.mask_ratio
                )

                # 只计算被掩盖位置的重构损失
                loss = self.reconstruction_loss(
                    reconstructed[mask],
                    batch_x[mask]
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"预训练 Epoch {epoch + 1}: 重构损失 = {avg_loss:.6f}")

        print("✅ 预训练完成!")

    def train_anomaly_detector(self, benign_sequences, malicious_sequences):
        """训练异常检测器（使用少量标注数据或完全无监督）"""
        print("🛡️ 训练异常检测器...")

        if self.config.unsupervised_only:
            # 完全无监督：只使用良性样本学习正常模式
            return self._train_fully_unsupervised(benign_sequences)
        else:
            # 半监督：使用少量标注数据
            return self._train_semi_supervised(benign_sequences, malicious_sequences)

    def _train_fully_unsupervised(self, benign_sequences):
        """完全无监督训练"""
        print("📊 使用完全无监督方法...")

        # 只使用良性样本
        X_tensor = torch.FloatTensor(benign_sequences).to(self.device)

        # 计算正常样本的异常分数分布
        self.model.eval()
        with torch.no_grad():
            anomaly_scores = []
            for i in range(0, len(X_tensor), self.config.transformer_batch_size):
                batch = X_tensor[i:i+self.config.transformer_batch_size]
                scores = self.model(batch)
                anomaly_scores.append(scores.cpu().numpy())

            anomaly_scores = np.concatenate(anomaly_scores, axis=0)

        # 设置阈值：正常样本95%分位数
        self.anomaly_threshold = np.percentile(anomaly_scores, 95)

        print(f"异常检测阈值设定为: {self.anomaly_threshold:.4f}")

        # 返回虚拟指标（因为没有真实标签）
        return {
            'threshold': self.anomaly_threshold,
            'benign_scores_mean': np.mean(anomaly_scores),
            'benign_scores_std': np.std(anomaly_scores)
        }

    def _train_semi_supervised(self, benign_sequences, malicious_sequences):
        """半监督训练（使用少量标注数据优化阈值）"""
        print("📊 使用半监督方法...")

        # 合并数据
        X = np.concatenate([benign_sequences, malicious_sequences], axis=0)
        y = np.concatenate([
            np.zeros(len(benign_sequences)),
            np.ones(len(malicious_sequences))
        ])

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)

        # 获取异常分数
        self.model.eval()
        with torch.no_grad():
            train_scores = self.model(X_train_tensor).cpu().numpy()
            test_scores = self.model(X_test_tensor).cpu().numpy()

        # 寻找最优阈值
        best_threshold = self._find_optimal_threshold(train_scores, y_train)
        self.anomaly_threshold = best_threshold

        # 评估性能
        test_predictions = (test_scores > best_threshold).astype(int)
        metrics = {
            'accuracy': accuracy_score(y_test, test_predictions),
            'precision': precision_score(y_test, test_predictions),
            'recall': recall_score(y_test, test_predictions),
            'f1_score': f1_score(y_test, test_predictions),
            'threshold': best_threshold
        }

        print(f"最优阈值: {best_threshold:.4f}")
        print(f"测试准确率: {metrics['accuracy']:.4f}")
        print(f"测试F1分数: {metrics['f1_score']:.4f}")

        return metrics

    def _find_optimal_threshold(self, scores, labels):
        """寻找最优异常检测阈值"""
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        best_f1 = 0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            f1 = f1_score(labels, predictions)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def detect_anomalies(self, activation_sequences):
        """检测异常（推理阶段）"""
        self.model.eval()

        X_tensor = torch.FloatTensor(activation_sequences).to(self.device)

        with torch.no_grad():
            anomaly_scores = self.model(X_tensor).cpu().numpy()

        # 二值化结果
        predictions = (anomaly_scores > self.anomaly_threshold).astype(int)

        return {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'threshold': self.anomaly_threshold
        }

    def get_attention_analysis(self, activation_sequences):
        """获取注意力权重分析（可解释性）"""
        self.model.eval()

        X_tensor = torch.FloatTensor(activation_sequences).to(self.device)

        with torch.no_grad():
            anomaly_scores, attention_weights = self.model(
                X_tensor, return_attention=True
            )

        return {
            'anomaly_scores': anomaly_scores.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy(),
            'layer_names': self.config.selected_layers
        }

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.anomaly_threshold,
            'config': self.config
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.anomaly_threshold = checkpoint['threshold']