# =============================================================================
# persistence/feature_extractor.py - Transformer特征提取器
# =============================================================================
import torch
import numpy as np
from torch.nn import functional as F


class TransformerFeatureExtractor:
    """为Transformer准备激活序列特征"""
    def __init__(self, config):
        self.config = config
        self.device = config.device

    def extract_activation_sequences(self, models, dataloader):
        """从模型列表中提取激活序列"""
        print("🔍 提取激活序列特征...")

        all_sequences = []

        for i, model in enumerate(models):
            model = model.to(self.device)
            model.eval()

            sequence = self._extract_single_model_sequence(model, dataloader)
            all_sequences.append(sequence)

            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(models)} 个模型")

        sequences_array = np.array(all_sequences)
        print(f"✅ 提取完成! 序列形状: {sequences_array.shape}")

        return sequences_array

    def _extract_single_model_sequence(self, model, dataloader):
        """从单个模型提取激活序列"""
        layer_features = []

        # 为每个目标层提取特征
        for layer_name in self.config.selected_layers:
            layer_feature = self._extract_layer_feature(model, dataloader, layer_name)
            layer_features.append(layer_feature)

        # 组织成序列格式: [num_layers, feature_dim]
        sequence = np.stack(layer_features, axis=0)

        return sequence

    def _extract_layer_feature(self, model, dataloader, layer_name):
        """从指定层提取特征向量"""
        all_activations = []
        sample_count = 0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if sample_count >= self.config.num_samples:
                    break

                data = data.to(self.device)
                _ = model(data)  # 前向传播触发hook

                # 获取激活值
                activations = model.get_activations()

                if layer_name in activations:
                    activation = activations[layer_name]

                    # 处理不同层的激活形状
                    if len(activation.shape) == 4:  # Conv层: [batch, channel, height, width]
                        # 全局平均池化
                        pooled = F.adaptive_avg_pool2d(activation, (1, 1))
                        pooled = pooled.view(pooled.size(0), -1)  # [batch, channel]
                    elif len(activation.shape) == 2:  # FC层: [batch, features]
                        pooled = activation
                    else:
                        # 其他情况：展平后取平均
                        pooled = activation.view(activation.size(0), -1)

                    all_activations.append(pooled.cpu())
                    sample_count += data.size(0)

        if all_activations:
            # 合并所有批次
            concatenated = torch.cat(all_activations, dim=0)

            # 计算统计特征向量
            feature_vector = self._compute_comprehensive_features(concatenated)
        else:
            # 如果没有激活值，返回零向量
            feature_vector = np.zeros(self.config.raw_feature_dim)

        return feature_vector

    def _compute_comprehensive_features(self, activations):
        """计算全面的统计特征"""
        if isinstance(activations, torch.Tensor):
            activations = activations.numpy()

        # 处理异常值
        activations = np.nan_to_num(activations, nan=0.0, posinf=1e6, neginf=-1e6)

        if activations.shape[0] == 0:
            return np.zeros(self.config.raw_feature_dim)

        features = []

        try:
            # 1. 基础统计量 (8个特征)
            features.extend([
                np.mean(activations),                    # 全局均值
                np.std(activations),                     # 全局标准差
                np.max(activations),                     # 全局最大值
                np.min(activations),                     # 全局最小值
                np.median(activations),                  # 全局中位数
                np.var(activations),                     # 全局方差
                np.ptp(activations),                     # 峰值到峰值
                np.mean(np.abs(activations))             # 平均绝对值
            ])

            # 2. 分位数特征 (7个特征)
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            for p in percentiles:
                features.append(np.percentile(activations, p))

            # 3. 分布形状特征 (4个特征)
            from scipy import stats
            features.extend([
                stats.skew(activations.flatten()),       # 偏度
                stats.kurtosis(activations.flatten()),   # 峰度
                stats.entropy(np.histogram(activations, bins=50)[0] + 1e-10),  # 熵
                np.sqrt(np.mean((activations - np.mean(activations))**2))     # RMS
            ])

            # 4. 通道间特征 (如果是多通道)
            if len(activations.shape) > 1 and activations.shape[1] > 1:
                # 通道间相关性特征 (5个特征)
                channel_means = np.mean(activations, axis=0)
                channel_stds = np.std(activations, axis=0)

                features.extend([
                    np.mean(channel_means),              # 通道均值的均值
                    np.std(channel_means),               # 通道均值的标准差
                    np.mean(channel_stds),               # 通道标准差的均值
                    np.std(channel_stds),                # 通道标准差的标准差
                    np.mean(np.corrcoef(activations.T)) if activations.shape[1] > 1 else 0  # 平均相关系数
                ])

                # 主成分分析特征 (3个特征)
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(3, activations.shape[1]))
                    pca.fit(activations)
                    features.extend(pca.explained_variance_ratio_.tolist())
                    # 补齐到3个特征
                    while len(features) < 27:  # 8+7+4+5+3=27
                        features.append(0.0)
                except:
                    features.extend([0.0, 0.0, 0.0])
            else:
                # 单通道情况，用零填充
                features.extend([0.0] * 8)  # 5+3=8个零特征

            # 5. 激活模式特征 (5个特征)
            features.extend([
                np.sum(activations > 0) / activations.size,  # 激活率（ReLU后）
                np.sum(activations > np.mean(activations)) / activations.size,  # 高于均值的比例
                np.sum(np.abs(activations) < 1e-6) / activations.size,  # 接近零的比例
                np.mean(activations[activations > 0]) if np.any(activations > 0) else 0,  # 正激活均值
                np.mean(activations[activations < 0]) if np.any(activations < 0) else 0   # 负激活均值
            ])

        except Exception as e:
            print(f"特征计算错误: {e}")
            # 发生错误时返回基础特征
            features = [
                           np.mean(activations), np.std(activations), np.max(activations), np.min(activations)
                       ] + [0.0] * (self.config.raw_feature_dim - 4)

        # 确保特征数量正确
        if len(features) > self.config.raw_feature_dim:
            features = features[:self.config.raw_feature_dim]
        elif len(features) < self.config.raw_feature_dim:
            features.extend([0.0] * (self.config.raw_feature_dim - len(features)))

        # 最终数值稳定性检查
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        # 特征标准化（可选）
        if self.config.normalize_features:
            features = self._normalize_features(features)

        return features

    def _normalize_features(self, features):
        """特征标准化"""
        # 避免除零
        std = np.std(features)
        if std > 1e-8:
            features = (features - np.mean(features)) / std

        # 裁剪极值
        features = np.clip(features, -5, 5)

        return features

    def generate_sequences_for_models(self, benign_models, malicious_models, dataloader):
        """为良性和恶意模型生成激活序列"""
        print("🔄 生成模型激活序列...")

        # 提取良性模型序列
        print("处理良性模型...")
        benign_sequences = self.extract_activation_sequences(benign_models, dataloader)

        # 提取恶意模型序列
        print("处理恶意模型...")
        malicious_sequences = self.extract_activation_sequences(malicious_models, dataloader)

        print(f"✅ 序列生成完成!")
        print(f"良性序列形状: {benign_sequences.shape}")
        print(f"恶意序列形状: {malicious_sequences.shape}")

        return benign_sequences, malicious_sequences

    def extract_sequence_for_single_model(self, model, dataloader):
        """为单个模型提取序列（用于在线检测）"""
        sequence = self._extract_single_model_sequence(model, dataloader)
        return sequence.reshape(1, *sequence.shape)  # 添加batch维度


class SequenceDataGenerator:
    """激活序列数据生成器"""
    def __init__(self, extractor):
        self.extractor = extractor

    def generate_from_server(self, server, dataloader):
        """从联邦学习服务器生成序列数据"""
        print("📦 从联邦学习服务器收集模型...")

        benign_models, malicious_models = server.get_models_for_analysis()

        # 平衡样本数量
        min_samples = min(len(benign_models), len(malicious_models))
        if min_samples < 10:
            print(f"⚠️ 样本数量过少: 良性={len(benign_models)}, 恶意={len(malicious_models)}")

            # 如果样本太少，重复样本
            while len(benign_models) < 20:
                additional_samples = min(len(benign_models), 20 - len(benign_models))
                benign_models.extend(benign_models[:additional_samples])
            while len(malicious_models) < 20:
                additional_samples = min(len(malicious_models), 20 - len(malicious_models))
                malicious_models.extend(malicious_models[:additional_samples])

        # 限制样本数量
        benign_models = benign_models[:30]
        malicious_models = malicious_models[:30]

        print(f"使用模型: {len(benign_models)}个良性, {len(malicious_models)}个恶意")

        # 生成序列
        benign_sequences, malicious_sequences = self.extractor.generate_sequences_for_models(
            benign_models, malicious_models, dataloader
        )

        return benign_sequences, malicious_sequences