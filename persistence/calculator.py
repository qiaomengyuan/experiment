# =============================================================================
# persistence/calculator.py  持久同调计算
# =============================================================================
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


class PersistenceCalculator:
    def __init__(self, config):
        self.config = config

    def extract_activations(self, model, dataloader):
        model.eval()
        all_activations = {layer: [] for layer in self.config.selected_layers}

        with torch.no_grad():
            sample_count = 0
            for batch_idx, (data, _) in enumerate(dataloader):
                if sample_count >= self.config.num_samples:
                    break

                data = data.to(self.config.device)
                _ = model(data)

                activations = model.get_activations()
                for layer_name in self.config.selected_layers:
                    if layer_name in activations:
                        # 取每个样本的激活值均值
                        act = activations[layer_name]
                        if len(act.shape) == 4:  # [batch, channel, height, width]
                            act_mean = torch.mean(act, dim=[2, 3])  # 空间维度求平均
                        else:
                            act_mean = act  # 如果已经是2D，直接使用
                        all_activations[layer_name].append(act_mean.cpu())

                sample_count += data.size(0)

        # 合并批次
        for layer_name in all_activations:
            if all_activations[layer_name]:
                all_activations[layer_name] = torch.cat(all_activations[layer_name], dim=0)

        return all_activations

    def compute_robust_features(self, activations):
        """计算数值稳定的统计特征"""
        if isinstance(activations, torch.Tensor):
            activations = activations.numpy()

        # 检查并处理异常值
        activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)

        if activations.shape[0] == 0 or activations.shape[1] == 0:
            return np.zeros(50)  # 返回固定长度的零向量

        features = []

        try:
            # 基本统计量
            mean_vals = np.mean(activations, axis=0)
            std_vals = np.std(activations, axis=0) + 1e-8  # 避免除零
            max_vals = np.max(activations, axis=0)
            min_vals = np.min(activations, axis=0)

            # 确保没有NaN或Inf
            mean_vals = np.nan_to_num(mean_vals, nan=0.0)
            std_vals = np.nan_to_num(std_vals, nan=1e-8)
            max_vals = np.nan_to_num(max_vals, nan=0.0)
            min_vals = np.nan_to_num(min_vals, nan=0.0)

            # 取前10个特征（如果不够就用0填充）
            for arr in [mean_vals, std_vals, max_vals, min_vals]:
                if len(arr) >= 10:
                    features.extend(arr[:10])
                else:
                    features.extend(arr.tolist() + [0.0] * (10 - len(arr)))

            # 添加更多简单特征
            features.append(np.mean(activations))  # 全局均值
            features.append(np.std(activations))  # 全局标准差
            features.append(np.max(activations))  # 全局最大值
            features.append(np.min(activations))  # 全局最小值

            # 分位数特征
            try:
                percentiles = np.percentile(activations.flatten(), [25, 50, 75, 90, 95])
                features.extend(percentiles.tolist())
            except:
                features.extend([0.0] * 5)

            # 形状特征
            features.append(float(activations.shape[0]))  # 样本数
            features.append(float(activations.shape[1]))  # 特征数

        except Exception as e:
            print(f"特征计算警告: {e}")
            features = [0.0] * 50

        # 确保返回固定长度
        if len(features) > 50:
            features = features[:50]
        else:
            features.extend([0.0] * (50 - len(features)))

        # 最终检查
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def generate_diagram_for_model(self, model, dataloader):
        """为单个模型生成特征向量"""
        try:
            activations = self.extract_activations(model, dataloader)
            layer_features = []

            for layer_name in self.config.selected_layers:
                if layer_name in activations and len(activations[layer_name]) > 0:
                    features = self.compute_robust_features(activations[layer_name])
                else:
                    features = np.zeros(50)
                layer_features.append(features)

            # 将所有层的特征组织成"图像"格式
            all_features = np.concatenate(layer_features)

            # 组织成多通道图像格式
            features_per_channel = self.config.grid_size * self.config.grid_size
            result = []

            for i in range(len(self.config.selected_layers)):
                start_idx = i * 50
                end_idx = start_idx + 50

                if end_idx <= len(all_features):
                    layer_feat = all_features[start_idx:end_idx]
                else:
                    layer_feat = np.zeros(50)

                # 调整到grid_size x grid_size
                if len(layer_feat) > features_per_channel:
                    layer_feat = layer_feat[:features_per_channel]
                else:
                    layer_feat = np.pad(layer_feat, (0, features_per_channel - len(layer_feat)))

                # 重塑为图像
                layer_img = layer_feat.reshape(self.config.grid_size, self.config.grid_size)

                # 归一化到[0,1]
                if np.max(layer_img) > np.min(layer_img):
                    layer_img = (layer_img - np.min(layer_img)) / (np.max(layer_img) - np.min(layer_img))

                result.append(layer_img)

            final_result = np.stack(result, axis=0)

            # 最终安全检查
            final_result = np.nan_to_num(final_result, nan=0.0, posinf=0.0, neginf=0.0)

            return final_result

        except Exception as e:
            print(f"模型特征生成失败: {e}")
            # 返回安全的默认值
            return np.zeros((len(self.config.selected_layers), self.config.grid_size, self.config.grid_size))
