# =============================================================================
# federated/server.py
# =============================================================================

# federated/server.py
import torch
import numpy as np
from .client import FederatedClient
from .aggregator import FedAvgAggregator


class FederatedServer:
    def __init__(self, fed_dataset, config, attacker_class):
        self.config = config
        self.fed_dataset = fed_dataset
        self.attacker_class = attacker_class
        self.aggregator = FedAvgAggregator()

        # 🔧 修复：先初始化attack_stats
        self.attack_stats = {
            'malicious_clients': [],
            'attack_type': config.attack_type,
            'total_rounds': 0
        }

        # 然后创建客户端（会用到attack_stats）
        self.clients = self._create_clients()
        self.global_model = self._init_global_model()
        self.model_history = []

    def _create_clients(self):
        clients = []
        malicious_ids = np.random.choice(self.config.num_clients,
                                         self.config.num_malicious, replace=False)

        print(f"恶意客户端ID: {malicious_ids}")
        print(f"攻击类型: {self.config.attack_type}")

        for i in range(self.config.num_clients):
            dataset = self.fed_dataset.get_client_data(i)
            is_malicious = i in malicious_ids

            # 为每个恶意客户端创建专门的攻击器
            attacker = None
            if is_malicious:
                if self.config.attack_type == "distributed":
                    # 分布式攻击需要传递客户端ID
                    from attack.backdoor import DistributedBackdoorAttack
                    attacker = DistributedBackdoorAttack(self.config, client_id=i)
                else:
                    # 其他攻击类型
                    attacker = self.attacker_class(self.config)

                self.attack_stats['malicious_clients'].append(i)

            client = FederatedClient(i, dataset, self.config, is_malicious, attacker)
            clients.append(client)

        return clients

    def _init_global_model(self):
        from models.resnet import ResNet20
        model = ResNet20(self.config.num_classes)
        return model.state_dict()

    def train(self):
        print(f"开始{self.config.attack_type}攻击的联邦学习训练...")

        for round_num in range(self.config.num_rounds):
            # 选择客户端
            selected_clients = np.random.choice(self.clients,
                                                self.config.clients_per_round, replace=False)

            # 统计本轮恶意客户端
            round_malicious = sum(1 for client in selected_clients if client.is_malicious)

            # 收集本地更新
            client_models = []
            for client in selected_clients:
                model_state = client.local_train(self.global_model)
                client_models.append({
                    'model': model_state,
                    'is_malicious': client.is_malicious,
                    'client_id': client.client_id,
                    'attack_type': self.config.attack_type if client.is_malicious else None
                })

            # 应用模型替换攻击（针对恶意客户端）
            for client_info in client_models:
                if client_info['is_malicious']:
                    client_info['model'] = self._apply_enhanced_model_replacement(
                        client_info['model'], client_info['attack_type']
                    )

            # 聚合模型
            model_states = [info['model'] for info in client_models]
            self.global_model = self.aggregator.aggregate(model_states)

            # 保存模型历史
            self.model_history.append(client_models)
            self.attack_stats['total_rounds'] += 1

            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}/{self.config.num_rounds} completed")
                print(f"  本轮恶意客户端数: {round_malicious}/{len(selected_clients)}")

                # 中期评估
                if (round_num + 1) % 20 == 0:
                    self._mid_training_evaluation(round_num + 1)

    def _apply_enhanced_model_replacement(self, model_state, attack_type):
        """增强的模型替换攻击"""
        scaled_state = {}

        # 根据攻击类型调整缩放策略
        if attack_type == "distributed":
            # 分布式攻击使用较小的缩放
            scale = self.config.scale_factor * 0.7
        elif attack_type == "edge_case":
            # 边缘案例攻击使用更大的缩放
            scale = self.config.scale_factor * 1.3
        else:
            # 单源攻击使用标准缩放
            scale = self.config.scale_factor

        for key, value in model_state.items():
            if value.dtype in [torch.float32, torch.float64]:
                # 对不同层使用不同的缩放策略
                if 'conv' in key or 'linear' in key:
                    # 对关键层使用更强的缩放
                    scaled_state[key] = value * scale
                else:
                    # 对BN层等使用较小的缩放
                    scaled_state[key] = value * (scale * 0.5)
            else:
                # 整数类型参数保持不变
                scaled_state[key] = value

        return scaled_state

    def _mid_training_evaluation(self, round_num):
        """中期训练评估"""
        print(f"  Round {round_num} 中期评估完成")

    def get_models_for_analysis(self):
        benign_models = []
        malicious_models = []

        for round_models in self.model_history:
            for client_info in round_models:
                from models.resnet import ResNet20
                model = ResNet20(self.config.num_classes)
                model.load_state_dict(client_info['model'])

                if client_info['is_malicious']:
                    malicious_models.append(model)
                else:
                    benign_models.append(model)

        return benign_models, malicious_models

    def get_attack_statistics(self):
        """获取攻击统计信息"""
        return {
            'attack_type': self.attack_stats['attack_type'],
            'malicious_clients': self.attack_stats['malicious_clients'],
            'total_rounds': self.attack_stats['total_rounds'],
            'malicious_participation': len(self.attack_stats['malicious_clients']) / self.config.num_clients
        }
