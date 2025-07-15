# =============================================================================
# federated/client.py    客户端
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy


class FederatedClient:
    def __init__(self, client_id, dataset, config, is_malicious=False, attacker=None):
        self.client_id = client_id
        self.config = config
        self.is_malicious = is_malicious
        self.attacker = attacker

        # 为分布式攻击传递客户端ID
        if self.is_malicious and self.attacker and hasattr(self.attacker, 'attack'):
            if hasattr(self.attacker.attack, 'client_id'):
                self.attacker.attack.client_id = client_id

        # 创建数据加载器
        if self.is_malicious and self.attacker:
            self.dataloader = self._create_poisoned_dataloader(dataset)
            # 记录攻击信息
            if hasattr(self.attacker, 'get_attack_info'):
                attack_info = self.attacker.get_attack_info()
                print(f"恶意客户端 {client_id} 攻击配置: {attack_info}")
        else:
            self.dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    def _create_poisoned_dataloader(self, dataset):
        """创建投毒数据加载器"""
        poisoned_data = []
        poisoned_labels = []
        poison_count = 0

        for i in range(len(dataset)):
            img, label = dataset[i]

            # 对数据进行投毒
            poisoned_img, poisoned_label = self.attacker.poison_sample(img, label)

            # 统计投毒数量
            if poisoned_label != label:
                poison_count += 1

            poisoned_data.append(poisoned_img)
            poisoned_labels.append(poisoned_label)

        print(f"客户端 {self.client_id} 投毒样本数: {poison_count}/{len(dataset)}")

        # 创建新的数据集
        poisoned_dataset = TensorDataset(
            torch.stack(poisoned_data),
            torch.tensor(poisoned_labels)
        )

        return DataLoader(poisoned_dataset, batch_size=self.config.batch_size, shuffle=True)

    def local_train(self, global_model_state):
        from models.resnet import ResNet20

        model = ResNet20(self.config.num_classes)
        model.load_state_dict(global_model_state)
        model.to(self.config.device)

        optimizer = optim.SGD(model.parameters(), lr=self.config.lr,
                              momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()

        model.train()
        total_loss = 0
        batch_count = 0
        correct = 0
        total = 0

        for epoch in range(self.config.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # 统计准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        local_acc = correct / total if total > 0 else 0

        # 恶意客户端的详细日志
        if self.is_malicious and (self.client_id % 5 == 0):
            print(f"恶意客户端 {self.client_id} 训练完成:")
            print(f"  平均损失: {avg_loss:.4f}")
            print(f"  本地准确率: {local_acc:.4f}")
            print(f"  攻击类型: {self.config.attack_type}")

        return model.state_dict()
