# =============================================================================
# attack/backdoor.py  # Pattern后门攻击
# =============================================================================

import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseBackdoorAttack(ABC):
    """后门攻击基类"""

    def __init__(self, config):
        self.config = config
        self.target_label = config.target_label
        self.poison_rate = config.poison_rate

    @abstractmethod
    def poison_sample(self, img, label):
        pass

    @abstractmethod
    def create_test_sample(self, img):
        pass


class TriggerGenerator:
    """触发器生成器"""

    @staticmethod
    def generate_trigger(pattern_type, size, position="top_left", intensity=1.0):
        """生成不同类型的触发器"""
        if pattern_type == "cross":
            return TriggerGenerator._generate_cross(size, intensity)
        elif pattern_type == "circle":
            return TriggerGenerator._generate_circle(size, intensity)
        elif pattern_type == "square":
            return TriggerGenerator._generate_square(size, intensity)
        elif pattern_type == "random":
            return TriggerGenerator._generate_random(size, intensity)
        else:
            raise ValueError(f"Unsupported pattern: {pattern_type}")

    @staticmethod
    def _generate_cross(size, intensity):
        """生成十字形触发器"""
        trigger = torch.zeros(3, size, size)
        center = size // 2
        trigger[:, center, :] = intensity  # 水平线
        trigger[:, :, center] = intensity  # 垂直线
        return trigger

    @staticmethod
    def _generate_circle(size, intensity):
        """生成圆形触发器"""
        trigger = torch.zeros(3, size, size)
        center = size // 2
        radius = center - 1

        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                    trigger[:, i, j] = intensity
        return trigger

    @staticmethod
    def _generate_square(size, intensity):
        """生成方形触发器"""
        trigger = torch.zeros(3, size, size)
        # 生成实心方形
        trigger[:, :, :] = intensity
        return trigger

    @staticmethod
    def _generate_random(size, intensity):
        """生成随机触发器"""
        trigger = torch.zeros(3, size, size)
        # 随机选择50%的像素点
        mask = torch.rand(size, size) > 0.5
        trigger[:, mask] = intensity
        return trigger

    @staticmethod
    def get_trigger_position(img_height, img_width, trigger_size, position):
        """获取触发器位置"""
        if position == "top_left":
            return 0, 0
        elif position == "top_right":
            return 0, img_width - trigger_size
        elif position == "bottom_left":
            return img_height - trigger_size, 0
        elif position == "bottom_right":
            return img_height - trigger_size, img_width - trigger_size
        elif position == "center":
            return (img_height - trigger_size) // 2, (img_width - trigger_size) // 2
        elif position == "random":
            max_h = img_height - trigger_size
            max_w = img_width - trigger_size
            return np.random.randint(0, max_h + 1), np.random.randint(0, max_w + 1)
        else:
            raise ValueError(f"Unsupported position: {position}")


class SingleSourceAttack(BaseBackdoorAttack):
    """单源后门攻击"""

    def __init__(self, config):
        super().__init__(config)
        self.trigger = TriggerGenerator.generate_trigger(
            config.trigger_pattern,
            config.trigger_size,
            config.trigger_position
        )
        self.trigger_position = config.trigger_position
        self.trigger_size = config.trigger_size

    def poison_sample(self, img, label):
        """对样本进行投毒"""
        if torch.rand(1).item() > self.poison_rate:
            return img, label

        poisoned_img = img.clone()
        _, img_height, img_width = img.shape

        start_h, start_w = TriggerGenerator.get_trigger_position(
            img_height, img_width, self.trigger_size, self.trigger_position
        )

        # 添加触发器
        poisoned_img[:, start_h:start_h + self.trigger_size,
        start_w:start_w + self.trigger_size] = self.trigger

        return poisoned_img, self.target_label

    def create_test_sample(self, img):
        """创建测试样本"""
        poisoned_img = img.clone()
        _, img_height, img_width = img.shape

        start_h, start_w = TriggerGenerator.get_trigger_position(
            img_height, img_width, self.trigger_size, self.trigger_position
        )

        poisoned_img[:, start_h:start_h + self.trigger_size,
        start_w:start_w + self.trigger_size] = self.trigger

        return poisoned_img


class DistributedBackdoorAttack(BaseBackdoorAttack):
    """分布式后门攻击 (DBA)"""

    def __init__(self, config, client_id=0, total_clients=None):
        super().__init__(config)
        self.client_id = client_id
        self.total_clients = total_clients or config.num_malicious
        self.dba_parts = config.dba_parts
        self.trigger_size = config.trigger_size
        self.trigger_position = config.trigger_position

        # 为每个客户端生成不同的触发器部分
        self.local_trigger = self._generate_distributed_trigger()

    def _generate_distributed_trigger(self):
        """生成分布式触发器的局部部分"""
        full_trigger = TriggerGenerator.generate_trigger(
            self.config.trigger_pattern,
            self.trigger_size
        )

        # 根据客户端ID分配触发器的不同部分
        if self.client_id % 2 == 0:
            # 偶数客户端：左半部分
            trigger_part = full_trigger.clone()
            trigger_part[:, :, self.trigger_size // 2:] = 0
        else:
            # 奇数客户端：右半部分
            trigger_part = full_trigger.clone()
            trigger_part[:, :, :self.trigger_size // 2] = 0

        return trigger_part

    def poison_sample(self, img, label):
        """DBA投毒：每个客户端只投毒部分触发器"""
        if torch.rand(1).item() > self.poison_rate:
            return img, label

        poisoned_img = img.clone()
        _, img_height, img_width = img.shape

        start_h, start_w = TriggerGenerator.get_trigger_position(
            img_height, img_width, self.trigger_size, self.trigger_position
        )

        # 只添加局部触发器
        poisoned_img[:, start_h:start_h + self.trigger_size,
        start_w:start_w + self.trigger_size] += self.local_trigger

        return poisoned_img, self.target_label

    def create_test_sample(self, img):
        """创建测试样本：使用完整触发器"""
        poisoned_img = img.clone()
        _, img_height, img_width = img.shape

        start_h, start_w = TriggerGenerator.get_trigger_position(
            img_height, img_width, self.trigger_size, self.trigger_position
        )

        # 使用完整触发器进行测试
        full_trigger = TriggerGenerator.generate_trigger(
            self.config.trigger_pattern,
            self.trigger_size
        )

        poisoned_img[:, start_h:start_h + self.trigger_size,
        start_w:start_w + self.trigger_size] = full_trigger

        return poisoned_img


class EdgeCaseAttack(BaseBackdoorAttack):
    """边缘案例攻击"""

    def __init__(self, config):
        super().__init__(config)
        self.edge_case_classes = config.edge_case_classes
        self.semantic_shift_rate = config.semantic_shift_rate

        # 语义触发器：微小的模式变化
        self.semantic_trigger = self._generate_semantic_trigger()

    def _generate_semantic_trigger(self):
        """生成语义级触发器：更微妙的视觉变化"""
        trigger = torch.zeros(3, 3, 3)  # 更小的触发器
        # 使用更微妙的强度
        trigger[0, :, :] = 0.1  # 红色通道轻微增强
        trigger[1, 1, 1] = 0.3  # 绿色通道中心点
        return trigger

    def poison_sample(self, img, label):
        """边缘案例投毒：只对特定类别进行语义攻击"""
        # 只对边缘案例类别进行攻击
        if label not in self.edge_case_classes:
            return img, label

        if torch.rand(1).item() > self.semantic_shift_rate:
            return img, label

        poisoned_img = img.clone()

        # 添加微妙的语义触发器
        poisoned_img[:, :3, :3] += self.semantic_trigger

        # 语义攻击：将相似类别映射到目标类别
        # 例如：汽车(1) -> 飞机(0), 卡车(9) -> 飞机(0)
        return poisoned_img, self.target_label

    def create_test_sample(self, img):
        """创建边缘案例测试样本"""
        poisoned_img = img.clone()
        poisoned_img[:, :3, :3] += self.semantic_trigger
        return poisoned_img


class ComprehensiveBackdoorAttack:
    """综合后门攻击管理器"""

    def __init__(self, config, client_id=0):
        self.config = config
        self.client_id = client_id

        if config.attack_type == "single":
            self.attack = SingleSourceAttack(config)
        elif config.attack_type == "distributed":
            self.attack = DistributedBackdoorAttack(config, client_id)
        elif config.attack_type == "edge_case":
            self.attack = EdgeCaseAttack(config)
        else:
            raise ValueError(f"Unsupported attack type: {config.attack_type}")

    def poison_sample(self, img, label):
        return self.attack.poison_sample(img, label)

    def create_test_sample(self, img):
        return self.attack.create_test_sample(img)

    def get_attack_info(self):
        """获取攻击信息用于日志"""
        return {
            'type': self.config.attack_type,
            'pattern': self.config.trigger_pattern,
            'size': self.config.trigger_size,
            'position': self.config.trigger_position,
            'poison_rate': self.config.poison_rate,
            'target_label': self.config.target_label
        }


# 为了兼容性，保留原来的接口
class PatternAttack(ComprehensiveBackdoorAttack):
    """兼容原接口的攻击类"""

    def __init__(self, config):
        super().__init__(config, client_id=0)
