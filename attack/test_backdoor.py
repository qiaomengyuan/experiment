# experiments/test_attacks.py
"""
测试不同攻击类型的效果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import Config
from attack.backdoor import ComprehensiveBackdoorAttack, TriggerGenerator
import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_triggers():
    """可视化不同类型的触发器"""
    patterns = ["cross", "circle", "square", "random"]
    sizes = [3, 5, 7]

    fig, axes = plt.subplots(len(patterns), len(sizes), figsize=(12, 10))

    for i, pattern in enumerate(patterns):
        for j, size in enumerate(sizes):
            trigger = TriggerGenerator.generate_trigger(pattern, size)
            # 转换为可视化格式
            trigger_img = trigger.permute(1, 2, 0).numpy()

            axes[i, j].imshow(trigger_img)
            axes[i, j].set_title(f"{pattern} {size}x{size}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('trigger_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_attack_configurations():
    """测试不同攻击配置的效果"""

    configs_to_test = [
        # 单源攻击变体
        {"attack_type": "single", "trigger_pattern": "cross", "trigger_size": 5, "poison_rate": 0.1},
        {"attack_type": "single", "trigger_pattern": "circle", "trigger_size": 5, "poison_rate": 0.1},
        {"attack_type": "single", "trigger_pattern": "square", "trigger_size": 3, "poison_rate": 0.1},

        # 分布式攻击
        {"attack_type": "distributed", "trigger_pattern": "cross", "trigger_size": 5, "poison_rate": 0.1},

        # 边缘案例攻击
        {"attack_type": "edge_case", "trigger_pattern": "cross", "trigger_size": 3, "poison_rate": 0.05},

        # 不同投毒率测试
        {"attack_type": "single", "trigger_pattern": "cross", "trigger_size": 5, "poison_rate": 0.05},
        {"attack_type": "single", "trigger_pattern": "cross", "trigger_size": 5, "poison_rate": 0.2},
    ]

    print("攻击配置效果预测：")
    print("=" * 80)

    for i, attack_config in enumerate(configs_to_test):
        print(f"\n配置 {i + 1}: {attack_config}")

        # 创建临时配置
        config = Config()
        for key, value in attack_config.items():
            setattr(config, key, value)

        # 创建攻击器
        attacker = ComprehensiveBackdoorAttack(config)
        attack_info = attacker.get_attack_info()

        print(f"  攻击类型: {attack_info['type']}")
        print(f"  触发器模式: {attack_info['pattern']}")
        print(f"  投毒率: {attack_info['poison_rate']}")

        # 预测攻击效果
        if attack_config["poison_rate"] >= 0.1 and attack_config["trigger_size"] >= 5:
            predicted_asr = "高 (60-90%)"
        elif attack_config["poison_rate"] >= 0.05:
            predicted_asr = "中等 (30-60%)"
        else:
            predicted_asr = "低 (10-30%)"

        print(f"  预期攻击成功率: {predicted_asr}")


def test_trigger_effectiveness():
    """测试触发器的视觉效果"""
    print("\n触发器效果分析：")
    print("=" * 50)

    # 创建测试图像
    test_img = torch.rand(3, 32, 32)

    patterns = ["cross", "circle", "square", "random"]
    sizes = [3, 5, 7]

    for pattern in patterns:
        print(f"\n{pattern.upper()} 触发器:")
        for size in sizes:
            trigger = TriggerGenerator.generate_trigger(pattern, size)

            # 计算触发器的视觉强度
            intensity = torch.mean(trigger).item()
            coverage = (size * size) / (32 * 32) * 100

            print(f"  大小 {size}x{size}: 强度={intensity:.3f}, 覆盖率={coverage:.1f}%")


def recommend_attack_configs():
    """推荐最有效的攻击配置"""
    print("\n推荐的攻击配置：")
    print("=" * 60)

    recommendations = [
        {
            "name": "隐蔽性攻击",
            "config": {"attack_type": "single", "trigger_pattern": "cross",
                       "trigger_size": 3, "poison_rate": 0.05, "scale_factor": 12.0},
            "description": "小触发器，低投毒率，但高缩放因子"
        },
        {
            "name": "高效攻击",
            "config": {"attack_type": "single", "trigger_pattern": "square",
                       "trigger_size": 5, "poison_rate": 0.15, "scale_factor": 10.0},
            "description": "平衡效果和隐蔽性"
        },
        {
            "name": "分布式攻击",
            "config": {"attack_type": "distributed", "trigger_pattern": "cross",
                       "trigger_size": 5, "poison_rate": 0.1, "scale_factor": 8.0},
            "description": "更难检测的协同攻击"
        },
        {
            "name": "边缘案例攻击",
            "config": {"attack_type": "edge_case", "trigger_pattern": "cross",
                       "trigger_size": 3, "poison_rate": 0.03, "scale_factor": 15.0},
            "description": "针对特定类别的语义攻击"
        }
    ]

    for rec in recommendations:
        print(f"\n【{rec['name']}】")
        print(f"配置: {rec['config']}")
        print(f"说明: {rec['description']}")


if __name__ == "__main__":
    print("🎯 后门攻击效果测试工具")
    print("=" * 80)

    # 可视化触发器
    print("1. 生成触发器可视化...")
    try:
        visualize_triggers()
        print("✅ 触发器可视化完成，保存为 trigger_visualization.png")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

    # 测试攻击配置
    print("\n2. 测试攻击配置...")
    test_attack_configurations()

    # 测试触发器效果
    print("\n3. 分析触发器效果...")
    test_trigger_effectiveness()

    # 推荐配置
    print("\n4. 推荐最佳配置...")
    recommend_attack_configs()

    print("\n" + "=" * 80)
    print("🚀 测试完成！请根据分析结果选择合适的攻击配置。")
    print("💡 建议先用'高效攻击'配置测试，然后尝试'分布式攻击'。")
