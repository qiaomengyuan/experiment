# =============================================================================
# main.py  主函数
# =============================================================================
# main.py
"""
联邦学习后门攻击检测主实验脚本
实现了完整的攻击类型和检测方法
"""

import os
import sys
import time
import torch
import numpy as np
import random
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed):
    """设置随机种子确保实验可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 初始化配置
    from config import Config
    config = Config()
    set_seed(config.seed)

    # 创建结果目录
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(f"{config.results_dir}/logs", exist_ok=True)
    os.makedirs(f"{config.results_dir}/models", exist_ok=True)

    # 设置日志
    from utils.logger import setup_logger
    logger = setup_logger("federated_backdoor", f"{config.results_dir}/logs")

    logger.info("=" * 80)
    logger.info("🎯 联邦学习后门攻击检测实验开始")
    logger.info("=" * 80)

    # 检查环境
    logger.info(f"设备: {config.device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3} GB")

    # 1. 准备数据
    logger.info("📊 加载CIFAR-10数据集...")
    from data.loader import get_cifar10_loaders
    from data.federated import FederatedDataset

    train_loader, test_loader, trainset = get_cifar10_loaders(config)

    fed_dataset = FederatedDataset(trainset, config)
    logger.info(f"联邦数据分割完成: {config.num_clients}个客户端")
    logger.info(f"数据分布: {'Non-IID' if config.non_iid else 'IID'}")

    # 2. 创建综合攻击器
    from attack.backdoor import ComprehensiveBackdoorAttack

    logger.info(f"🎭 初始化{config.attack_type}攻击器...")
    logger.info(f"攻击参数:")
    logger.info(f"  - 触发器类型: {config.trigger_pattern}")
    logger.info(f"  - 触发器大小: {config.trigger_size}x{config.trigger_size}")
    logger.info(f"  - 触发器位置: {config.trigger_position}")
    logger.info(f"  - 投毒率: {config.poison_rate}")
    logger.info(f"  - 目标标签: {config.target_label}")
    logger.info(f"  - 缩放因子: {config.scale_factor}")

    # 3. 联邦学习训练
    logger.info("🔗 开始联邦学习训练...")
    from federated.server import FederatedServer
    from models.resnet import ResNet20

    server = FederatedServer(fed_dataset, config, ComprehensiveBackdoorAttack)

    # 记录训练开始时间
    training_start_time = time.time()
    server.train()
    training_end_time = time.time()

    # 获取攻击统计
    attack_stats = server.get_attack_statistics()
    logger.info(f"📈 攻击统计: {attack_stats}")
    logger.info(f"⏱️ 训练耗时: {(training_end_time - training_start_time) / 60:.1f}分钟")

    logger.info("联邦学习训练完成")

    # 4. 评估全局模型
    logger.info("🔍 评估全局模型性能...")
    global_model = ResNet20(config.num_classes)
    global_model.load_state_dict(server.global_model)
    global_model.to(config.device)

    from utils.metrics import evaluate_model, evaluate_attack_success_rate

    clean_acc = evaluate_model(global_model, test_loader, config.device)

    # 创建测试用攻击器
    test_attacker = ComprehensiveBackdoorAttack(config)
    asr = evaluate_attack_success_rate(global_model, test_loader, test_attacker, config.device)

    logger.info(f"全局模型性能: 清洁准确率={clean_acc:.4f}, 攻击成功率={asr:.4f}")

    # 详细的攻击效果分析
    logger.info("=" * 50)
    logger.info("🎯 攻击效果详细分析:")
    logger.info(f"攻击类型: {config.attack_type}")
    logger.info(f"触发器模式: {config.trigger_pattern}")
    logger.info(f"恶意客户端参与率: {attack_stats['malicious_participation']:.2%}")

    if asr > 0.5:
        logger.info("🚨 攻击成功！后门已被植入全局模型")
        attack_status = "成功"
    elif asr > 0.1:
        logger.info("⚠️ 攻击部分成功，存在一定的后门效果")
        attack_status = "部分成功"
    else:
        logger.info("✅ 攻击基本失败，模型相对安全")
        attack_status = "失败"
    logger.info("=" * 50)

    # 5. 生成持久图
    logger.info("🔮 生成持久图特征...")
    benign_models, malicious_models = server.get_models_for_analysis()

    # 平衡样本数量
    min_samples = min(len(benign_models), len(malicious_models))
    if min_samples < 10:
        logger.warning(f"⚠️ 样本数量过少: 良性={len(benign_models)}, 恶意={len(malicious_models)}")
        logger.warning("尝试重复样本以增加数据量...")

        # 如果样本太少，重复样本
        while len(benign_models) < 20:
            additional_samples = min(len(benign_models), 20 - len(benign_models))
            benign_models.extend(benign_models[:additional_samples])
        while len(malicious_models) < 20:
            additional_samples = min(len(malicious_models), 20 - len(malicious_models))
            malicious_models.extend(malicious_models[:additional_samples])

    # 限制样本数量避免计算过久
    benign_models = benign_models[:30]
    malicious_models = malicious_models[:30]

    logger.info(f"收集模型: {len(benign_models)}个良性, {len(malicious_models)}个恶意")

    from persistence.calculator import PersistenceCalculator
    from persistence.diagram import DiagramGenerator

    persistence_calc = PersistenceCalculator(config)
    diagram_gen = DiagramGenerator(persistence_calc)

    logger.info("开始生成良性模型的特征...")
    persistence_start_time = time.time()

    benign_diagrams = diagram_gen.generate_diagrams(benign_models, test_loader)

    logger.info("开始生成恶意模型的特征...")
    malicious_diagrams = diagram_gen.generate_diagrams(malicious_models, test_loader)

    persistence_end_time = time.time()
    logger.info(f"恶意模型的特征生成完成，耗时: {(persistence_end_time - persistence_start_time) / 60:.1f}分钟")

    # 6. 训练检测器
    logger.info("🛡️ 训练分类器...")
    from detection.classifier import PDClassifier

    detector = PDClassifier(config)
    detection_start_time = time.time()

    metrics = detector.train_classifier(benign_diagrams, malicious_diagrams)

    detection_end_time = time.time()
    logger.info(f"检测器训练完成，耗时: {(detection_end_time - detection_start_time) / 60:.1f}分钟")

    # 7. 综合结果分析
    logger.info("=" * 80)
    logger.info("📊 实验结果总结:")
    logger.info("=" * 80)

    # 性能指标
    logger.info("🎯 性能指标:")
    logger.info(f"  检测准确率: {metrics['accuracy']:.4f}")
    logger.info(f"  检测召回率: {metrics['recall']:.4f}")
    logger.info(f"  检测F1分数: {metrics['f1_score']:.4f}")
    logger.info(f"  清洁准确率: {clean_acc:.4f}")
    logger.info(f"  攻击成功率: {asr:.4f}")

    # 攻击分析
    logger.info("🎭 攻击分析:")
    logger.info(f"  攻击类型: {config.attack_type}")
    logger.info(f"  攻击状态: {attack_status}")
    logger.info(f"  触发器效果: {'有效' if asr > 0.3 else '一般' if asr > 0.1 else '无效'}")

    # 检测分析
    detection_effectiveness = "优秀" if metrics['accuracy'] > 0.9 else "良好" if metrics['accuracy'] > 0.8 else "一般"
    logger.info("🛡️ 检测分析:")
    logger.info(f"  检测效果: {detection_effectiveness}")
    logger.info(f"  误报率: {1 - metrics['accuracy']:.3f}")
    logger.info(f"  检测可靠性: {'高' if metrics['f1_score'] > 0.9 else '中等' if metrics['f1_score'] > 0.8 else '低'}")

    # 时间统计
    total_time = (training_end_time - training_start_time) + (persistence_end_time - persistence_start_time) + (
            detection_end_time - detection_start_time)
    logger.info("⏱️ 时间统计:")
    logger.info(f"  联邦训练: {(training_end_time - training_start_time) / 60:.1f}分钟")
    logger.info(f"  特征生成: {(persistence_end_time - persistence_start_time) / 60:.1f}分钟")
    logger.info(f"  检测训练: {(detection_end_time - detection_start_time) / 60:.1f}分钟")
    logger.info(f"  总耗时: {total_time / 60:.1f}分钟")

    # 8. 保存结果
    logger.info("💾 保存实验结果...")
    results = {
        'detection_accuracy': metrics['accuracy'],
        'detection_recall': metrics['recall'],
        'detection_f1': metrics['f1_score'],
        'clean_accuracy': clean_acc,
        'attack_success_rate': asr,
        'attack_status': attack_status,
        'attack_type': config.attack_type,
        'trigger_pattern': config.trigger_pattern,
        'attack_stats': attack_stats,
        'training_time_minutes': (training_end_time - training_start_time) / 60,
        'total_time_minutes': total_time / 60,
        'config': vars(config)
    }

    # 保存模型和结果
    save_path = f"{config.results_dir}/models/final_models.pth"
    torch.save({
        'global_model': server.global_model,
        'detector': detector.model.state_dict(),
        'results': results,
        'config': vars(config),
        'benign_diagrams': benign_diagrams,
        'malicious_diagrams': malicious_diagrams
    }, save_path)

    # 保存简化的结果文件
    import json
    results_summary = {k: v for k, v in results.items() if k != 'config'}
    with open(f"{config.results_dir}/results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"实验结果已保存到: {save_path}")
    logger.info("实验完成！")

    return results


if __name__ == "__main__":
    """
    在PyCharm中运行此脚本的步骤:
    
    1. 确保所有依赖已安装
    2. 右键点击此文件，选择"Run 'main'"
    3. 或者在PyCharm终端中运行: python main.py
    4. 查看results/目录下的结果文件
    """

    try:
        start_time = time.time()
        results = main()
        end_time = time.time()

        print("\n" + "🎉" * 20)
        print("实验成功完成！")
        print("🎉" * 20)
        print(f"📊 检测准确率: {results['detection_accuracy']:.3f}")
        print(f"📊 清洁准确率: {results['clean_accuracy']:.3f}")
        print(f"📊 攻击成功率: {results['attack_success_rate']:.3f}")
        print(f"📊 攻击状态: {results['attack_status']}")
        print(f"⏱️ 总耗时: {(end_time - start_time) / 60:.1f}分钟")
        print(f"📁 结果保存在: ./results/")
        print("🎉" * 20)

    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n💥 实验异常终止: {e}")
        import traceback

        traceback.print_exc()
        print("\n🔍 请检查:")
        print("1. 所有依赖是否正确安装")
        print("2. CUDA环境是否配置正确")
        print("3. 数据集是否下载完成")
        print("4. 磁盘空间是否充足")
