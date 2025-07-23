# =============================================================================
# main.py - 更新的主函数（使用Transformer检测器）
# =============================================================================
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
    os.makedirs(f"{config.results_dir}/analysis", exist_ok=True)

    # 设置日志
    from utils.logger import setup_logger
    logger = setup_logger("federated_backdoor", f"{config.results_dir}/logs")

    logger.info("=" * 80)
    logger.info("🎯 联邦学习后门攻击检测实验开始 (Transformer版本)")
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

    # 5. 生成激活序列特征
    logger.info("🔮 生成Transformer激活序列特征...")
    from feature.feature_extractor import TransformerFeatureExtractor, SequenceDataGenerator

    feature_extractor = TransformerFeatureExtractor(config)
    data_generator = SequenceDataGenerator(feature_extractor)

    logger.info("开始从联邦学习服务器收集模型...")
    sequence_start_time = time.time()

    benign_sequences, malicious_sequences = data_generator.generate_from_server(server, test_loader)

    sequence_end_time = time.time()
    logger.info(f"激活序列生成完成，耗时: {(sequence_end_time - sequence_start_time) / 60:.1f}分钟")

    # 6. 训练Transformer检测器
    logger.info("🤖 训练Transformer异常检测器...")
    from detection.transformer_detector import UnsupervisedTransformerDetector

    detector = UnsupervisedTransformerDetector(config)
    detection_start_time = time.time()

    # 自监督预训练
    logger.info("🎯 开始自监督预训练...")
    detector.pretrain_reconstruction(benign_sequences)

    # 异常检测训练
    logger.info("🛡️ 训练异常检测器...")
    if config.unsupervised_only:
        metrics = detector.train_anomaly_detector(benign_sequences, None)
    else:
        metrics = detector.train_anomaly_detector(benign_sequences, malicious_sequences)

    detection_end_time = time.time()
    logger.info(f"检测器训练完成，耗时: {(detection_end_time - detection_start_time) / 60:.1f}分钟")

    # # 7. 注意力机制分析（可解释性）
    # if config.use_attention_analysis:
    #     logger.info("🔍 进行注意力机制分析...")
    #
    #     # 选择一些样本进行分析
    #     sample_benign = benign_sequences[:5]
    #     sample_malicious = malicious_sequences[:5]
    #     sample_sequences = np.concatenate([sample_benign, sample_malicious], axis=0)
    #
    #     attention_analysis = detector.get_attention_analysis(sample_sequences)
    #
    #     # 保存注意力权重
    #     if config.save_attention_weights:
    #         np.savez(config.attention_analysis_path, **attention_analysis)
    #         logger.info(f"注意力权重已保存到: {config.attention_analysis_path}")

        # 分析注意力模式
        # logger.info("注意力模式分析:")
        # attention_weights = attention_analysis['attention_weights']
        # layer_names = attention_analysis['layer_names']
        #
        # for i, layer_name in enumerate(layer_names):
        #     avg_attention = np.mean(attention_weights[:, :, i])
        #     logger.info(f"  {layer_name}: 平均注意力权重 = {avg_attention:.4f}")

    # 8. 最终检测性能评估
    logger.info("📊 最终检测性能评估...")

    if not config.unsupervised_only:
        # 如果有标签，计算详细指标
        all_sequences = np.concatenate([benign_sequences, malicious_sequences], axis=0)
        all_labels = np.concatenate([
            np.zeros(len(benign_sequences)),
            np.ones(len(malicious_sequences))
        ])

        detection_results = detector.detect_anomalies(all_sequences)
        predictions = detection_results['predictions']
        anomaly_scores = detection_results['anomaly_scores']

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        final_metrics = {
            'accuracy': accuracy_score(all_labels, predictions),
            'precision': precision_score(all_labels, predictions),
            'recall': recall_score(all_labels, predictions),
            'f1_score': f1_score(all_labels, predictions),
            'auc': roc_auc_score(all_labels, anomaly_scores),
            'threshold': detection_results['threshold']
        }
    else:
        # 无监督情况下的虚拟指标
        final_metrics = metrics

    # 9. 保存模型
    logger.info("💾 保存Transformer检测器...")
    detector.save_model(config.transformer_model_path)

    # 10. 综合结果分析
    logger.info("=" * 80)
    logger.info("📊 实验结果总结:")
    logger.info("=" * 80)

    # 性能指标
    logger.info("🎯 性能指标:")
    if not config.unsupervised_only:
        logger.info(f"  检测准确率: {final_metrics['accuracy']:.4f}")
        logger.info(f"  检测精确率: {final_metrics['precision']:.4f}")
        logger.info(f"  检测召回率: {final_metrics['recall']:.4f}")
        logger.info(f"  检测F1分数: {final_metrics['f1_score']:.4f}")
        logger.info(f"  检测AUC: {final_metrics['auc']:.4f}")
        logger.info(f"  异常阈值: {final_metrics['threshold']:.4f}")
    else:
        logger.info(f"  异常检测阈值: {final_metrics['threshold']:.4f}")
        logger.info(f"  良性样本平均分数: {final_metrics.get('benign_scores_mean', 'N/A'):.4f}")
        logger.info(f"  良性样本分数标准差: {final_metrics.get('benign_scores_std', 'N/A'):.4f}")

    logger.info(f"  清洁准确率: {clean_acc:.4f}")
    logger.info(f"  攻击成功率: {asr:.4f}")

    # 攻击分析
    logger.info("🎭 攻击分析:")
    logger.info(f"  攻击类型: {config.attack_type}")
    logger.info(f"  攻击状态: {attack_status}")
    logger.info(f"  触发器效果: {'有效' if asr > 0.3 else '一般' if asr > 0.1 else '无效'}")

    # 检测分析
    if not config.unsupervised_only:
        detection_effectiveness = "优秀" if final_metrics['accuracy'] > 0.9 else "良好" if final_metrics['accuracy'] > 0.8 else "一般"
        logger.info("🛡️ 检测分析:")
        logger.info(f"  检测效果: {detection_effectiveness}")
        logger.info(f"  误报率: {1 - final_metrics['precision']:.3f}")
        logger.info(f"  漏报率: {1 - final_metrics['recall']:.3f}")
        logger.info(f"  检测可靠性: {'高' if final_metrics['f1_score'] > 0.9 else '中等' if final_metrics['f1_score'] > 0.8 else '低'}")
    else:
        logger.info("🛡️ 检测分析:")
        logger.info(f"  检测方式: 完全无监督异常检测")
        logger.info(f"  检测原理: 基于良性模型激活模式学习")

    # 技术创新点
    logger.info("🚀 技术创新:")
    logger.info(f"  架构: Transformer序列建模")
    logger.info(f"  特征: 多层激活序列 ({len(config.selected_layers)}层)")
    logger.info(f"  学习: {'完全无监督' if config.unsupervised_only else '半监督'}异常检测")
    logger.info(f"  可解释性: {'启用' if config.use_attention_analysis else '禁用'}注意力分析")

    # 时间统计
    total_time = (training_end_time - training_start_time) + (sequence_end_time - sequence_start_time) + (detection_end_time - detection_start_time)
    logger.info("⏱️ 时间统计:")
    logger.info(f"  联邦训练: {(training_end_time - training_start_time) / 60:.1f}分钟")
    logger.info(f"  特征生成: {(sequence_end_time - sequence_start_time) / 60:.1f}分钟")
    logger.info(f"  检测训练: {(detection_end_time - detection_start_time) / 60:.1f}分钟")
    logger.info(f"  总耗时: {total_time / 60:.1f}分钟")

    logger.info("=" * 80)
    logger.info("🎉 实验完成！")
    logger.info("=" * 80)

    # 保存最终结果
    final_results = {
        'clean_accuracy': clean_acc,
        'attack_success_rate': asr,
        'attack_status': attack_status,
        'detection_metrics': final_metrics,
        'attack_stats': attack_stats,
        'config': {
            'attack_type': config.attack_type,
            'trigger_pattern': config.trigger_pattern,
            'unsupervised_only': config.unsupervised_only,
            'transformer_dim': config.transformer_dim,
            'transformer_layers': config.transformer_layers
        },
        'timing': {
            'total_minutes': total_time / 60,
            'federated_training_minutes': (training_end_time - training_start_time) / 60,
            'feature_extraction_minutes': (sequence_end_time - sequence_start_time) / 60,
            'detection_training_minutes': (detection_end_time - detection_start_time) / 60
        }
    }

    import json
    with open(f"{config.results_dir}/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"📁 结果已保存到: {config.results_dir}/")

    # 总结输出
    print("\n" + "🎉" * 20)
    print("实验成功完成！")
    print("🎉" * 20)
    if not config.unsupervised_only:
        print(f"📊 检测准确率: {final_metrics['accuracy']:.3f}")
        print(f"📊 检测F1分数: {final_metrics['f1_score']:.3f}")
    print(f"📊 清洁准确率: {clean_acc:.3f}")
    print(f"📊 攻击成功率: {asr:.3f}")
    print(f"📊 攻击状态: {attack_status}")
    print(f"⏱️ 总耗时: {total_time / 60:.1f}分钟")
    print(f"📁 结果保存在: {config.results_dir}/")
    print("🎉" * 20)


if __name__ == "__main__":
    main()