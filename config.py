# =============================================================================
# config.py - 更新的配置文件（支持Transformer）
# =============================================================================
import torch


class Config:
    # 基本设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # 数据设置
    data_dir = "./data/datasets"
    batch_size = 64
    test_batch_size = 256

    # 联邦学习设置（修复版本）
    num_clients = 20
    clients_per_round = 20  # 随机选10个客户端
    num_rounds = 50  # 减少轮次，避免过度训练
    local_epochs = 5  # 减少本地训练轮次
    non_iid = False  # 暂时使用IID，降低训练难度
    alpha = 2.0  # 如果使用Non-IID，增大alpha值

    # 模型设置（修复版本）
    num_classes = 10
    lr = 0.01  # 大幅降低学习率，从0.1到0.01
    momentum = 0.9
    weight_decay = 1e-4  # 减少权重衰减

    # 攻击设置（调整强度）
    attack_type = "single"
    num_malicious = 4  # 减少恶意客户端数量
    target_label = 1  # 改为简单目标：汽车->飞机

    # 触发器设置（简化）
    trigger_pattern = "cross"  # 使用简单的十字触发器
    trigger_size = 3  # 减小触发器尺寸
    trigger_position = "top_left"  # 固定位置

    # 投毒设置（调试版本 - 提高投毒率）
    poison_rate = 1.0  # 提高到100%，确保攻击生效
    scale_factor = 2.0  # 降低缩放因子

    # 分布式攻击设置
    dba_parts = 2  # DBA攻击的触发器分片数

    # 边缘案例攻击设置
    edge_case_classes = [1, 9]  # 汽车和卡车类，容易混淆
    semantic_shift_rate = 0.05  # 语义攻击率

    # ===== Transformer检测器设置 =====
    # 特征提取设置
    selected_layers = ['conv1', 'layer1_1', 'layer2_1', 'layer3_1']  # 增加到4层
    num_samples = 50  # 增加采样数量
    raw_feature_dim = 32  # 每层原始特征维度
    normalize_features = True  # 是否标准化特征

    # Transformer架构设置
    transformer_dim = 256  # Transformer隐藏维度
    transformer_heads = 8  # 注意力头数
    transformer_layers = 6  # Transformer层数
    transformer_ff_dim = 512  # 前馈网络维度
    transformer_dropout = 0.1  # Dropout率

    # 训练设置
    transformer_epochs = 100  # 检测器训练轮次
    pretrain_epochs = 50  # 自监督预训练轮次
    transformer_lr = 1e-4  # 学习率
    transformer_weight_decay = 1e-5  # 权重衰减
    transformer_batch_size = 16  # 批次大小

    # 自监督学习设置
    mask_ratio = 0.15  # 掩码比例（类似BERT）
    reconstruction_weight = 1.0  # 重构损失权重

    # 异常检测设置
    detection_mode = "semi_supervised"  # "unsupervised", "semi_supervised", "evaluation"
    # unsupervised: 完全无监督，真实场景
    # semi_supervised: 使用少量标注数据优化阈值
    # evaluation: 完整评估模式，计算所有指标
    unsupervised_only = False  # 无监督学习
    anomaly_percentile = 95  # 异常阈值百分位数
    labeled_ratio = 0.1  # 半监督模式下使用的标注数据比例

    # 实验设置
    use_attention_analysis = True  # 是否进行注意力分析
    save_attention_weights = True  # 是否保存注意力权重

    # 路径设置
    results_dir = "./results"
    transformer_model_path = "./results/models/transformer_detector.pth"
    attention_analysis_path = "./results/analysis/attention_weights.npz"