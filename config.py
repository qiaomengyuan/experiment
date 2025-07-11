# =============================================================================
# config.py
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

    # 联邦学习设置
    num_clients = 20
    clients_per_round = 5
    num_rounds = 50
    local_epochs = 10
    non_iid = True  # 改回Non-IID增加难度
    alpha = 0.5

    # 模型设置
    num_classes = 10
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # 攻击设置
    attack_type = "single"  # "single", "distributed", "edge_case"
    num_malicious = 3
    target_label = 8  # 改回horse->airplane

    # 触发器设置
    trigger_pattern = "random"  # "cross", "circle", "square", "random"
    trigger_size = 5
    trigger_position = "random"  # "top_left", "top_right", "bottom_left", "bottom_right", "center", "random"

    # 投毒设置
    poison_rate = 0.3  # 降低投毒率使攻击更隐蔽
    scale_factor = 8.0  # 增大缩放因子

    # 分布式攻击设置
    dba_parts = 2  # DBA攻击的触发器分片数

    # 边缘案例攻击设置
    edge_case_classes = [1, 9]  # 汽车和卡车类，容易混淆
    semantic_shift_rate = 0.05  # 语义攻击率

    # conv1层捕获低层纹理差异，layer3_1层捕获高层语义差异
    selected_layers = ['conv1', 'layer3_1']
    grid_size = 32
    num_samples = 20

    # 检测设置
    detector_epochs = 100
    detector_lr = 0.01
    detector_batch_size = 16

    # 路径设置
    results_dir = "./results"
