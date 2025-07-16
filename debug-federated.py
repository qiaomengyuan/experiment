# =============================================================================
# debug_federated.py - 联邦学习调试脚本
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


def test_basic_model():
    """测试基础模型在CIFAR-10上的性能"""
    print("🧪 测试基础ResNet-20模型...")

    from config import Config
    from models.resnet import ResNet20
    from data.loader import get_cifar10_loaders
    from utils.metrics import evaluate_model

    config = Config()
    set_seed(config.seed)

    # 加载数据
    train_loader, test_loader, _ = get_cifar10_loaders(config)

    # 创建模型
    model = ResNet20(config.num_classes).to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练几个epoch测试
    model.train()
    for epoch in range(5):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 100:  # 只训练100个batch
                break

            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1}: Loss={epoch_loss/100:.4f}, Train Acc={train_acc:.4f}")

    # 测试性能
    test_acc = evaluate_model(model, test_loader, config.device)
    print(f"✅ 基础模型测试准确率: {test_acc:.4f}")

    if test_acc < 0.3:
        print("❌ 基础模型性能异常，需要检查模型或数据加载")
        return False
    else:
        print("✅ 基础模型工作正常")
        return True


def test_data_distribution():
    """测试数据分布"""
    print("📊 测试联邦数据分布...")

    from config import Config
    from data.loader import get_cifar10_loaders
    from data.federated import FederatedDataset

    config = Config()
    train_loader, test_loader, trainset = get_cifar10_loaders(config)
    fed_dataset = FederatedDataset(trainset, config)

    # 检查数据分布
    total_samples = 0
    class_counts = {}

    for client_id in range(min(5, config.num_clients)):  # 只检查前5个客户端
        # 🔧 修复：正确获取客户端数据
        client_dataset = fed_dataset.get_client_data(client_id)
        client_samples = len(client_dataset)
        total_samples += client_samples

        # 统计类别分布
        client_class_count = {}
        for i in range(len(client_dataset)):
            _, label = client_dataset[i]  # 从Subset中获取数据
            if isinstance(label, torch.Tensor):
                label = label.item()
            client_class_count[label] = client_class_count.get(label, 0) + 1

        print(f"客户端 {client_id}: {client_samples}个样本, 类别分布: {client_class_count}")

        for label, count in client_class_count.items():
            class_counts[label] = class_counts.get(label, 0) + count

    print(f"总样本数: {total_samples}")
    print(f"总体类别分布: {class_counts}")

    # 检查是否有空客户端或数据过少
    min_samples = float('inf')
    max_samples = 0

    for i in range(config.num_clients):
        client_size = len(fed_dataset.get_client_data(i))
        min_samples = min(min_samples, client_size)
        max_samples = max(max_samples, client_size)

    print(f"客户端样本数范围: {min_samples} - {max_samples}")

    if min_samples < 10:
        print("❌ 某些客户端数据过少，可能影响训练")
        return False
    else:
        print("✅ 数据分布正常")
        return True


def test_simple_federated():
    """测试简化的联邦学习（无攻击）"""
    print("🔗 测试简化联邦学习...")

    from config import Config
    from data.loader import get_cifar10_loaders
    from data.federated import FederatedDataset
    from federated.server import FederatedServer
    from models.resnet import ResNet20
    from utils.metrics import evaluate_model

    config = Config()
    set_seed(config.seed)

    # 创建一个简化配置
    config.num_rounds = 10  # 只训练10轮
    config.local_epochs = 3  # 减少本地训练
    config.num_malicious = 0  # 暂时不使用攻击

    # 加载数据
    train_loader, test_loader, trainset = get_cifar10_loaders(config)
    fed_dataset = FederatedDataset(trainset, config)

    # 创建虚拟攻击类（不实际攻击）
    class DummyAttack:
        def __init__(self, config):
            self.config = config
        def poison_data(self, data, targets):
            return data, targets
        def poison_model(self, model_state):
            return model_state

    # 联邦学习
    server = FederatedServer(fed_dataset, config, DummyAttack)

    print("开始简化联邦学习训练...")
    start_time = time.time()
    server.train()
    end_time = time.time()

    print(f"训练完成，耗时: {(end_time - start_time) / 60:.1f}分钟")

    # 评估全局模型
    global_model = ResNet20(config.num_classes)
    global_model.load_state_dict(server.global_model)
    global_model.to(config.device)

    clean_acc = evaluate_model(global_model, test_loader, config.device)
    print(f"✅ 联邦学习准确率: {clean_acc:.4f}")

    if clean_acc < 0.5:
        print("❌ 联邦学习性能异常")
        return False
    else:
        print("✅ 联邦学习工作正常")
        return True


def test_attack_components():
    """测试攻击组件"""
    print("🎭 测试攻击组件...")

    from config import Config
    from attack.backdoor import ComprehensiveBackdoorAttack
    from data.loader import get_cifar10_loaders
    import torch

    config = Config()
    train_loader, test_loader, _ = get_cifar10_loaders(config)

    # 创建攻击器
    attacker = ComprehensiveBackdoorAttack(config)

    # 测试单个样本的投毒
    for batch_idx, (data, targets) in enumerate(train_loader):
        if batch_idx >= 1:  # 只测试一个batch
            break

        print(f"原始数据形状: {data.shape}")
        print(f"原始标签: {targets[:10]}")

        # 🔧 修复：使用正确的方法名
        # 测试单个样本投毒
        sample_img = data[0]
        sample_label = targets[0].item()

        print(f"测试样本标签: {sample_label}")

        # 测试投毒方法
        try:
            poisoned_img, poisoned_label = attacker.poison_sample(sample_img, sample_label)
            print(f"投毒后标签: {poisoned_label}")

            # 检查是否有变化
            img_changed = not torch.equal(sample_img, poisoned_img)
            label_changed = sample_label != poisoned_label

            print(f"图像是否改变: {img_changed}")
            print(f"标签是否改变: {label_changed}")

            if img_changed or label_changed:
                print("✅ 攻击组件工作正常")
                return True
            else:
                print("⚠️ 攻击组件可能未生效（可能投毒率较低）")
                # 尝试多个样本
                changed_count = 0
                for i in range(min(10, len(data))):
                    test_img = data[i]
                    test_label = targets[i].item()
                    p_img, p_label = attacker.poison_sample(test_img, test_label)
                    if not torch.equal(test_img, p_img) or test_label != p_label:
                        changed_count += 1

                print(f"10个样本中有 {changed_count} 个被投毒")
                return changed_count > 0

        except AttributeError as e:
            print(f"❌ 攻击器缺少 poison_sample 方法: {e}")
            return False
        except Exception as e:
            print(f"❌ 攻击组件测试异常: {e}")
            return False


def main():
    """主调试函数"""
    print("🔧 开始联邦学习系统调试...")
    print("=" * 60)

    # 测试步骤
    tests = [
        ("基础模型测试", test_basic_model),
        ("数据分布测试", test_data_distribution),
        ("简化联邦学习测试", test_simple_federated),
        ("攻击组件测试", test_attack_components),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            print(f"{'✅ 通过' if result else '❌ 失败'}")
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results[test_name] = False

    # 总结
    print("\n" + "="*60)
    print("🔧 调试结果总结:")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")

    passed_tests = sum(results.values())
    total_tests = len(results)

    print(f"\n通过率: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("🎉 所有测试通过，可以运行完整实验！")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")

        # 给出建议
        if not results.get("基础模型测试", False):
            print("建议: 检查模型定义和数据加载")
        if not results.get("数据分布测试", False):
            print("建议: 调整联邦数据分割参数")
        if not results.get("简化联邦学习测试", False):
            print("建议: 检查FedAvg聚合算法和通信过程")
        if not results.get("攻击组件测试", False):
            print("建议: 检查后门攻击实现")


if __name__ == "__main__":
    main()