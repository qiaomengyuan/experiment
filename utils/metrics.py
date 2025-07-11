# =============================================================================
# utils/metrics.py 评估指标
# =============================================================================
import torch
import numpy as np


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def evaluate_attack_success_rate(model, test_loader, attacker, device):
    model.eval()
    attack_success = 0
    total_attacks = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            # 只对非目标类测试攻击
            non_target_mask = target != attacker.config.target_label
            if non_target_mask.sum() == 0:
                continue

            non_target_data = data[non_target_mask]

            # 对每个样本添加触发器
            poisoned_batch = []
            for img in non_target_data:
                poisoned_img = attacker.create_test_sample(img)
                poisoned_batch.append(poisoned_img)

            if poisoned_batch:
                poisoned_tensor = torch.stack(poisoned_batch).to(device)
                outputs = model(poisoned_tensor)
                _, predicted = torch.max(outputs, 1)

                # 计算攻击成功的数量
                attack_success += (predicted == attacker.config.target_label).sum().item()
                total_attacks += len(poisoned_batch)

    if total_attacks == 0:
        return 0.0

    asr = attack_success / total_attacks
    print(f"攻击成功评估: {attack_success}/{total_attacks} = {asr:.4f}")
    return asr
