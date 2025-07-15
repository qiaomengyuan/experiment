# =============================================================================
# federated/aggregator.py   # FedAvg聚合
# =============================================================================
import torch
from collections import OrderedDict


class FedAvgAggregator:
    def aggregate(self, model_states):
        if not model_states:
            return None

        aggregated_state = OrderedDict()
        num_models = len(model_states)

        for key in model_states[0].keys():
            # 确保初始化为浮点类型
            aggregated_state[key] = torch.zeros_like(model_states[0][key], dtype=torch.float32)

            for model_state in model_states:
                # 转换为浮点类型进行计算
                aggregated_state[key] += model_state[key].float()

            aggregated_state[key] /= num_models

            # 如果原始参数是整数类型，转换回去
            if model_states[0][key].dtype in [torch.int64, torch.long]:
                aggregated_state[key] = aggregated_state[key].long()
            elif model_states[0][key].dtype in [torch.int32, torch.int]:
                aggregated_state[key] = aggregated_state[key].int()

        return aggregated_state
