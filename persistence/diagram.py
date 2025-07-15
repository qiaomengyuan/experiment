# =============================================================================
# persistence/diagram.py  持久图生成
# =============================================================================
import numpy as np


class DiagramGenerator:
    def __init__(self, calculator):
        self.calculator = calculator

    def generate_diagrams(self, models, dataloader):
        diagrams = []
        for i, model in enumerate(models):
            model = model.to(self.calculator.config.device)
            diagram = self.calculator.generate_diagram_for_model(model, dataloader)
            diagrams.append(diagram)
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{len(models)} persistence diagrams")

        return np.array(diagrams)
