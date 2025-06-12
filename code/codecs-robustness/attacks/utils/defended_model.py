import torch

class MetricModel(torch.nn.Module):
    def __init__(self, model, defence, device):
        super().__init__()
        self.device = device
        self.model = model
        self.lower_better = model.lower_better
        self.defence = defence
    
    def forward(self, image, **kwargs):
        image = self.defence(image)
        return self.model(image, **kwargs)

class CodecModel(torch.nn.Module):
    def __init__(self, model, defence, device):
        super().__init__()
        self.device = device
        self.model = model
        self.defence = defence
        if hasattr(model, 'input_range'):
            self.input_range = model.input_range
        if hasattr(model, 'output_range'):
            self.output_range = model.output_range
        if hasattr(model, 'output_cspace'):
            self.output_cspace = model.output_cspace
        
    
    def forward(self, image, **kwargs):
        image = self.defence(image)
        return self.model(image, **kwargs)