import torch

class QLinear:
    """
    QLinear is a quantized linear layer that can be used in neural networks.
    """
    def __init__(self, in_features, out_features, bias=True, quantize=False):
        super(QLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None
        self.needs_quantization = quantize

    def __call__(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)
    
    def set_weight(self, weight):
        """
        Set the weight for the QLinear layer.
        
        Args:
            weight (torch.Tensor): The weight tensor to set.
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"Weight shape {weight.shape} does not match expected shape {(self.out_features, self.in_features)}")
        self.weight = weight