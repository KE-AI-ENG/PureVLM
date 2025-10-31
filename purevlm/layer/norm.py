import torch

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = None
        self.bias = None

    def __call__(self, x):
        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    
class RMSNorm:
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = None
        self.eps = eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)