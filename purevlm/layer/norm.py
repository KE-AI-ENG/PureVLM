import torch
import purevlm.cuda_ops as cuda_ops

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
        output_tensor = torch.empty_like(hidden_states)
        cuda_ops.rms_norm_(output_tensor, hidden_states, self.weight, self.eps)
        return output_tensor