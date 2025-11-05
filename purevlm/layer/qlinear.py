import torch

from .gemmw4a16_marlin.preprocess_for_weight import preprocess, run

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
        # marlin
        self.using_marlin = False
        self.marlin_q_w = None
        self.marlin_s = None
        self.marlin_zp = None
        self.g_idx = None 
        self.sort_indices = None
        self.workspace = None
        self.N = None

    def __call__(self, input):
        if self.needs_quantization and self.using_marlin:
            return self.run_marlin_linear(input)    # 调用marlin算子
        else:
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

    def set_using_marlin(self):
        self.needs_quantization = True
        self.using_marlin = True
    
    def get_marlin_status(self):
        return self.needs_quantization, self.using_marlin

    def marlin_weight_preprocess(self, weight):
        self.N = weight.shape[0]
        self.marlin_q_w, self.marlin_s, self.marlin_zp, self.g_idx, self.sort_indices, self.workspace = preprocess(weight, self.N, group_size=128)

    def run_marlin_linear(self,
                          input: torch.Tensor, # shape[m, k]/[1, m, k]
        ):
        output = run(input,
                    self.marlin_q_w,
                    None,
                    self.marlin_s,
                    self.marlin_zp,
                    self.g_idx,
                    self.sort_indices,
                    self.workspace,
                    self.N)

        return output