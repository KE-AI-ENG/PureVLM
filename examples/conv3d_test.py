import torch
import time

temporal_patch_size = 2
patch_size = 16
in_channels = 3
embed_dim = 1152
kernel_size = [temporal_patch_size, patch_size, patch_size]

conv3d_weight = torch.randn((embed_dim, in_channels, temporal_patch_size, patch_size, patch_size), dtype=torch.bfloat16, device='cuda')
conv3d_bias = torch.randn((embed_dim,), dtype=torch.bfloat16, device='cuda')
hidden_states = torch.randn((1500,1536), dtype=torch.bfloat16, device='cuda')

def conv3d_run(input_tensor, weight, bias):
    hidden_states = input_tensor.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)
    hidden_states = torch.nn.functional.conv3d(hidden_states, weight, bias, stride=(temporal_patch_size, patch_size, patch_size), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1)
    return hidden_states.view(-1, embed_dim)

#warmup
_ = conv3d_run(hidden_states, conv3d_weight, conv3d_bias)
torch.cuda.synchronize()

start_time = time.time()
conv3d_output = conv3d_run(hidden_states, conv3d_weight, conv3d_bias)
torch.cuda.synchronize()
print(f"Conv3D forward time: {time.time() - start_time:.4f} seconds")