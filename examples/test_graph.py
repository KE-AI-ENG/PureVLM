import torch
import time

class KVCache:
    def __init__(self, batch_size, num_heads, head_dim, max_seq_len, device=None, dtype=torch.float16):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cuda")
        self.dtype = dtype
        self.k_cache = torch.empty((batch_size, num_heads, max_seq_len, head_dim), device=self.device, dtype=dtype)
        self.v_cache = torch.empty((batch_size, num_heads, max_seq_len, head_dim), device=self.device, dtype=dtype)
        self.cur_len = 0

    def update(self, k, v):
        step_len = k.shape[2]
        self.k_cache[:, :, self.cur_len:self.cur_len+step_len, :] = k
        self.v_cache[:, :, self.cur_len:self.cur_len+step_len, :] = v
        self.cur_len += step_len
        return self.k_cache[:, :, :self.cur_len, :], self.v_cache[:, :, :self.cur_len, :]

    def reset(self):
        self.cur_len = 0


class DecodeLayer(torch.nn.Module):
    def __init__(self, num_heads, head_dim, max_seq_len):
        super().__init__()
        embed_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x, kv_cache: KVCache):
        bsz = x.size(0)
        q = self.q_proj(x).view(bsz, self.num_heads, 1, self.head_dim)
        k = self.k_proj(x).view(bsz, self.num_heads, 1, self.head_dim)
        v = self.v_proj(x).view(bsz, self.num_heads, 1, self.head_dim)
        full_k, full_v = kv_cache.update(k, v)
        attn_scores = torch.matmul(q, full_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, full_v)
        attn_output = attn_output.view(bsz, -1)
        return self.out_proj(attn_output)


class TransformerDecoder(torch.nn.Module):
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            DecodeLayer(num_heads, head_dim, max_seq_len)
            for _ in range(num_layers)
        ])

    def forward(self, x, kv_caches):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, kv_cache)
        return x


class CUDAGraphDecoder:
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, batch_size=1, dtype=torch.float16):
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.model = TransformerDecoder(num_layers, num_heads, head_dim, max_seq_len).to(self.device).to(dtype)
        self.kv_caches = [
            KVCache(batch_size, num_heads, head_dim, max_seq_len, device=self.device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.static_input = torch.empty((batch_size, num_heads * head_dim), device=self.device, dtype=dtype)
        self.static_output = torch.empty_like(self.static_input)
        self.graph = torch.cuda.CUDAGraph()
        self._warmup_and_capture()

    def _warmup_and_capture(self):
        for cache in self.kv_caches:
            cache.reset()
        for _ in range(3):
            _ = self.model(self.static_input, self.kv_caches)
        torch.cuda.synchronize()
        for cache in self.kv_caches:
            cache.reset()
        with torch.cuda.graph(self.graph):
            self.static_output.copy_(self.model(self.static_input, self.kv_caches))

    def decode_step_graph(self, token_embedding):
        self.static_input.copy_(token_embedding)
        self.graph.replay()
        return self.static_output.clone()

    def decode_step_normal(self, token_embedding):
        return self.model(token_embedding, self.kv_caches)

    def reset(self):
        for cache in self.kv_caches:
            cache.reset()


# ===== 性能测试 =====
if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.manual_seed(0)

    num_layers = 8
    num_heads = 8
    head_dim = 64
    max_seq_len = 128
    batch_size = 1
    steps = 200  # decode步数

    decoder = CUDAGraphDecoder(num_layers, num_heads, head_dim, max_seq_len, batch_size=batch_size)

    # 测试普通模式
    decoder.reset()
    token_emb = torch.randn((batch_size, num_heads * head_dim), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        _ = decoder.decode_step_normal(token_emb)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"普通模式耗时: {t1 - t0:.4f} 秒, 每步 {1000*(t1-t0)/steps:.3f} ms")

    # 测试 CUDA Graph 模式
    decoder.reset()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        _ = decoder.decode_step_graph(token_emb)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"CUDA Graph 模式耗时: {t1 - t0:.4f} 秒, 每步 {1000*(t1-t0)/steps:.3f} ms")