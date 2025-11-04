#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <cassert>

using namespace torch::indexing;

// ============ LLaMA配置类 ============
class LlamaConfig {
public:
    int64_t dim = 4096;
    int64_t n_layers = 32;
    int64_t n_heads = 32;
    int64_t n_kv_heads = 32;
    int64_t vocab_size = 32000;
    int64_t hidden_dim = 11008;
    int64_t max_seq_len = 2048;
    torch::Device device = torch::kCUDA;
    torch::Dtype dtype = torch::kFloat32;
    
    LlamaConfig() {}
    LlamaConfig(int64_t dim, int64_t n_layers, int64_t n_heads, int64_t n_kv_heads, 
               int64_t vocab_size, int64_t hidden_dim, int64_t max_seq_len)
        : dim(dim), n_layers(n_layers), n_heads(n_heads), n_kv_heads(n_kv_heads),
          vocab_size(vocab_size), hidden_dim(hidden_dim), max_seq_len(max_seq_len) {}
};

// ============ RMSNorm层 ============
class RMSNormImpl : public torch::nn::Module {
private:
    torch::Tensor weight;
    double eps;
    
public:
    RMSNormImpl(int64_t dim, double eps = 1e-5) : eps(eps) {
        weight = register_parameter("weight", torch::ones({dim}));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        auto variance = x.pow(2).mean(-1, true);
        auto x_normed = x * torch::rsqrt(variance + eps);
        return x_normed * weight;
    }
};
TORCH_MODULE(RMSNorm);

// ============ RoPE位置编码 ============
class RoPEImpl : public torch::nn::Module {
private:
    torch::Tensor freqs_cos, freqs_sin;
    int64_t dim;
    
public:
    RoPEImpl(int64_t dim, int64_t max_seq_len = 2048, double base = 10000.0) : dim(dim) {
        // 预计算旋转频率
        auto freqs = 1.0 / torch::pow(base, torch::arange(0, dim, 2).to(torch::kFloat32) / dim);
        auto t = torch::arange(max_seq_len).to(torch::kFloat32);
        auto freqs_outer = torch::outer(t, freqs);
        
        freqs_cos = register_buffer("freqs_cos", torch::cos(freqs_outer));
        freqs_sin = register_buffer("freqs_sin", torch::sin(freqs_outer));
    }
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor q, torch::Tensor k, int64_t start_pos) {
        auto seq_len = q.size(1);
        auto cos = freqs_cos.index({Slice(start_pos, start_pos + seq_len)}).unsqueeze(0).unsqueeze(2);
        auto sin = freqs_sin.index({Slice(start_pos, start_pos + seq_len)}).unsqueeze(0).unsqueeze(2);
        
        auto apply_rope = [&](torch::Tensor x) {
            auto x1 = x.index({"...", Slice(None, None, 2)});  // 偶数位置
            auto x2 = x.index({"...", Slice(1, None, 2)});     // 奇数位置
            
            auto rotated = torch::stack({
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos
            }, -1).flatten(-2);
            
            return rotated;
        };
        
        return std::make_pair(apply_rope(q), apply_rope(k));
    }
};
TORCH_MODULE(RoPE);

// ============ KV缓存管理 ============
class KVCache {
private:
    std::vector<torch::Tensor> k_caches;
    std::vector<torch::Tensor> v_caches;
    std::vector<int64_t> seq_lens;
    
public:
    KVCache() {}
    
    KVCache(const LlamaConfig& config) {
        int64_t head_dim = config.dim / config.n_heads;
        seq_lens.assign(config.n_layers, 0);
        
        auto options = torch::TensorOptions()
            .dtype(config.dtype)
            .device(config.device);
        
        for (int64_t i = 0; i < config.n_layers; ++i) {
            k_caches.push_back(torch::zeros({1, config.max_seq_len, config.n_kv_heads, head_dim}, options));
            v_caches.push_back(torch::zeros({1, config.max_seq_len, config.n_kv_heads, head_dim}, options));
        }
    }
    
    void update(int64_t layer_idx, torch::Tensor k, torch::Tensor v, int64_t pos) {
        // k, v shape: [1, 1, n_kv_heads, head_dim]
        k_caches[layer_idx].index_put_({0, pos}, k.squeeze(1));
        v_caches[layer_idx].index_put_({0, pos}, v.squeeze(1));
        seq_lens[layer_idx] = std::max(seq_lens[layer_idx], pos + 1);
    }
    
    std::pair<torch::Tensor, torch::Tensor> get(int64_t layer_idx) {
        int64_t seq_len = seq_lens[layer_idx];
        auto k = k_caches[layer_idx].index({0, Slice(0, seq_len)});  // [seq_len, n_kv_heads, head_dim]
        auto v = v_caches[layer_idx].index({0, Slice(0, seq_len)});  // [seq_len, n_kv_heads, head_dim]
        return std::make_pair(k, v);
    }
    
    int64_t get_seq_len(int64_t layer_idx) const { return seq_lens[layer_idx]; }
    
    void reset() {
        std::fill(seq_lens.begin(), seq_lens.end(), 0);
    }
};

// ============ 多头注意力 ============
class MultiHeadAttentionImpl : public torch::nn::Module {
private:
    int64_t dim, n_heads, n_kv_heads, head_dim, group_size;
    torch::nn::Linear wq{nullptr}, wk{nullptr}, wv{nullptr}, wo{nullptr};
    RoPE rope{nullptr};
    
public:
    MultiHeadAttentionImpl(const LlamaConfig& config) 
        : dim(config.dim), n_heads(config.n_heads), n_kv_heads(config.n_kv_heads) {
        
        assert(dim % n_heads == 0);
        head_dim = dim / n_heads;
        group_size = n_heads / n_kv_heads;
        
        // 初始化线性层
        wq = register_module("wq", torch::nn::Linear(dim, n_heads * head_dim));
        wk = register_module("wk", torch::nn::Linear(dim, n_kv_heads * head_dim));
        wv = register_module("wv", torch::nn::Linear(dim, n_kv_heads * head_dim));
        wo = register_module("wo", torch::nn::Linear(dim, dim));
        
        // 初始化RoPE
        rope = register_module("rope", RoPE(head_dim, config.max_seq_len));
        
        // 移动到指定设备
        this->to(config.device, config.dtype);
    }
    
    torch::Tensor forward(torch::Tensor x, KVCache& kv_cache, int64_t layer_idx, int64_t start_pos) {
        auto batch_size = x.size(0);
        auto seq_len = x.size(1);
        
        // 1. 计算Q, K, V
        auto q = wq->forward(x).view({batch_size, seq_len, n_heads, head_dim});
        auto k = wk->forward(x).view({batch_size, seq_len, n_kv_heads, head_dim});
        auto v = wv->forward(x).view({batch_size, seq_len, n_kv_heads, head_dim});
        
        // 2. 应用RoPE
        auto [q_rope, k_rope] = rope->forward(q, k, start_pos);
        
        // 3. 更新KV缓存
        if (seq_len == 1) {  // 解码阶段
            kv_cache.update(layer_idx, k_rope, v, start_pos);
        }
        
        // 4. 获取完整的K, V
        torch::Tensor keys, values;
        if (seq_len == 1) {  // 解码阶段，从缓存获取
            std::tie(keys, values) = kv_cache.get(layer_idx);
            keys = keys.transpose(0, 1).unsqueeze(0);      // [1, seq_len, n_kv_heads, head_dim]
            values = values.transpose(0, 1).unsqueeze(0);  // [1, seq_len, n_kv_heads, head_dim]
        } else {  // 预填充阶段
            keys = k_rope;
            values = v;
            // 更新缓存
            for (int64_t i = 0; i < seq_len; ++i) {
                kv_cache.update(layer_idx, k_rope.index({Slice(), i}).unsqueeze(1), 
                               v.index({Slice(), i}).unsqueeze(1), start_pos + i);
            }
        }
        
        // 5. 计算注意力
        auto output = compute_attention(q_rope, keys, values, start_pos);
        
        // 6. 输出投影
        return wo->forward(output);
    }
    
private:
    torch::Tensor compute_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, int64_t start_pos) {
        auto batch_size = q.size(0);
        auto q_len = q.size(1);
        auto kv_len = k.size(1);
        
        // Grouped Query Attention: 复制KV heads
        if (group_size > 1) {
            k = k.repeat_interleave(group_size, 2);  // [batch, kv_len, n_heads, head_dim]
            v = v.repeat_interleave(group_size, 2);
        }
        
        // 转置为 [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);
        
        // 计算注意力分数
        auto scale = 1.0 / std::sqrt(head_dim);
        auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;
        
        // 应用因果掩码
        if (q_len > 1) {  // 预填充阶段需要掩码
            auto mask = torch::triu(torch::ones({q_len, kv_len}, q.options()), 1);
            scores = scores.masked_fill(mask.to(torch::kBool), -std::numeric_limits<float>::infinity());
        } else if (kv_len > 1) {  // 解码阶段，只关注当前位置之前的tokens
            // 不需要额外掩码，因为KV缓存只包含有效的历史tokens
        }
        
        // Softmax + 注意力加权
        auto attn_weights = torch::softmax(scores, -1);
        auto output = torch::matmul(attn_weights, v);
        
        // 转回 [batch, seq_len, n_heads, head_dim] 并合并heads
        output = output.transpose(1, 2).contiguous().view({batch_size, q_len, dim});
        
        return output;
    }
};
TORCH_MODULE(MultiHeadAttention);

// ============ MLP层 ============
class MLPImpl : public torch::nn::Module {
private:
    torch::nn::Linear w_gate{nullptr}, w_up{nullptr}, w_down{nullptr};
    
public:
    MLPImpl(const LlamaConfig& config) {
        w_gate = register_module("w_gate", torch::nn::Linear(config.dim, config.hidden_dim));
        w_up = register_module("w_up", torch::nn::Linear(config.dim, config.hidden_dim));
        w_down = register_module("w_down", torch::nn::Linear(config.hidden_dim, config.dim));
        
        this->to(config.device, config.dtype);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // SwiGLU激活函数: SiLU(W_gate * x) * (W_up * x)
        auto gate = torch::silu(w_gate->forward(x));
        auto up = w_up->forward(x);
        auto hidden = gate * up;
        return w_down->forward(hidden);
    }
};
TORCH_MODULE(MLP);

// ============ Transformer块 ============
class TransformerBlockImpl : public torch::nn::Module {
private:
    MultiHeadAttention attention{nullptr};
    MLP mlp{nullptr};
    RMSNorm attention_norm{nullptr};
    RMSNorm ffn_norm{nullptr};
    
public:
    TransformerBlockImpl(const LlamaConfig& config) {
        attention = register_module("attention", MultiHeadAttention(config));
        mlp = register_module("mlp", MLP(config));
        attention_norm = register_module("attention_norm", RMSNorm(config.dim));
        ffn_norm = register_module("ffn_norm", RMSNorm(config.dim));
        
        this->to(config.device, config.dtype);
    }
    
    torch::Tensor forward(torch::Tensor x, KVCache& kv_cache, int64_t layer_idx, int64_t start_pos) {
        // 1. 注意力分支: x + Attention(RMSNorm(x))
        auto attn_output = attention->forward(attention_norm->forward(x), kv_cache, layer_idx, start_pos);
        x = x + attn_output;
        
        // 2. MLP分支: x + MLP(RMSNorm(x))
        auto mlp_output = mlp->forward(ffn_norm->forward(x));
        x = x + mlp_output;
        
        return x;
    }
};
TORCH_MODULE(TransformerBlock);

// ============ LLaMA模型 ============
class LlamaModelImpl : public torch::nn::Module {
private:
    LlamaConfig config;
    torch::nn::Embedding tok_embeddings{nullptr};
    torch::nn::ModuleList layers;
    RMSNorm norm{nullptr};
    torch::nn::Linear lm_head{nullptr};
    
    std::unique_ptr<KVCache> kv_cache;
    
public:
    LlamaModelImpl(const LlamaConfig& cfg) : config(cfg) {
        // 初始化嵌入层
        tok_embeddings = register_module("tok_embeddings", 
            torch::nn::Embedding(config.vocab_size, config.dim));
        
        // 初始化Transformer层
        layers = register_module("layers", torch::nn::ModuleList());
        for (int64_t i = 0; i < config.n_layers; ++i) {
            layers->push_back(TransformerBlock(config));
        }
        
        // 最终归一化和输出层
        norm = register_module("norm", RMSNorm(config.dim));
        lm_head = register_module("lm_head", 
            torch::nn::Linear(torch::nn::LinearOptions(config.dim, config.vocab_size).bias(false)));
        
        // 移动到指定设备
        this->to(config.device, config.dtype);
        
        // 初始化KV缓存
        kv_cache = std::make_unique<KVCache>(config);
        
        // 权重初始化
        initialize_weights();
        
        std::cout << "模型参数数量: " << count_parameters() << std::endl;
        std::cout << "GPU显存使用: " << get_gpu_memory_usage() << " MB" << std::endl;
    }
    
    torch::Tensor forward(torch::Tensor tokens, int64_t start_pos = 0) {
        auto batch_size = tokens.size(0);
        auto seq_len = tokens.size(1);
        
        // 1. Token嵌入
        auto x = tok_embeddings->forward(tokens);
        
        // 2. 通过所有Transformer层
        for (int64_t i = 0; i < config.n_layers; ++i) {
            auto layer = layers[i]->as<TransformerBlock>();
            x = layer->forward(x, *kv_cache, i, start_pos);
        }
        
        // 3. 最终归一化
        x = norm->forward(x);
        
        // 4. 输出投影
        auto logits = lm_head->forward(x);
        
        return logits;
    }
    
    // 单步推理（解码阶段）
    torch::Tensor forward_step(int64_t token_id, int64_t pos) {
        auto tokens = torch::tensor({{token_id}}, torch::TensorOptions()
            .dtype(torch::kLong).device(config.device));
        
        torch::NoGradGuard no_grad;  // 推理时不需要梯度
        auto logits = this->forward(tokens, pos);
        
        return logits.index({0, 0});  // 返回 [vocab_size]
    }
    
    // 预填充阶段
    torch::Tensor prefill(const std::vector<int64_t>& token_ids) {
        auto tokens = torch::tensor({token_ids}, torch::TensorOptions()
            .dtype(torch::kLong).device(config.device));
        
        torch::NoGradGuard no_grad;
        auto logits = this->forward(tokens, 0);
        
        return logits.index({0, -1});  // 返回最后一个位置的logits
    }
    
    // 贪心采样
    int64_t sample_greedy(torch::Tensor logits) {
        return torch::argmax(logits).item<int64_t>();
    }
    
    // Top-k采样
    int64_t sample_top_k(torch::Tensor logits, int64_t k = 50, double temperature = 1.0) {
        if (temperature != 1.0) {
            logits = logits / temperature;
        }
        
        auto [top_k_values, top_k_indices] = torch::topk(logits, k);
        auto probs = torch::softmax(top_k_values, -1);
        auto sampled_idx = torch::multinomial(probs, 1);
        
        return top_k_indices.index({sampled_idx}).item<int64_t>();
    }
    
    // 重置KV缓存
    void reset_kv_cache() {
        kv_cache->reset();
    }
    
    // 生成文本
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt, int64_t max_new_tokens = 100,
                                 double temperature = 1.0, int64_t top_k = 50) {
        reset_kv_cache();
        
        std::vector<int64_t> generated = prompt;
        
        // 预填充
        auto logits = prefill(prompt);
        auto next_token = (temperature > 0) ? sample_top_k(logits, top_k, temperature) 
                                           : sample_greedy(logits);
        generated.push_back(next_token);
        
        // 自回归生成
        int64_t pos = prompt.size();
        for (int64_t i = 0; i < max_new_tokens - 1; ++i) {
            logits = forward_step(next_token, pos);
            next_token = (temperature > 0) ? sample_top_k(logits, top_k, temperature) 
                                          : sample_greedy(logits);
            generated.push_back(next_token);
            pos++;
            
            // 可以添加停止条件，如遇到EOS token
            if (next_token == 2) break;  // 假设2是EOS token
        }
        
        return generated;
    }
    
    const LlamaConfig& get_config() const { return config; }
    
    // 保存模型
    void save(const std::string& path) {
        torch::save(this->parameters(), path);
    }
    
    // 加载模型
    void load(const std::string& path) {
        torch::load(this->parameters(), path);
    }
    
private:
    void initialize_weights() {
        // 使用Xavier/Glorot初始化
        for (auto& param : this->named_parameters()) {
            if (param.key().find("weight") != std::string::npos) {
                if (param.value().dim() >= 2) {
                    torch::nn::init::xavier_uniform_(param.value());
                } else {
                    torch::nn::init::normal_(param.value(), 0, 0.02);
                }
            }
        }
    }
    
    int64_t count_parameters() {
        int64_t total = 0;
        for (const auto& param : this->parameters()) {
            total += param.numel();
        }
        return total;
    }
    
    double get_gpu_memory_usage() {
        if (torch::cuda::is_available()) {
            return torch::cuda::memory_allocated() / (1024.0 * 1024.0);
        }
        return 0.0;
    }
};
TORCH_MODULE(LlamaModel);

// ============ 主函数 ============
int main() {
    // 检查CUDA可用性
    if (!torch::cuda::is_available()) {
        std::cout << "CUDA不可用，使用CPU" << std::endl;
    } else {
        std::cout << "使用GPU: " << torch::cuda::get_device_count() << " 个设备可用" << std::endl;
    }
    
    // 创建配置
    LlamaConfig config(512, 4, 8, 4, 32000, 1536, 256);
    if (torch::cuda::is_available()) {
        config.device = torch::kCUDA;
    } else {
        config.device = torch::kCPU;
    }
    
    // 创建模型
    auto model = LlamaModel(config);
    model->eval();  // 设置为评估模式
    
    // 测试推理
    std::vector<int64_t> prompt = {1, 42, 123, 7, 256};
    
    std::cout << "\n开始推理..." << std::endl;
    std::cout << "输入prompt: ";
    for (auto token : prompt) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    // 生成文本
    auto start = std::chrono::high_resolution_clock::now();
    auto generated = model->generate(prompt, 20, 0.8, 50);  // temperature=0.8, top_k=50
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "生成结果: ";
    for (auto token : generated) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "生成耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "生成速度: " << (generated.size() - prompt.size()) * 1000.0 / duration.count() 
              << " tokens/s" << std::endl;
    
    // 显存使用情况
    if (torch::cuda::is_available()) {
        std::cout << "当前GPU显存使用: " << torch::cuda::memory_allocated() / (1024*1024) << " MB" << std::endl;
        std::cout << "峰值GPU显存使用: " << torch::cuda::max_memory_allocated() / (1024*1024) << " MB" << std::endl;
    }
    
    return 0;
}