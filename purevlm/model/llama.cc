#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

using namespace std;

// ============ CUDA错误检查宏 ============
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

// ============ 显存池管理器 ============
class GPUMemoryPool {
private:
    std::vector<void*> allocated_ptrs;
    size_t total_allocated = 0;
    
public:
    ~GPUMemoryPool() {
        for (void* ptr : allocated_ptrs) {
            cudaFree(ptr);
        }
    }
    
    float* allocate(size_t size_in_floats) {
        float* ptr;
        size_t bytes = size_in_floats * sizeof(float);
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        allocated_ptrs.push_back(ptr);
        total_allocated += bytes;
        return ptr;
    }
    
    size_t get_total_allocated() const { return total_allocated; }
};

// ============ GPU张量类（不拥有内存） ============
class GPUTensor {
private:
    float* d_data;
    std::vector<int> shape_;
    size_t size_;
    
public:
    GPUTensor() : d_data(nullptr), size_(0) {}
    
    GPUTensor(float* data, const std::vector<int>& shape) 
        : d_data(data), shape_(shape) {
        size_ = 1;
        for (int dim : shape) size_ *= dim;
    }
    
    float* data() { return d_data; }
    const float* data() const { return d_data; }
    size_t size() const { return size_; }
    const std::vector<int>& shape() const { return shape_; }
    int dim(int i) const { return shape_[i]; }
    int ndim() const { return shape_.size(); }
    
    // 重新绑定到不同的内存地址（用于复用activation buffer）
    void rebind(float* data, const std::vector<int>& shape) {
        d_data = data;
        shape_ = shape;
        size_ = 1;
        for (int dim : shape) size_ *= dim;
    }
    
    // 从CPU数据拷贝到GPU
    void copy_from_host(const std::vector<float>& host_data) {
        assert(host_data.size() == size_);
        CUDA_CHECK(cudaMemcpy(d_data, host_data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // 从GPU拷贝到CPU
    std::vector<float> copy_to_host() const {
        std::vector<float> host_data(size_);
        CUDA_CHECK(cudaMemcpy(host_data.data(), d_data, size_ * sizeof(float), cudaMemcpyDeviceToHost));
        return host_data;
    }
    
    // 清零
    void zero() {
        CUDA_CHECK(cudaMemset(d_data, 0, size_ * sizeof(float)));
    }
};

// ============ CUDA核函数声明 ============
extern "C" {
    void launch_rmsnorm_kernel(float* x, const float* weight, int size, float eps, cudaStream_t stream);
    void launch_rope_kernel(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, 
                           int pos, float base, cudaStream_t stream);
    void launch_silu_mul_kernel(float* gate, const float* up, int size, cudaStream_t stream);
    void launch_add_kernel(float* a, const float* b, int size, cudaStream_t stream);
    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream);
    void launch_attention_kernel(const float* q, const float* k_cache, const float* v_cache,
                                float* output, float* scores_buffer,
                                int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos,
                                float scale, cudaStream_t stream);
    void launch_update_kv_cache_kernel(float* k_cache, float* v_cache, const float* k, const float* v,
                                      int n_kv_heads, int head_dim, int max_seq_len, int pos,
                                      cudaStream_t stream);
    void launch_embedding_lookup_kernel(const float* embeddings, float* output, int token_id, 
                                       int vocab_size, int dim, cudaStream_t stream);
}

// ============ GPU算子封装类 ============
class CudaOps {
private:
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
    
public:
    CudaOps() {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    }
    
    ~CudaOps() {
        cublasDestroy(cublas_handle);
        cudaStreamDestroy(stream);
    }
    
    cudaStream_t get_stream() const { return stream; }
    
    // 矩阵乘法: C = A * B
    void gemm(const GPUTensor& A, const GPUTensor& B, GPUTensor& C, 
              bool trans_a = false, bool trans_b = false, float alpha = 1.0f, float beta = 0.0f) {
        assert(A.ndim() == 2 && B.ndim() == 2 && C.ndim() == 2);
        
        int M = A.dim(0), K = A.dim(1), N = B.dim(1);
        assert(B.dim(0) == K && C.dim(0) == M && C.dim(1) == N);
        
        cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        
        // cuBLAS使用列主序，需要转换: C^T = B^T * A^T
        CUBLAS_CHECK(cublasSgemm(cublas_handle, op_b, op_a, N, M, K,
                                &alpha, B.data(), trans_b ? K : N,
                                A.data(), trans_a ? M : K,
                                &beta, C.data(), N));
    }
    
    // 向量-矩阵乘法: y = x * W
    void gemv(const GPUTensor& x, const GPUTensor& W, GPUTensor& y) {
        assert(x.ndim() == 2 && W.ndim() == 2 && y.ndim() == 2);
        assert(x.dim(0) == 1 && y.dim(0) == 1);
        assert(x.dim(1) == W.dim(0) && y.dim(1) == W.dim(1));
        
        int M = W.dim(1), N = W.dim(0);
        float alpha = 1.0f, beta = 0.0f;
        
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T, N, M,
                                &alpha, W.data(), N,
                                x.data(), 1,
                                &beta, y.data(), 1));
    }
    
    // 各种kernel调用的封装
    void rmsnorm(GPUTensor& x, const GPUTensor& weight, float eps = 1e-5f) {
        launch_rmsnorm_kernel(x.data(), weight.data(), x.size(), eps, stream);
    }
    
    void rope(GPUTensor& q, GPUTensor& k, int pos, float base = 10000.0f) {
        int n_heads = q.dim(2);
        int n_kv_heads = k.dim(2);
        int head_dim = q.dim(3);
        launch_rope_kernel(q.data(), k.data(), n_heads, n_kv_heads, head_dim, pos, base, stream);
    }
    
    void silu_mul(GPUTensor& gate, const GPUTensor& up) {
        launch_silu_mul_kernel(gate.data(), up.data(), gate.size(), stream);
    }
    
    void add(GPUTensor& a, const GPUTensor& b) {
        launch_add_kernel(a.data(), b.data(), a.size(), stream);
    }
    
    void embedding_lookup(const GPUTensor& embeddings, GPUTensor& output, int token_id) {
        int vocab_size = embeddings.dim(0);
        int dim = embeddings.dim(1);
        launch_embedding_lookup_kernel(embeddings.data(), output.data(), token_id, vocab_size, dim, stream);
    }
    
    void update_kv_cache(GPUTensor& k_cache, GPUTensor& v_cache, 
                        const GPUTensor& k, const GPUTensor& v, int pos) {
        int n_kv_heads = k_cache.dim(0);
        int max_seq_len = k_cache.dim(1);
        int head_dim = k_cache.dim(2);
        launch_update_kv_cache_kernel(k_cache.data(), v_cache.data(), k.data(), v.data(),
                                     n_kv_heads, head_dim, max_seq_len, pos, stream);
    }
    
    void attention(const GPUTensor& q, const GPUTensor& k_cache, const GPUTensor& v_cache,
                  GPUTensor& output, GPUTensor& scores_buffer, int seq_len, int pos) {
        int n_heads = q.dim(2);
        int n_kv_heads = k_cache.dim(0);
        int head_dim = q.dim(3);
        float scale = 1.0f / sqrtf(head_dim);
        
        launch_attention_kernel(q.data(), k_cache.data(), v_cache.data(), output.data(),
                               scores_buffer.data(), n_heads, n_kv_heads, head_dim, 
                               seq_len, pos, scale, stream);
    }
    
    void sync() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// ============ 预分配的激活缓存 ============
class ActivationBuffers {
private:
    GPUMemoryPool* pool;
    std::unordered_map<std::string, GPUTensor> buffers;
    
public:
    ActivationBuffers(GPUMemoryPool* pool, const class LlamaConfig& config) : pool(pool) {
        allocate_buffers(config);
    }
    
    void allocate_buffers(const class LlamaConfig& config);
    
    GPUTensor& get(const std::string& name) {
        auto it = buffers.find(name);
        assert(it != buffers.end());
        return it->second;
    }
    
    // 获取临时buffer，可以复用
    GPUTensor get_temp_buffer(const std::string& base_name, const std::vector<int>& shape) {
        std::string name = base_name + "_temp";
        auto it = buffers.find(name);
        if (it != buffers.end()) {
            it->second.rebind(it->second.data(), shape);
            return it->second;
        }
        
        size_t size = 1;
        for (int dim : shape) size *= dim;
        float* ptr = pool->allocate(size);
        GPUTensor tensor(ptr, shape);
        buffers[name] = tensor;
        return tensor;
    }
};

// ============ LLaMA配置类 ============
class LlamaConfig {
public:
    int dim = 4096;
    int n_layers = 32;
    int n_heads = 32;
    int n_kv_heads = 32;
    int vocab_size = 32000;
    int hidden_dim = 11008;
    int max_seq_len = 2048;
    
    LlamaConfig() {}
    LlamaConfig(int dim, int n_layers, int n_heads, int n_kv_heads, 
               int vocab_size, int hidden_dim, int max_seq_len)
        : dim(dim), n_layers(n_layers), n_heads(n_heads), n_kv_heads(n_kv_heads),
          vocab_size(vocab_size), hidden_dim(hidden_dim), max_seq_len(max_seq_len) {}
};

// 实现ActivationBuffers的allocate_buffers方法
void ActivationBuffers::allocate_buffers(const LlamaConfig& config) {
    int head_dim = config.dim / config.n_heads;
    
    // 主要的activation buffers
    buffers["hidden_states"] = GPUTensor(pool->allocate(config.dim), {1, config.dim});
    buffers["residual"] = GPUTensor(pool->allocate(config.dim), {1, config.dim});
    buffers["norm_output"] = GPUTensor(pool->allocate(config.dim), {1, config.dim});
    
    // 注意力相关buffers
    buffers["q"] = GPUTensor(pool->allocate(config.n_heads * head_dim), {1, 1, config.n_heads, head_dim});
    buffers["k"] = GPUTensor(pool->allocate(config.n_kv_heads * head_dim), {1, 1, config.n_kv_heads, head_dim});
    buffers["v"] = GPUTensor(pool->allocate(config.n_kv_heads * head_dim), {1, 1, config.n_kv_heads, head_dim});
    buffers["attn_output"] = GPUTensor(pool->allocate(config.dim), {1, config.dim});
    buffers["scores"] = GPUTensor(pool->allocate(config.max_seq_len), {config.max_seq_len});
    
    // MLP相关buffers
    buffers["gate"] = GPUTensor(pool->allocate(config.hidden_dim), {1, config.hidden_dim});
    buffers["up"] = GPUTensor(pool->allocate(config.hidden_dim), {1, config.hidden_dim});
    buffers["mlp_output"] = GPUTensor(pool->allocate(config.dim), {1, config.dim});
    
    // 最终输出
    buffers["logits"] = GPUTensor(pool->allocate(config.vocab_size), {1, config.vocab_size});
}

// ============ KV缓存类 ============
class KVCache {
private:
    std::vector<GPUTensor> k_caches;  // 每层的K缓存
    std::vector<GPUTensor> v_caches;  // 每层的V缓存
    std::vector<int> seq_lens;        // 每层当前序列长度
    
public:
    KVCache() {}
    
    KVCache(GPUMemoryPool* pool, const LlamaConfig& config) {
        int head_dim = config.dim / config.n_heads;
        seq_lens.assign(config.n_layers, 0);
        
        k_caches.reserve(config.n_layers);
        v_caches.reserve(config.n_layers);
        
        for (int i = 0; i < config.n_layers; ++i) {
            size_t cache_size = config.n_kv_heads * config.max_seq_len * head_dim;
            float* k_ptr = pool->allocate(cache_size);
            float* v_ptr = pool->allocate(cache_size);
            
            k_caches.emplace_back(k_ptr, std::vector<int>{config.n_kv_heads, config.max_seq_len, head_dim});
            v_caches.emplace_back(v_ptr, std::vector<int>{config.n_kv_heads, config.max_seq_len, head_dim});
            
            // 初始化为0
            k_caches[i].zero();
            v_caches[i].zero();
        }
    }
    
    GPUTensor& get_k_cache(int layer) { return k_caches[layer]; }
    GPUTensor& get_v_cache(int layer) { return v_caches[layer]; }
    
    int get_seq_len(int layer) const { return seq_lens[layer]; }
    void set_seq_len(int layer, int len) { seq_lens[layer] = len; }
    
    void reset() {
        std::fill(seq_lens.begin(), seq_lens.end(), 0);
    }
};

// ============ 线性层类 ============
class Linear {
private:
    GPUTensor weight;  // [in_dim, out_dim]
    
public:
    Linear() {}
    
    Linear(GPUMemoryPool* pool, int in_dim, int out_dim) {
        float* w_ptr = pool->allocate(in_dim * out_dim);
        weight = GPUTensor(w_ptr, {in_dim, out_dim});
    }
    
    void forward(const GPUTensor& x, GPUTensor& y, CudaOps& ops) {
        if (x.dim(0) == 1) {
            ops.gemv(x, weight, y);
        } else {
            ops.gemm(x, weight, y);
        }
    }
    
    GPUTensor& get_weight() { return weight; }
    const GPUTensor& get_weight() const { return weight; }
};

// ============ 多头注意力类 ============
class MultiHeadAttention {
private:
    int dim, n_heads, n_kv_heads, head_dim;
    Linear wq, wk, wv, wo;
    
public:
    MultiHeadAttention() {}
    
    MultiHeadAttention(GPUMemoryPool* pool, int dim, int n_heads, int n_kv_heads) 
        : dim(dim), n_heads(n_heads), n_kv_heads(n_kv_heads) {
        
        assert(dim % n_heads == 0);
        head_dim = dim / n_heads;
        
        wq = Linear(pool, dim, n_heads * head_dim);
        wk = Linear(pool, dim, n_kv_heads * head_dim);  
        wv = Linear(pool, dim, n_kv_heads * head_dim);
        wo = Linear(pool, dim, dim);
    }
    
    void forward(const GPUTensor& x, GPUTensor& output, KVCache& kv_cache, 
                int layer_id, int pos, ActivationBuffers& act_buffers, CudaOps& ops) {
        
        // 获取预分配的buffers
        GPUTensor& q = act_buffers.get("q");
        GPUTensor& k = act_buffers.get("k");
        GPUTensor& v = act_buffers.get("v");
        GPUTensor& attn_out = act_buffers.get("attn_output");
        GPUTensor& scores = act_buffers.get("scores");
        
        // 1. 计算Q, K, V (复用临时buffer)
        GPUTensor q_flat = act_buffers.get_temp_buffer("q_flat", {1, n_heads * head_dim});
        GPUTensor k_flat = act_buffers.get_temp_buffer("k_flat", {1, n_kv_heads * head_dim});
        GPUTensor v_flat = act_buffers.get_temp_buffer("v_flat", {1, n_kv_heads * head_dim});
        
        wq.forward(x, q_flat, ops);
        wk.forward(x, k_flat, ops);
        wv.forward(x, v_flat, ops);
        
        // 2. Reshape为多头格式并应用RoPE
        q.rebind(q_flat.data(), {1, 1, n_heads, head_dim});
        k.rebind(k_flat.data(), {1, 1, n_kv_heads, head_dim});
        v.rebind(v_flat.data(), {1, 1, n_kv_heads, head_dim});
        
        ops.rope(q, k, pos);
        
        // 3. 更新KV缓存
        ops.update_kv_cache(kv_cache.get_k_cache(layer_id), kv_cache.get_v_cache(layer_id), k, v, pos);
        kv_cache.set_seq_len(layer_id, std::max(kv_cache.get_seq_len(layer_id), pos + 1));
        
        // 4. 计算注意力
        int seq_len = kv_cache.get_seq_len(layer_id);
        ops.attention(q, kv_cache.get_k_cache(layer_id), kv_cache.get_v_cache(layer_id),
                     attn_out, scores, seq_len, pos);
        
        // 5. 输出投影
        wo.forward(attn_out, output, ops);
    }
    
    Linear& get_wq() { return wq; }
    Linear& get_wk() { return wk; }
    Linear& get_wv() { return wv; }
    Linear& get_wo() { return wo; }
};

// ============ MLP类 ============
class MLP {
private:
    Linear w_gate, w_up, w_down;
    
public:
    MLP() {}
    
    MLP(GPUMemoryPool* pool, int dim, int hidden_dim) {
        w_gate = Linear(pool, dim, hidden_dim);
        w_up = Linear(pool, dim, hidden_dim);
        w_down = Linear(pool, hidden_dim, dim);
    }
    
    void forward(const GPUTensor& x, GPUTensor& output, ActivationBuffers& act_buffers, CudaOps& ops) {
        GPUTensor& gate = act_buffers.get("gate");
        GPUTensor& up = act_buffers.get("up");
        
        // gate = W_gate * x, up = W_up * x
        w_gate.forward(x, gate, ops);
        w_up.forward(x, up, ops);
        
        // gate = silu(gate) * up
        ops.silu_mul(gate, up);
        
        // output = W_down * gate
        w_down.forward(gate, output, ops);
    }
    
    Linear& get_w_gate() { return w_gate; }
    Linear& get_w_up() { return w_up; }
    Linear& get_w_down() { return w_down; }
};

// ============ Transformer层类 ============
class TransformerBlock {
private:
    MultiHeadAttention attention;
    MLP mlp;
    GPUTensor rms_att_weight;
    GPUTensor rms_ffn_weight;
    
public:
    TransformerBlock() {}
    
    TransformerBlock(GPUMemoryPool* pool, int dim, int n_heads, int n_kv_heads, int hidden_dim) 
        : attention(pool, dim, n_heads, n_kv_heads), mlp(pool, dim, hidden_dim) {
        
        // 分配RMSNorm权重
        float* att_w_ptr = pool->allocate(dim);
        float* ffn_w_ptr = pool->allocate(dim);
        rms_att_weight = GPUTensor(att_w_ptr, {dim});
        rms_ffn_weight = GPUTensor(ffn_w_ptr, {dim});
        
        // 初始化为1
        std::vector<float> ones(dim, 1.0f);
        rms_att_weight.copy_from_host(ones);
        rms_ffn_weight.copy_from_host(ones);
    }
    
    void forward(const GPUTensor& x, GPUTensor& output, KVCache& kv_cache, 
                int layer_id, int pos, ActivationBuffers& act_buffers, CudaOps& ops) {
        
        GPUTensor& norm_out = act_buffers.get("norm_output");
        GPUTensor& residual = act_buffers.get("residual");
        GPUTensor& mlp_out = act_buffers.get("mlp_output");
        
        // 1. 注意力分支: x + Attention(RMSNorm(x))
        // 拷贝x到norm_out并应用RMSNorm
        CUDA_CHECK(cudaMemcpy(norm_out.data(), x.data(), x.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        ops.rmsnorm(norm_out, rms_att_weight);
        
        // 计算注意力，结果直接写入residual
        attention.forward(norm_out, residual, kv_cache, layer_id, pos, act_buffers, ops);
        
        // residual connection: residual += x
        ops.add(residual, x);
        
        // 2. MLP分支: residual + MLP(RMSNorm(residual))
        // 拷贝residual到norm_out并应用RMSNorm
        CUDA_CHECK(cudaMemcpy(norm_out.data(), residual.data(), residual.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        ops.rmsnorm(norm_out, rms_ffn_weight);
        
        // 计算MLP
        mlp.forward(norm_out, mlp_out, act_buffers, ops);
        
        // 最终residual connection: output = residual + mlp_out
        CUDA_CHECK(cudaMemcpy(output.data(), residual.data(), residual.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        ops.add(output, mlp_out);
    }
    
    MultiHeadAttention& get_attention() { return attention; }
    MLP& get_mlp() { return mlp; }
    GPUTensor& get_rms_att_weight() { return rms_att_weight; }
    GPUTensor& get_rms_ffn_weight() { return rms_ffn_weight; }
};

// ============ LLaMA模型类 ============
class LlamaModel {
private:
    LlamaConfig config;
    std::unique_ptr<GPUMemoryPool> memory_pool;
    std::unique_ptr<ActivationBuffers> act_buffers;
    std::unique_ptr<KVCache> kv_cache;
    std::unique_ptr<CudaOps> ops;
    
    // 模型权重（都在GPU上）
    GPUTensor tok_embeddings;  // [vocab_size, dim]
    std::vector<TransformerBlock> layers;
    GPUTensor rms_final_weight;  // [dim]
    GPUTensor lm_head;  // [dim, vocab_size]
    
public:
    LlamaModel() {}
    
    LlamaModel(const LlamaConfig& cfg) : config(cfg) {
        // 初始化内存池和CUDA算子
        memory_pool = std::make_unique<GPUMemoryPool>();
        ops = std::make_unique<CudaOps>();
        
        // 分配权重内存
        allocate_weights();
        
        // 分配activation buffers和KV cache
        act_buffers = std::make_unique<ActivationBuffers>(memory_pool.get(), config);
        kv_cache = std::make_unique<KVCache>(memory_pool.get(), config);
        
        std::cout << "Total GPU memory allocated: " 
                  << memory_pool->get_total_allocated() / (1024*1024) << " MB" << std::endl;
    }
    
private:
    void allocate_weights() {
        // Token embeddings
        float* emb_ptr = memory_pool->allocate(config.vocab_size * config.dim);
        tok_embeddings = GPUTensor(emb_ptr, {config.vocab_size, config.dim});
        
        // Transformer layers
        layers.reserve(config.n_layers);
        for (int i = 0; i < config.n_layers; ++i) {
            layers.emplace_back(memory_pool.get(), config.dim, config.n_heads, 
                               config.n_kv_heads, config.hidden_dim);
        }
        
        // Final RMSNorm
        float* final_rms_ptr = memory_pool->allocate(config.dim);
        rms_final_weight = GPUTensor(final_rms_ptr, {config.dim});
        
        // LM head
        float* lm_head_ptr = memory_pool->allocate(config.dim * config.vocab_size);
        lm_head = GPUTensor(lm_head_ptr, {config.dim, config.vocab_size});
        
        // 初始化权重（这里用随机初始化，实际应该加载预训练权重）
        initialize_weights_random();
    }
    
    void initialize_weights_random() {
        // 简单的随机初始化，实际使用时应该加载预训练权重
        auto init_tensor = [](GPUTensor& tensor, float scale = 0.02f) {
            std::vector<float> data(tensor.size());
            for (auto& x : data) {
                x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
            }
            tensor.copy_from_host(data);
        };
        
        init_tensor(tok_embeddings);
        init_tensor(lm_head);
        
        // RMSNorm权重初始化为1
        std::vector<float> ones(config.dim, 1.0f);
        rms_final_weight.copy_from_host(ones);
    }
    
public:
    // 单步前向传播
    std::vector<float> forward_step(int token_id, int pos) {
        GPUTensor& hidden = act_buffers->get("hidden_states");
        GPUTensor& logits = act_buffers->get("logits");
        
        // 1. Token嵌入
        ops->embedding_lookup(tok_embeddings, hidden, token_id);
        
        // 2. 通过所有Transformer层
        GPUTensor* current = &hidden;
        GPUTensor& next = act_buffers->get("residual");  // 复用作为下一层输入
        
        for (int i = 0; i < config.n_layers; ++i) {
            layers[i].forward(*current, next, *kv_cache, i, pos, *act_buffers, *ops);
            std::swap(current, &next);  // 交换指针，避免拷贝
        }
        
        // 确保最终结果在hidden中
        if (current != &hidden) {
            CUDA_CHECK(cudaMemcpy(hidden.data(), current->data(), 
                                 hidden.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        
        // 3. 最终RMSNorm
        ops->rmsnorm(hidden, rms_final_weight);
        
        // 4. 语言模型头
        ops->gemv(hidden, lm_head, logits);
        
        // 5. 拷贝回CPU
        return logits.copy_to_host();
    }
    
    // 预填充
    std::vector<float> prefill(const std::vector<int>& token_ids) {
        std::vector<float> logits;
        for (size_t i = 0; i < token_ids.size(); ++i) {
            logits = forward_step(token_ids[i], i);
        }
        return logits;
    }
    
    // 贪心采样
    int sample_greedy(const std::vector<float>& logits) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        return std::distance(logits.begin(), max_it);
    }
    
    // 重置KV缓存
    void reset_kv_cache() {
        kv_cache->reset();
    }
    
    const LlamaConfig& get_config() const { return config; }
    
    // 权重加载接口
    void load_weights_from_file(const std::string& path) {
        // TODO: 实现权重加载
        std::cout << "权重加载功能待实现" << std::endl;
    }
};

// ============ 主函数 ============
int main() {
    // 初始化CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // 创建小模型用于测试
    LlamaConfig config(512, 4, 8, 4, 32000, 1536, 256);
    LlamaModel model(config);
    
    // 测试推理
    std::vector<int> prompt = {1, 42, 123, 7};
    model.reset_kv_cache();
    
    std::cout << "开始推理..." << std::endl;
    
    // 预填充
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> logits = model.prefill(prompt);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "预填充耗时: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;
    
    // 生成tokens
    int pos = prompt.size();
    int current_token = prompt.back();
    
    start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 10; ++step) {
        std::vector<float> next_logits = model.forward_step(current_token, pos);
        int next_token = model.sample_greedy(next_logits);
        
        std::cout << next_token << " ";
        
        current_token = next_token;
        pos++;
    }
    end = std::chrono::high_resolution_clock::now();
    
    std::cout << std::endl;
    std::cout << "生成10个token耗时: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;
    
    return 0;
}