#pragma once
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

enum class DType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT32,
    UINT8
};

size_t dtype_size(DType dtype);

enum class Device {
    CPU,
    GPU
};

class CUDAMemoryPool {
public:
    static CUDAMemoryPool& instance();
    void* malloc(size_t bytes);
    void free(void* ptr, size_t bytes);
    ~CUDAMemoryPool();
private:
    std::unordered_map<size_t, std::vector<void*>> pool_;
    std::mutex mutex_;
};

class Buffer {
public:
    Buffer(size_t bytes, Device device);
    ~Buffer();

    void* ptr();
    size_t bytes() const;
    Device device() const;

private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
    Device device_;
};

class Tensor {
public:
    Tensor(std::vector<int> shape, DType dtype, Device device = Device::CPU);
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    void* data();
    const std::vector<int>& shape() const;
    const std::vector<int>& stride() const;
    DType dtype() const;
    Device device() const;

    size_t numel() const;
    size_t bytes() const;

    void to(Device new_device, cudaStream_t stream = 0);

    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor slice(int dim, int start, int end) const;
    Tensor contiguous(cudaStream_t stream = 0) const;

    bool is_contiguous() const;
    Tensor transpose(int dim0, int dim1) const;

private:
    std::vector<int> shape_;
    std::vector<int> stride_;
    DType dtype_;
    Device device_;
    std::shared_ptr<Buffer> buffer_;
    size_t offset_ = 0; // 字节偏移

    void compute_default_stride();
};