#include "tensor.h"

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT16: return 2;
        case DType::BFLOAT16: return 2;
        case DType::INT32: return 4;
        case DType::UINT8: return 1;
        default: throw std::runtime_error("Unsupported dtype");
    }
}

CUDAMemoryPool& CUDAMemoryPool::instance() {
    static CUDAMemoryPool pool;
    return pool;
}

void* CUDAMemoryPool::malloc(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& bucket = pool_[bytes];
    if (!bucket.empty()) {
        void* ptr = bucket.back();
        bucket.pop_back();
        return ptr;
    }
    void* ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void CUDAMemoryPool::free(void* ptr, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_[bytes].push_back(ptr);
}

CUDAMemoryPool::~CUDAMemoryPool() {
    for (auto& kv : pool_) {
        for (auto ptr : kv.second) {
            cudaFree(ptr);
        }
    }
}

Buffer::Buffer(size_t bytes, Device device) : bytes_(bytes), device_(device) {
    if (device_ == Device::CPU) {
        data_ = malloc(bytes_);
    } else {
        data_ = CUDAMemoryPool::instance().malloc(bytes_);
    }
}

Buffer::~Buffer() {
    if (data_) {
        if (device_ == Device::CPU) {
            free(data_);
        } else {
            CUDAMemoryPool::instance().free(data_, bytes_);
        }
    }
}

void* Buffer::ptr() { return data_; }
size_t Buffer::bytes() const { return bytes_; }
Device Buffer::device() const { return device_; }

void Tensor::compute_default_stride() {
    stride_.resize(shape_.size());
    if (shape_.empty()) return;
    stride_[shape_.size() - 1] = 1;
    for (int i = (int)shape_.size() - 2; i >= 0; --i) {
        stride_[i] = stride_[i + 1] * shape_[i + 1];
    }
}

Tensor::Tensor(std::vector<int> shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    compute_default_stride();
    buffer_ = std::make_shared<Buffer>(bytes(), device_);
}

void* Tensor::data() { 
    return static_cast<char*>(buffer_->ptr()) + offset_; 
}

const std::vector<int>& Tensor::shape() const { return shape_; }
const std::vector<int>& Tensor::stride() const { return stride_; }
DType Tensor::dtype() const { return dtype_; }
Device Tensor::device() const { return device_; }

size_t Tensor::numel() const {
    size_t total = 1;
    for (auto s : shape_) total *= s;
    return total;
}

size_t Tensor::bytes() const {
    return numel() * dtype_size(dtype_);
}

void Tensor::to(Device new_device, cudaStream_t stream) {
    if (new_device == device_) return;

    auto new_buffer = std::make_shared<Buffer>(bytes(), new_device);
    if (new_device == Device::GPU) {
        cudaMemcpyAsync(new_buffer->ptr(), data(), bytes(),
                        cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpyAsync(new_buffer->ptr(), data(), bytes(),
                        cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    buffer_ = new_buffer;
    device_ = new_device;
    offset_ = 0;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != numel()) {
        throw std::runtime_error("reshape: 元素数量不匹配");
    }
    Tensor t = *this;
    t.shape_ = new_shape;
    t.compute_default_stride();
    return t;
}

Tensor Tensor::view(const std::vector<int>& new_shape) const {
    int infer_dim = -1;
    size_t known_numel = 1;

    for (int i = 0; i < (int)new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::runtime_error("view: 只能有一个 -1 维度");
            }
            infer_dim = i;
        } else if (new_shape[i] > 0) {
            known_numel *= new_shape[i];
        } else {
            throw std::runtime_error("view: 维度必须为正数或 -1");
        }
    }

    size_t total = numel();
    std::vector<int> final_shape = new_shape;

    if (infer_dim != -1) {
        if (total % known_numel != 0) {
            throw std::runtime_error("view: 元素数量不匹配，无法推导 -1 维度");
        }
        final_shape[infer_dim] = total / known_numel;
    }

    size_t check_numel = 1;
    for (auto s : final_shape) check_numel *= s;
    if (check_numel != total) {
        throw std::runtime_error("view: 元素数量不匹配");
    }

    Tensor t = *this;
    t.shape_ = final_shape;
    t.compute_default_stride();
    return t;
}

Tensor Tensor::slice(int dim, int start, int end) const {
    if (dim < 0 || dim >= (int)shape_.size()) {
        throw std::runtime_error("slice: 维度越界");
    }
    if (start < 0 || end > shape_[dim] || start >= end) {
        throw std::runtime_error("slice: 起止位置非法");
    }

    size_t elem_size = dtype_size(dtype_);
    size_t byte_offset = start * stride_[dim] * elem_size;

    Tensor t = *this;
    t.shape_[dim] = end - start;
    t.offset_ += byte_offset;
    return t;
}

Tensor Tensor::contiguous(cudaStream_t stream) const {
    if (is_contiguous() && offset_ == 0) {
        return *this;
    }

    Tensor t(shape_, dtype_, device_);

    if (device_ == Device::CPU) {
        std::memcpy(t.data(), data(), bytes());
    } else {
        cudaMemcpyAsync(t.data(), data(), bytes(),
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return t;
}

bool Tensor::is_contiguous() const {
    std::vector<int> expected_stride;
    expected_stride.resize(shape_.size());
    if (shape_.empty()) return true;
    expected_stride[shape_.size() - 1] = 1;
    for (int i = (int)shape_.size() - 2; i >= 0; --i) {
        expected_stride[i] = expected_stride[i + 1] * shape_[i + 1];
    }
    return expected_stride == stride_;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0 || dim0 >= (int)shape_.size() ||
        dim1 < 0 || dim1 >= (int)shape_.size()) {
        throw std::runtime_error("transpose: 维度越界");
    }
    Tensor t = *this;
    std::swap(t.shape_[dim0], t.shape_[dim1]);
    std::swap(t.stride_[dim0], t.stride_[dim1]);
    return t;
}