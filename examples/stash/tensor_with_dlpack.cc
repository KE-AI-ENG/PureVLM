#include "dlpack/dlpack.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

// ===== CUDA Caching Allocator =====
class CudaCachingAllocator {
public:
    void* allocate(size_t size) {
        auto& bucket = pool[size];
        if (!bucket.empty()) {
            void* ptr = bucket.back();
            bucket.pop_back();
            reserved_bytes_ -= size;
            allocated_bytes_ += size;
            return ptr;
        }
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        allocated_bytes_ += size;
        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        pool[size].push_back(ptr);
        allocated_bytes_ -= size;
        reserved_bytes_ += size;
    }

    void empty_cache() {
        for (auto& kv : pool) {
            for (auto ptr : kv.second) {
                cudaFree(ptr);
            }
        }
        pool.clear();
        reserved_bytes_ = 0;
    }

    ~CudaCachingAllocator() {
        empty_cache();
    }

    size_t memory_allocated() const { return allocated_bytes_; }
    size_t memory_reserved() const { return reserved_bytes_; }

private:
    std::unordered_map<size_t, std::vector<void*>> pool;
    size_t allocated_bytes_ = 0; // 当前使用中的显存
    size_t reserved_bytes_ = 0;  // 缓存池保留的显存
};

static CudaCachingAllocator g_allocator;

// ===== RAII 封装的 TensorHandle =====
class TensorHandle {
public:
    TensorHandle() : internal_(nullptr) {}

    TensorHandle(const std::vector<int64_t>& shape,
                 const std::string& device = "cuda",
                 DLDataType dtype = {kDLFloat, 32, 1},
                 int device_id = 0) {
        internal_ = create_dlpack_tensor(shape, device, dtype, device_id);
    }

    static TensorHandle empty(const std::vector<int64_t>& shape,
                              const std::string& device = "cuda",
                              DLDataType dtype = {kDLFloat, 32, 1},
                              int device_id = 0) {
        return TensorHandle(shape, device, dtype, device_id);
    }

    TensorHandle(const TensorHandle&) = delete;
    TensorHandle& operator=(const TensorHandle&) = delete;
    TensorHandle(TensorHandle&& other) noexcept {
        internal_ = other.internal_;
        other.internal_ = nullptr;
    }
    TensorHandle& operator=(TensorHandle&& other) noexcept {
        if (this != &other) {
            release();
            internal_ = other.internal_;
            other.internal_ = nullptr;
        }
        return *this;
    }

    ~TensorHandle() {
        release();
    }

    std::vector<int64_t> shape() const {
        DLManagedTensor* t = static_cast<DLManagedTensor*>(internal_);
        return std::vector<int64_t>(t->dl_tensor.shape,
                                    t->dl_tensor.shape + t->dl_tensor.ndim);
    }

    void* data() const {
        DLManagedTensor* t = static_cast<DLManagedTensor*>(internal_);
        return t->dl_tensor.data;
    }

    DLDataType dtype() const {
        DLManagedTensor* t = static_cast<DLManagedTensor*>(internal_);
        return t->dl_tensor.dtype;
    }

    DLDevice device() const {
        DLManagedTensor* t = static_cast<DLManagedTensor*>(internal_);
        return t->dl_tensor.device;
    }

    DLManagedTensor* to_dlpack() const {
        return static_cast<DLManagedTensor*>(internal_);
    }

    // ===== 显存统计接口 =====
    static size_t memory_allocated() {
        return g_allocator.memory_allocated();
    }
    static size_t memory_reserved() {
        return g_allocator.memory_reserved();
    }

    // ===== 新增：显存清理接口 =====
    static void empty_cache() {
        g_allocator.empty_cache();
    }

private:
    void* internal_;

    static DLManagedTensor* create_dlpack_tensor(const std::vector<int64_t>& shape,
                                                 const std::string& device,
                                                 DLDataType dtype,
                                                 int device_id) {
        DLManagedTensor* managed = new DLManagedTensor();

        size_t type_size = dtype.bits / 8;
        size_t total_size = type_size;
        for (auto dim : shape) {
            total_size *= dim;
        }

        managed->dl_tensor.data = g_allocator.allocate(total_size);
        managed->dl_tensor.device = {kDLCUDA, device_id};
        managed->dl_tensor.ndim = shape.size();

        int64_t* shape_copy = new int64_t[shape.size()];
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_copy[i] = shape[i];
        }
        managed->dl_tensor.shape = shape_copy;
        managed->dl_tensor.strides = nullptr;
        managed->dl_tensor.dtype = dtype;
        managed->dl_tensor.byte_offset = 0;

        managed->manager_ctx = nullptr;
        managed->deleter = [](DLManagedTensor* self) {
            size_t type_size = self->dl_tensor.dtype.bits / 8;
            size_t total_size = type_size;
            for (int i = 0; i < self->dl_tensor.ndim; ++i) {
                total_size *= self->dl_tensor.shape[i];
            }
            g_allocator.deallocate(self->dl_tensor.data, total_size);
            delete[] self->dl_tensor.shape;
            delete self;
        };

        return managed;
    }

    void release() {
        if (internal_) {
            DLManagedTensor* t = static_cast<DLManagedTensor*>(internal_);
            t->deleter(t);
            internal_ = nullptr;
        }
    }
};

// ===== 使用示例 =====
#ifdef TENSOR_MAIN
int main() {
    std::cout << "Allocated: " << TensorHandle::memory_allocated()
              << " bytes, Reserved: " << TensorHandle::memory_reserved() << " bytes\n";

    {
        TensorHandle t1 = TensorHandle::empty({2, 3}, "cuda");
        TensorHandle t2 = TensorHandle::empty({4, 5}, "cuda");
        std::cout << "Allocated: " << TensorHandle::memory_allocated()
                  << " bytes, Reserved: " << TensorHandle::memory_reserved() << " bytes\n";
    } // t1, t2 析构，显存进入缓存池

    std::cout << "Allocated: " << TensorHandle::memory_allocated()
              << " bytes, Reserved: " << TensorHandle::memory_reserved() << " bytes\n";

    // 主动清理缓存池显存
    TensorHandle::empty_cache();
    std::cout << "After empty_cache -> Allocated: " << TensorHandle::memory_allocated()
              << " bytes, Reserved: " << TensorHandle::memory_reserved() << " bytes\n";

    return 0;
}
#endif
