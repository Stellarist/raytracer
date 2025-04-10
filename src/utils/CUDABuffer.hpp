#pragma once

#include <cassert>
#include <vector>
#include <cuda.h>

#include "debug.hpp"

template <typename T>
class CUDABuffer {
private:
	T*     ptr{nullptr};
	size_t count{0};
    size_t alloc_count{0};

public:
	CUDABuffer(size_t count = 0)
	{
		alloc(count);
	}

	CUDABuffer(size_t count, const T* data)
	{
		alloc(count);
		upload(data);
	}

    CUDABuffer(std::initializer_list<T> data)
	{
		alloc(data.size());
		upload(data.begin());
	}

	CUDABuffer(const std::vector<T>& data)
	{
		alloc(data.size());
		upload(data.data());
	}

	CUDABuffer(const CUDABuffer&) = delete;
	CUDABuffer& operator=(const CUDABuffer&) = delete;

	CUDABuffer(CUDABuffer&& other)
	{
		swap(other);
	}

	CUDABuffer& operator=(CUDABuffer&& other)
	{
		swap(other);
		return *this;
	}

	~CUDABuffer()
	{
		free();
	}

	T* get() {return ptr;}
	const T* get() const {return ptr;}
	T* get(size_t index) {return ptr + index;}
	const T* get(size_t index) const {return ptr + index;}
	size_t size() const {return count;}
	size_t byteSize() const {return count * sizeof(T);}
	size_t capacity() const {return alloc_count;}
	size_t byteCapacity() const {return alloc_count * sizeof(T);}
	bool empty() const {return count == 0;}
	static unsigned int stride() {return sizeof(T);}

	CUdeviceptr getDevicePtr() const {return reinterpret_cast<CUdeviceptr>(ptr);}
	CUdeviceptr getDevicePtr(size_t index) const {return reinterpret_cast<CUdeviceptr>(ptr+index);}

	void alloc(size_t cnt)
	{
		if (alloc_count == cnt) {
			count = cnt;	
			return;
		}

		free();
		alloc_count = count = cnt;
		if (alloc_count)
			CUDA_CHECK(cudaMalloc((void**)&ptr, alloc_count * sizeof(T)));
	}

	void free()
	{
		count = 0;
		alloc_count = 0;
		if (ptr)
			CUDA_CHECK(cudaFree(ptr));
		ptr = nullptr;
	}

	void upload(const T* data)
	{
		CUDA_CHECK(cudaMemcpy(ptr, data, count*sizeof(T), cudaMemcpyHostToDevice));
	}

	void upload(std::initializer_list<T> data)
	{
		assert(data.size() <= count);
		CUDA_CHECK(cudaMemcpy(ptr, data.begin(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
	}

    void download(T* data) const
	{
		CUDA_CHECK(cudaMemcpy(data, ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	void download(std::vector<T>& vec) const
	{
		vec.resize(count);
		CUDA_CHECK(cudaMemcpy(vec.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	std::vector<T> download() const
	{
		std::vector<T> result(count);
		download(result.data());
		return result;
	}

	void resize(size_t cnt)
	{
		if (alloc_count < cnt) {
			CUDABuffer<T> temp_buffer(cnt);
			CUDA_CHECK(cudaMemcpy(temp_buffer.ptr, ptr, count * sizeof(T), cudaMemcpyDeviceToDevice));
			this->swap(temp_buffer);
		} else
			count = cnt;
	}

	void copy(const T* data)
	{
		CUDA_CHECK(cudaMemcpy(ptr, data, count * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename U>
	void copy(const CUDABuffer<U>& other)
	{
		assert(other.byteSize() == byteSize());
		CUDA_CHECK(cudaMemcpy(ptr, other.get(), count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	void swap(CUDABuffer<T>& other)
	{
		std::swap(ptr, other.ptr);
		std::swap(count, other.count);
		std::swap(alloc_count, other.alloc_count);
	}
};
