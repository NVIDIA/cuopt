/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "pdlp_memory_resource.hpp"

#include <utilities/logger.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace cuopt::mathematical_optimization::pdlp {
namespace {

struct driver_error_t : rmm::cuda_error {
  driver_error_t(CUresult result, char const* call)
    : rmm::cuda_error(std::string(call) + " failed with CUDA driver error " +
                      std::to_string(static_cast<int>(result))),
      result(result)
  {
  }
  CUresult result;
};

void check_driver(CUresult result, char const* call)
{
  if (result != CUDA_SUCCESS) { throw driver_error_t(result, call); }
}

void check_cleanup(CUresult result) noexcept
{
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    std::fputs("cuOpt: CUDA failure while releasing PDLP memory\n", stderr);
    std::terminate();
  }
}

class context_scope_t {
 public:
  explicit context_scope_t(CUcontext context) : context_(context)
  {
    CUcontext current{};
    check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current != context_) {
      check_driver(cuCtxPushCurrent(context_), "cuCtxPushCurrent");
      pushed_ = true;
    }
  }
  ~context_scope_t()
  {
    if (pushed_) {
      CUcontext popped{};
      check_cleanup(cuCtxPopCurrent(&popped));
    }
  }
  context_scope_t(context_scope_t const&)            = delete;
  context_scope_t& operator=(context_scope_t const&) = delete;

 private:
  CUcontext context_{};
  bool pushed_{};
};

struct allocation_t {
  std::size_t bytes{};
  std::size_t alignment{};
  std::size_t mapped_bytes{};
  CUmemGenericAllocationHandle handle{};
  bool reserved{};
  bool handle_live{};
  bool mapped{};
};

void release_mapping(CUdeviceptr address, allocation_t const& allocation) noexcept
{
  if (allocation.mapped) { check_cleanup(cuMemUnmap(address, allocation.mapped_bytes)); }
  if (allocation.handle_live) { check_cleanup(cuMemRelease(allocation.handle)); }
  if (allocation.reserved) { check_cleanup(cuMemAddressFree(address, allocation.mapped_bytes)); }
}

void require_outside_capture(cuda::stream_ref stream)
{
  cudaStreamCaptureStatus status{};
  RMM_CUDA_TRY(cudaStreamIsCapturing(stream.get(), &status));
  RMM_EXPECTS(status == cudaStreamCaptureStatusNone,
              "PDLP compressed memory backing cannot grow during CUDA graph capture");
}

class compressed_memory_t {
 public:
  compressed_memory_t(CUcontext context,
                      CUmemAllocationProp property,
                      std::size_t granularity,
                      device_memory_resource_t upstream)
    : context(context),
      property_(property),
      granularity_(granularity),
      upstream_(std::move(upstream))
  {
  }

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    if (bytes == 0) { return nullptr; }
    context_scope_t current(context);
    require_outside_capture(stream);
    RMM_EXPECTS(bytes <= std::numeric_limits<std::size_t>::max() - (granularity_ - 1),
                "PDLP allocation size overflows VMM rounding",
                rmm::out_of_memory);
    allocation_t allocation{
      bytes, alignment, ((bytes + granularity_ - 1) / granularity_) * granularity_};
    CUdeviceptr address{};
    try {
      check_driver(cuMemAddressReserve(
                     &address, allocation.mapped_bytes, std::max(alignment, granularity_), 0, 0),
                   "cuMemAddressReserve");
      allocation.reserved = true;
      check_driver(cuMemCreate(&allocation.handle, allocation.mapped_bytes, &property_, 0),
                   "cuMemCreate");
      allocation.handle_live = true;
      check_driver(cuMemMap(address, allocation.mapped_bytes, 0, allocation.handle, 0), "cuMemMap");
      allocation.mapped = true;
      CUmemAccessDesc access{};
      access.location = property_.location;
      access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      check_driver(cuMemSetAccess(address, allocation.mapped_bytes, &access, 1), "cuMemSetAccess");
      CUmemAllocationProp actual{};
      check_driver(cuMemGetAllocationPropertiesFromHandle(&actual, allocation.handle),
                   "cuMemGetAllocationPropertiesFromHandle");
      if (actual.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC &&
          !reported_compression_.exchange(true)) {
        CUOPT_LOG_INFO("PDLP memory pool: hardware compression confirmed for backing allocation");
      }
      auto* pointer = reinterpret_cast<void*>(address);
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.emplace(pointer, allocation);
      return pointer;
    } catch (driver_error_t const& error) {
      release_mapping(address, allocation);
      // Compression availability can change after the capability query. Keep solving with the
      // original allocation policy when the driver cannot supply this optional allocation type.
      if (error.result != CUDA_ERROR_NOT_SUPPORTED && error.result != CUDA_ERROR_NOT_PERMITTED &&
          error.result != CUDA_ERROR_OUT_OF_MEMORY) {
        throw;
      }
      return allocate_fallback(stream, bytes, alignment);
    } catch (...) {
      release_mapping(address, allocation);
      throw;
    }
  }

  void deallocate(cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment,
                  bool wait_for_stream = true) noexcept
  {
    if (pointer == nullptr) { return; }
    try {
      context_scope_t current(context);
      allocation_t allocation;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto const found = allocations_.find(pointer);
        if (found == allocations_.end() || found->second.bytes != bytes ||
            found->second.alignment != alignment) {
          std::terminate();
        }
        allocation = found->second;
        allocations_.erase(found);
      }
      if (!allocation.mapped) {
        if (wait_for_stream) {
          upstream_.deallocate(stream, pointer, bytes, alignment);
        } else {
          upstream_.deallocate_sync(pointer, bytes, alignment);
        }
        return;
      }
      // Direct over-aligned frees wait for their stream. Pool teardown has already waited for
      // the completion events covering every client deallocation.
      if (wait_for_stream) { RMM_CUDA_TRY(cudaStreamSynchronize(stream.get())); }
      release_mapping(reinterpret_cast<CUdeviceptr>(pointer), allocation);
    } catch (...) {
      std::terminate();
    }
  }

  CUcontext const context;

 private:
  void* allocate_fallback(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    void* pointer = upstream_.allocate(stream, bytes, alignment);
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.emplace(pointer, allocation_t{bytes, alignment});
    } catch (...) {
      upstream_.deallocate(stream, pointer, bytes, alignment);
      throw;
    }
    return pointer;
  }

  CUmemAllocationProp property_{};
  std::size_t granularity_{};
  device_memory_resource_t upstream_;
  std::mutex mutex_;
  std::unordered_map<void*, allocation_t> allocations_;
  std::atomic<bool> reported_compression_{false};
};

class compressed_backing_resource_t {
 public:
  explicit compressed_backing_resource_t(std::shared_ptr<compressed_memory_t> backing)
    : backing_(std::move(backing))
  {
  }
  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    return backing_->allocate(stream, bytes, alignment);
  }
  void deallocate(cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    backing_->deallocate(stream, pointer, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    auto stream   = cuda::stream_ref{cudaStreamLegacy};
    auto* pointer = allocate(stream, bytes, alignment);
    stream.sync();
    return pointer;
  }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    // Only the owning pool calls this, after it drains all client completion events.
    backing_->deallocate(cuda::stream_ref{cudaStreamLegacy}, pointer, bytes, alignment, false);
  }
  bool operator==(compressed_backing_resource_t const& other) const noexcept
  {
    return backing_ == other.backing_;
  }
  friend void get_property(compressed_backing_resource_t const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<compressed_memory_t> backing_;
};

class compressed_pool_t {
 public:
  explicit compressed_pool_t(std::shared_ptr<compressed_memory_t> backing)
    : backing_(std::move(backing)), pool_(std::in_place, compressed_backing_resource_t{backing_}, 0)
  {
  }
  ~compressed_pool_t()
  {
    if (rmm::process_is_exiting()) { return; }
    try {
      context_scope_t current(backing_->context);
      // Events remain valid after the caller destroys a stream. Wait only for work queued
      // before each client's free; unrelated streams and later work do not delay this pool.
      for (auto const& [id, event] : release_events_) {
        RMM_CUDA_TRY(cudaEventSynchronize(event));
      }
      pool_.reset();
      for (auto const& [id, event] : release_events_) {
        RMM_CUDA_TRY(cudaEventDestroy(event));
      }
    } catch (...) {
      std::terminate();
    }
  }
  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    context_scope_t current(backing_->context);
    return alignment > rmm::CUDA_ALLOCATION_ALIGNMENT ? backing_->allocate(stream, bytes, alignment)
                                                      : pool_->allocate(stream, bytes, alignment);
  }
  void deallocate(cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    try {
      context_scope_t current(backing_->context);
      if (alignment > rmm::CUDA_ALLOCATION_ALIGNMENT) {
        backing_->deallocate(stream, pointer, bytes, alignment);
      } else {
        pool_->deallocate(stream, pointer, bytes, alignment);
        if (pointer != nullptr) { record_release(stream); }
      }
    } catch (...) {
      std::terminate();
    }
  }

 private:
  void record_release(cuda::stream_ref stream)
  {
    unsigned long long id{};
    RMM_CUDA_TRY(cudaStreamGetId(stream.get(), &id));
    std::lock_guard<std::mutex> lock(events_mutex_);
    auto [entry, inserted] = release_events_.try_emplace(id, nullptr);
    if (inserted) {
      RMM_CUDA_TRY(cudaEventCreateWithFlags(&entry->second, cudaEventDisableTiming));
    }
    RMM_CUDA_TRY(cudaEventRecord(entry->second, stream.get()));
  }

  std::shared_ptr<compressed_memory_t> backing_;
  std::optional<rmm::mr::pool_memory_resource> pool_;
  std::mutex events_mutex_;
  // CUDA stream IDs distinguish per-thread defaults and are never reused during the process.
  std::unordered_map<unsigned long long, cudaEvent_t> release_events_;
};

class compressed_pool_resource_t {
 public:
  explicit compressed_pool_resource_t(std::shared_ptr<compressed_pool_t> pool)
    : pool_(std::move(pool))
  {
  }
  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    return pool_->allocate(stream, bytes, alignment);
  }
  void deallocate(cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    pool_->deallocate(stream, pointer, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    auto stream   = cuda::stream_ref{cudaStreamLegacy};
    auto* pointer = allocate(stream, bytes, alignment);
    stream.sync();
    return pointer;
  }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    deallocate(stream, pointer, bytes, alignment);
    try {
      stream.sync();
    } catch (...) {
      std::terminate();
    }
  }
  bool operator==(compressed_pool_resource_t const& other) const noexcept
  {
    return pool_ == other.pool_;
  }
  friend void get_property(compressed_pool_resource_t const&, cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<compressed_pool_t> pool_;
};

}  // namespace

device_memory_resource_t make_pdlp_memory_resource(device_memory_resource_t upstream)
{
  if (!cuda::mr::resource_cast<rmm::mr::cuda_memory_resource>(&upstream) &&
      !cuda::mr::resource_cast<rmm::mr::cuda_async_memory_resource>(&upstream)) {
    return upstream;
  }
  RMM_CUDA_TRY(cudaFree(nullptr));
  CUcontext context{};
  CUdevice device{};
  check_driver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  check_driver(cuCtxGetDevice(&device), "cuCtxGetDevice");
  int vmm{}, compression{};
  check_driver(
    cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device),
    "query VMM capability");
  check_driver(
    cuDeviceGetAttribute(&compression, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device),
    "query compression capability");
  if (!vmm || !compression) { return upstream; }
  CUmemAllocationProp property{};
  property.type                       = CU_MEM_ALLOCATION_TYPE_PINNED;
  property.location.type              = CU_MEM_LOCATION_TYPE_DEVICE;
  property.location.id                = device;
  property.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
  std::size_t granularity{};
  auto const status =
    cuMemGetAllocationGranularity(&granularity, &property, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (status == CUDA_ERROR_NOT_SUPPORTED || status == CUDA_ERROR_NOT_PERMITTED) { return upstream; }
  check_driver(status, "cuMemGetAllocationGranularity");
  RMM_EXPECTS(std::has_single_bit(granularity), "Invalid PDLP VMM allocation granularity");
  auto backing =
    std::make_shared<compressed_memory_t>(context, property, granularity, std::move(upstream));
  auto pool = std::make_shared<compressed_pool_t>(std::move(backing));
  return device_memory_resource_t{compressed_pool_resource_t{std::move(pool)}};
}

bool is_pdlp_memory_resource(device_memory_resource_t const& resource) noexcept
{
  return cuda::mr::resource_cast<compressed_pool_resource_t>(&resource) != nullptr;
}

}  // namespace cuopt::mathematical_optimization::pdlp
