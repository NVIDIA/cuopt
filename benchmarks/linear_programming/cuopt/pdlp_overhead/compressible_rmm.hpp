/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <rmm/error.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace lpfeas_compression {

using clock_t                            = std::chrono::steady_clock;
using resource_t                         = cuda::mr::any_resource<cuda::mr::device_accessible>;
constexpr std::size_t initial_pool_bytes = std::size_t{1} << 30;

inline void check_runtime(cudaError_t status, char const* call)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

inline void check_driver(CUresult status, char const* call)
{
  if (status == CUDA_SUCCESS) { return; }
  char const* name = nullptr;
  cuGetErrorName(status, &name);
  auto message = std::string(call) + ": " + (name ? name : "unknown CUDA driver error");
  if (status == CUDA_ERROR_OUT_OF_MEMORY) { throw rmm::out_of_memory(message); }
  throw std::runtime_error(message);
}

[[noreturn]] inline void fatal(char const* message) noexcept
{
  std::fprintf(stderr, "ALLOCATOR_FATAL %s\n", message);
  std::fflush(stderr);
  std::abort();
}

class context_scope_t {
 public:
  explicit context_scope_t(CUcontext wanted) : context_(wanted)
  {
    CUcontext current = nullptr;
    check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current != context_) {
      check_driver(cuCtxPushCurrent(context_), "cuCtxPushCurrent");
      pushed_ = true;
    }
  }
  ~context_scope_t() noexcept
  {
    if (pushed_) {
      CUcontext popped = nullptr;
      if (cuCtxPopCurrent(&popped) != CUDA_SUCCESS || popped != context_) {
        fatal("Cannot restore CUDA context");
      }
    }
  }
  context_scope_t(context_scope_t const&)            = delete;
  context_scope_t& operator=(context_scope_t const&) = delete;

 private:
  CUcontext context_{};
  bool pushed_{};
};

struct allocation_t {
  std::size_t requested{};
  std::size_t alignment{};
  std::size_t mapped_bytes{};
  CUmemGenericAllocationHandle handle{};
  bool reserved{};
  bool handle_live{};
  bool mapped{};
  bool compressed{};
};

inline void release_mapping(CUdeviceptr address, allocation_t& allocation) noexcept
{
  if (allocation.mapped) {
    if (cuMemUnmap(address, allocation.mapped_bytes) != CUDA_SUCCESS) {
      fatal("cuMemUnmap failed; backing memory retained");
    }
    allocation.mapped = false;
  }
  if (allocation.handle_live) {
    if (cuMemRelease(allocation.handle) != CUDA_SUCCESS) { fatal("cuMemRelease failed"); }
    allocation.handle_live = false;
  }
  if (allocation.reserved) {
    if (cuMemAddressFree(address, allocation.mapped_bytes) != CUDA_SUCCESS) {
      fatal("cuMemAddressFree failed");
    }
    allocation.reserved = false;
  }
}

inline void require_outside_capture(cuda::stream_ref stream)
{
  cudaStreamCaptureStatus status{};
  check_runtime(cudaStreamIsCapturing(stream.get(), &status), "cudaStreamIsCapturing");
  if (status != cudaStreamCaptureStatusNone) {
    throw std::runtime_error("VMM backing allocation/deallocation requested during graph capture");
  }
}

class vmm_state_t {
 public:
  explicit vmm_state_t(bool compress) : compression_requested(compress)
  {
    check_runtime(cudaFree(nullptr), "cudaFree context initialization");
    check_driver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    check_driver(cuCtxGetDevice(&device), "cuCtxGetDevice");
    if (context == nullptr) { throw std::runtime_error("No current CUDA context"); }
    check_driver(
      cuDeviceGetAttribute(
        &vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device),
      "query VMM capability");
    check_driver(
      cuDeviceGetAttribute(
        &compression_supported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device),
      "query compression capability");
    if (!vmm_supported || (compress && !compression_supported)) {
      throw std::runtime_error("Requested VMM/compression capability is unavailable; no fallback");
    }
    property.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id   = device;
    property.allocFlags.compressionType =
      compress ? CU_MEM_ALLOCATION_COMP_GENERIC : CU_MEM_ALLOCATION_COMP_NONE;
    check_driver(
      cuMemGetAllocationGranularity(&granularity, &property, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      "cuMemGetAllocationGranularity");
    if (granularity == 0 || (granularity & (granularity - 1)) != 0) {
      throw std::runtime_error("Invalid VMM allocation granularity");
    }
  }

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    if (bytes == 0) { return nullptr; }
    context_scope_t current(context);
    require_outside_capture(stream);
    if (bytes > std::numeric_limits<std::size_t>::max() - (granularity - 1)) {
      throw rmm::out_of_memory("VMM allocation size overflows granularity rounding");
    }
    allocation_t allocation;
    allocation.requested    = bytes;
    allocation.alignment    = alignment;
    allocation.mapped_bytes = ((bytes + granularity - 1) / granularity) * granularity;
    CUdeviceptr address{};
    try {
      check_driver(cuMemAddressReserve(
                     &address, allocation.mapped_bytes, std::max(alignment, granularity), 0, 0),
                   "cuMemAddressReserve");
      allocation.reserved = true;
      check_driver(cuMemCreate(&allocation.handle, allocation.mapped_bytes, &property, 0),
                   "cuMemCreate");
      allocation.handle_live = true;
      CUmemAllocationProp actual{};
      check_driver(cuMemGetAllocationPropertiesFromHandle(&actual, allocation.handle),
                   "cuMemGetAllocationPropertiesFromHandle");
      allocation.compressed = actual.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC;
      if (allocation.compressed != compression_requested) {
        throw std::runtime_error(
          "Actual compression property differs from requested mode; no fallback");
      }
      check_driver(cuMemMap(address, allocation.mapped_bytes, 0, allocation.handle, 0), "cuMemMap");
      allocation.mapped = true;
      CUmemAccessDesc access{};
      access.location = property.location;
      access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      check_driver(cuMemSetAccess(address, allocation.mapped_bytes, &access, 1), "cuMemSetAccess");
      auto* pointer = reinterpret_cast<void*>(address);
      std::lock_guard<std::mutex> lock(mutex);
      if (!allocations.emplace(pointer, allocation).second) {
        throw std::runtime_error("Duplicate VMM allocation address");
      }
      ++allocation_calls;
      cumulative_mapped_bytes += allocation.mapped_bytes;
      live_mapped_bytes += allocation.mapped_bytes;
      peak_mapped_bytes = std::max(peak_mapped_bytes, live_mapped_bytes);
      if (allocation.compressed) {
        ++compressed_allocation_calls;
        cumulative_compressed_bytes += allocation.mapped_bytes;
      }
      return pointer;
    } catch (...) {
      release_mapping(address, allocation);
      throw;
    }
  }

  void deallocate(cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    if (pointer == nullptr) { return; }
    try {
      context_scope_t current(context);
      require_outside_capture(stream);
      // RMM's pool orders users before this upstream free. VMM itself is synchronous.
      check_runtime(cudaStreamSynchronize(stream.get()),
                    "VMM upstream free stream synchronization");
      std::lock_guard<std::mutex> lock(mutex);
      auto found = allocations.find(pointer);
      if (found == allocations.end() || found->second.requested != bytes ||
          found->second.alignment != alignment) {
        fatal("VMM deallocation pointer/size mismatch");
      }
      auto const size = found->second.mapped_bytes;
      release_mapping(reinterpret_cast<CUdeviceptr>(pointer), found->second);
      live_mapped_bytes -= size;
      ++deallocation_calls;
      allocations.erase(found);
    } catch (std::exception const& error) {
      fatal(error.what());
    }
  }

  std::mutex mutex;
  CUcontext context{};
  CUdevice device{};
  CUmemAllocationProp property{};
  int vmm_supported{};
  int compression_supported{};
  bool compression_requested{};
  std::size_t granularity{};
  std::unordered_map<void*, allocation_t> allocations;
  std::size_t allocation_calls{};
  std::size_t deallocation_calls{};
  std::size_t compressed_allocation_calls{};
  std::size_t cumulative_mapped_bytes{};
  std::size_t cumulative_compressed_bytes{};
  std::size_t live_mapped_bytes{};
  std::size_t peak_mapped_bytes{};
};

inline rmm::mr::callback_memory_resource make_upstream(std::shared_ptr<vmm_state_t> state)
{
  return {
    [state](cuda::stream_ref stream, std::size_t bytes, std::size_t alignment, void*) {
      return state->allocate(stream, bytes, alignment);
    },
    [state](
      cuda::stream_ref stream, void* pointer, std::size_t bytes, std::size_t alignment, void*) {
      state->deallocate(stream, pointer, bytes, alignment);
    }};
}

struct pool_state_t {
  explicit pool_state_t(std::shared_ptr<vmm_state_t> storage)
    : backing(std::move(storage)), pool(resource_t{make_upstream(backing)}, initial_pool_bytes)
  {
  }
  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    // This RMM pool guarantees 256-byte suballocation alignment only.
    auto* pointer = alignment > rmm::CUDA_ALLOCATION_ALIGNMENT
                      ? backing->allocate(stream, bytes, alignment)
                      : pool.allocate(stream, bytes, alignment);
    if (bytes) {
      std::lock_guard<std::mutex> lock(mutex);
      ++allocation_calls;
      cumulative_bytes += bytes;
      live_bytes += bytes;
      ++live_allocations;
      peak_bytes = std::max(peak_bytes, live_bytes);
    }
    return pointer;
  }
  void deallocate(cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    if (alignment > rmm::CUDA_ALLOCATION_ALIGNMENT) {
      backing->deallocate(stream, pointer, bytes, alignment);
    } else {
      pool.deallocate(stream, pointer, bytes, alignment);
    }
    if (bytes) {
      std::lock_guard<std::mutex> lock(mutex);
      if (live_allocations == 0 || live_bytes < bytes) { fatal("RMM client accounting underflow"); }
      ++deallocation_calls;
      live_bytes -= bytes;
      --live_allocations;
    }
  }
  std::shared_ptr<vmm_state_t> backing;
  rmm::mr::pool_memory_resource pool;
  std::mutex mutex;
  std::size_t allocation_calls{};
  std::size_t deallocation_calls{};
  std::size_t cumulative_bytes{};
  std::size_t live_bytes{};
  std::size_t live_allocations{};
  std::size_t peak_bytes{};
};

inline rmm::mr::callback_memory_resource make_front(std::shared_ptr<pool_state_t> state)
{
  return {
    [state](cuda::stream_ref stream, std::size_t bytes, std::size_t alignment, void*) {
      return state->allocate(stream, bytes, alignment);
    },
    [state](
      cuda::stream_ref stream, void* pointer, std::size_t bytes, std::size_t alignment, void*) {
      state->deallocate(stream, pointer, bytes, alignment);
    }};
}

class allocator_scope_t {
 public:
  allocator_scope_t()
  {
    auto const began      = clock_t::now();
    auto const* requested = std::getenv("LPFEAS_RMM_MEMORY_MODE");
    mode_                 = requested && *requested ? requested : "default";
    if (mode_ != "default" && mode_ != "vmm" && mode_ != "compressed") {
      throw std::runtime_error("LPFEAS_RMM_MEMORY_MODE must be default, vmm or compressed");
    }
    if (mode_ != "default") {
      backing_ = std::make_shared<vmm_state_t>(mode_ == "compressed");
      pooled_  = std::make_shared<pool_state_t>(backing_);
      previous_.emplace(rmm::mr::set_current_device_resource(resource_t{make_front(pooled_)}));
    }
    initialization_seconds_ = std::chrono::duration<double>(clock_t::now() - began).count();
    report("installed");
  }
  allocator_scope_t(allocator_scope_t const&)            = delete;
  allocator_scope_t& operator=(allocator_scope_t const&) = delete;

  ~allocator_scope_t() noexcept
  {
    if (!backing_) { return; }
    auto const began = clock_t::now();
    try {
      context_scope_t current(backing_->context);
      check_runtime(cudaDeviceSynchronize(), "allocator teardown device synchronization");
      report("before_release");
      {
        std::lock_guard<std::mutex> lock(pooled_->mutex);
        if (pooled_->live_allocations || pooled_->live_bytes) {
          fatal("cuOpt allocations outlive allocator scope");
        }
      }
      {
        auto removed = rmm::mr::set_per_device_resource(rmm::cuda_device_id{backing_->device},
                                                        std::move(*previous_));
        previous_.reset();
      }
      pooled_.reset();
      if (!backing_->allocations.empty() || backing_->live_mapped_bytes != 0) {
        fatal("VMM backing allocations remain after pool destruction");
      }
      teardown_seconds_ = std::chrono::duration<double>(clock_t::now() - began).count();
      report("released");
    } catch (std::exception const& error) {
      fatal(error.what());
    }
  }

  void report(char const* phase) const noexcept
  {
    std::unique_lock<std::mutex> backing_lock;
    std::unique_lock<std::mutex> pool_lock;
    if (backing_) { backing_lock = std::unique_lock<std::mutex>(backing_->mutex); }
    if (pooled_) { pool_lock = std::unique_lock<std::mutex>(pooled_->mutex); }
    std::printf(
      "\nALLOCATOR_JSON {\"phase\":\"%s\",\"mode\":\"%s\","
      "\"initial_pool_bytes\":%zu,\"device\":%d,\"vmm_supported\":%d,"
      "\"compression_supported\":%d,\"granularity\":%zu,"
      "\"vmm_allocation_calls\":%zu,\"vmm_deallocation_calls\":%zu,"
      "\"confirmed_compressed_allocations\":%zu,\"mapped_bytes_total\":%zu,"
      "\"confirmed_compressed_bytes_total\":%zu,\"mapped_bytes_live\":%zu,"
      "\"mapped_bytes_peak\":%zu,\"client_allocation_calls\":%zu,"
      "\"client_deallocation_calls\":%zu,\"client_bytes_total\":%zu,"
      "\"client_bytes_live\":%zu,\"client_bytes_peak\":%zu,"
      "\"fallback_allocations\":0,\"initialization_seconds\":%.9f,"
      "\"teardown_seconds\":%.9f}\n",
      phase,
      mode_.c_str(),
      backing_ ? initial_pool_bytes : 0,
      backing_ ? backing_->device : -1,
      backing_ ? backing_->vmm_supported : -1,
      backing_ ? backing_->compression_supported : -1,
      backing_ ? backing_->granularity : 0,
      backing_ ? backing_->allocation_calls : 0,
      backing_ ? backing_->deallocation_calls : 0,
      backing_ ? backing_->compressed_allocation_calls : 0,
      backing_ ? backing_->cumulative_mapped_bytes : 0,
      backing_ ? backing_->cumulative_compressed_bytes : 0,
      backing_ ? backing_->live_mapped_bytes : 0,
      backing_ ? backing_->peak_mapped_bytes : 0,
      pooled_ ? pooled_->allocation_calls : 0,
      pooled_ ? pooled_->deallocation_calls : 0,
      pooled_ ? pooled_->cumulative_bytes : 0,
      pooled_ ? pooled_->live_bytes : 0,
      pooled_ ? pooled_->peak_bytes : 0,
      initialization_seconds_,
      teardown_seconds_);
    std::fflush(stdout);
  }

 private:
  std::string mode_;
  std::shared_ptr<vmm_state_t> backing_;
  std::shared_ptr<pool_state_t> pooled_;
  std::optional<resource_t> previous_;
  double initialization_seconds_{};
  double teardown_seconds_{};
};

}  // namespace lpfeas_compression
