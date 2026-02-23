/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * @file memory_instrumentation.hpp
 * @brief Memory access instrumentation utilities
 *
 * This file provides wrapper classes for tracking memory reads and writes.
 *
 * Usage:
 *   - Define CUOPT_ENABLE_MEMORY_INSTRUMENTATION to enable tracking
 *   - When undefined, all instrumentation becomes zero-overhead passthrough
 *     (record_*() calls inline away, no counter storage overhead)
 *
 * Example:
 *   ins_vector<int> vec;  // Instrumented std::vector<int>
 *   vec.push_back(42);
 *   auto val = vec[0];
 *   // When enabled: tracking occurs, counters accumulate
 *   // When disabled: direct passthrough, compiler optimizes away all overhead
 */

#pragma once

#include <functional>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define CUOPT_ENABLE_MEMORY_INSTRUMENTATION 1

#ifdef __NVCC__
#define HDI inline __host__ __device__
#else
#define HDI inline
#endif

namespace cuopt {

// Define CUOPT_ENABLE_MEMORY_INSTRUMENTATION to 1 to enable memory tracking
// When 0, instrumentation becomes a zero-overhead passthrough

// Base class for memory operation instrumentation
struct memory_instrumentation_base_t {
#if CUOPT_ENABLE_MEMORY_INSTRUMENTATION
  HDI void reset_counters() const { byte_loads = byte_stores = 0; }

  template <typename T>
  HDI void record_load() const
  {
    byte_loads += sizeof(T);
  }

  template <typename T>
  HDI void record_store() const
  {
    byte_stores += sizeof(T);
  }

  template <typename T>
  HDI void record_rmw() const
  {
    byte_loads += sizeof(T);
    byte_stores += sizeof(T);
  }

  mutable size_t byte_loads{0};
  mutable size_t byte_stores{0};
#else
  // No-op methods when instrumentation is disabled - these inline away to zero overhead
  HDI void reset_counters() const {}
  template <typename T>
  HDI void record_load() const
  {
  }
  template <typename T>
  HDI void record_store() const
  {
  }
  template <typename T>
  HDI void record_rmw() const
  {
  }
#endif  // CUOPT_ENABLE_MEMORY_INSTRUMENTATION
};

#if CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// aggregator class to collect statistics from multiple instrumented objects
class instrumentation_aggregator_t {
 public:
  instrumentation_aggregator_t() = default;

  // Construct with initializer list of (description, instrumented object) pairs
  instrumentation_aggregator_t(
    std::initializer_list<
      std::pair<std::string, std::reference_wrapper<const memory_instrumentation_base_t>>>
      instrumented)
  {
    for (const auto& [name, instr] : instrumented) {
      instrumented_.insert_or_assign(name, instr);
    }
  }

  // Add an instrumented object to track with a description
  void add(const std::string& description, const memory_instrumentation_base_t& instrumented)
  {
    instrumented_.insert_or_assign(description, std::cref(instrumented));
  }

  // Collect total loads and stores across all instrumented objects
  std::pair<size_t, size_t> collect()
  {
    size_t total_loads  = 0;
    size_t total_stores = 0;

    for (auto& [name, instr] : instrumented_) {
      total_loads += instr.get().byte_loads;
      total_stores += instr.get().byte_stores;
    }

    return {total_loads, total_stores};
  }

  // Collect per-wrapper statistics
  std::vector<std::tuple<std::string, size_t, size_t>> collect_per_wrapper()
  {
    std::vector<std::tuple<std::string, size_t, size_t>> results;
    results.reserve(instrumented_.size());

    for (auto& [name, instr] : instrumented_) {
      results.emplace_back(name, instr.get().byte_loads, instr.get().byte_stores);
    }

    return results;
  }

  // Collect total loads and stores, then flush counters
  std::pair<size_t, size_t> collect_and_flush()
  {
    auto result = collect();
    flush();
    return result;
  }

  void flush()
  {
    for (auto& [name, instr] : instrumented_) {
      instr.get().reset_counters();
    }
  }

 private:
  std::unordered_map<std::string, std::reference_wrapper<const memory_instrumentation_base_t>>
    instrumented_;
};

#else

// No-op aggregator when instrumentation is disabled
class instrumentation_aggregator_t {
 public:
  instrumentation_aggregator_t() = default;
  instrumentation_aggregator_t(
    std::initializer_list<
      std::pair<std::string, std::reference_wrapper<const memory_instrumentation_base_t>>>)
  {
  }
  void add(const std::string&, const memory_instrumentation_base_t&) {}
  std::pair<size_t, size_t> collect() { return {0, 0}; }
  std::vector<std::tuple<std::string, size_t, size_t>> collect_per_wrapper() { return {}; }
  std::pair<size_t, size_t> collect_and_flush() { return {0, 0}; }
  void flush() {}
};

#endif  // CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Helper traits to detect container capabilities
namespace type_traits_utils {

template <typename T, typename = void>
struct has_reserve : std::false_type {};

template <typename T>
struct has_reserve<T, std::void_t<decltype(std::declval<T>().reserve(size_t{}))>> : std::true_type {
};

template <typename T, typename = void>
struct has_capacity : std::false_type {};

template <typename T>
struct has_capacity<T, std::void_t<decltype(std::declval<T>().capacity())>> : std::true_type {};

template <typename T, typename = void>
struct has_shrink_to_fit : std::false_type {};

template <typename T>
struct has_shrink_to_fit<T, std::void_t<decltype(std::declval<T>().shrink_to_fit())>>
  : std::true_type {};

template <typename T, typename = void>
struct has_push_back : std::false_type {};

template <typename T>
struct has_push_back<
  T,
  std::void_t<decltype(std::declval<T>().push_back(std::declval<typename T::value_type>()))>>
  : std::true_type {};

template <typename T, typename = void>
struct has_emplace_back : std::false_type {};

template <typename T>
struct has_emplace_back<T, std::void_t<decltype(std::declval<T>().emplace_back())>>
  : std::true_type {};

template <typename T, typename = void>
struct has_pop_back : std::false_type {};

template <typename T>
struct has_pop_back<T, std::void_t<decltype(std::declval<T>().pop_back())>> : std::true_type {};

template <typename T, typename = void>
struct has_data : std::false_type {};

template <typename T>
struct has_data<T, std::void_t<decltype(std::declval<T>().data())>> : std::true_type {};

template <typename T, typename = void>
struct has_resize : std::false_type {};

template <typename T>
struct has_resize<T, std::void_t<decltype(std::declval<T>().resize(size_t{}))>> : std::true_type {};

template <typename T, typename = void>
struct has_clear : std::false_type {};

template <typename T>
struct has_clear<T, std::void_t<decltype(std::declval<T>().clear())>> : std::true_type {};

template <typename T, typename = void>
struct has_max_size : std::false_type {};

template <typename T>
struct has_max_size<T, std::void_t<decltype(std::declval<T>().max_size())>> : std::true_type {};

template <typename T, typename = void>
struct has_front : std::false_type {};

template <typename T>
struct has_front<T, std::void_t<decltype(std::declval<T>().front())>> : std::true_type {};

template <typename T, typename = void>
struct has_back : std::false_type {};

template <typename T>
struct has_back<T, std::void_t<decltype(std::declval<T>().back())>> : std::true_type {};

}  // namespace type_traits_utils

#if CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Memory operation instrumentation wrapper for container-like types
template <typename T>
struct memop_instrumentation_wrapper_t : public memory_instrumentation_base_t {
  // Standard container type traits
  using value_type      = std::remove_reference_t<decltype(std::declval<T>()[0])>;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;

  static_assert(std::is_standard_layout_v<value_type>,
                "value_type must have standard layout for memory instrumentation");
  static constexpr size_t type_size = sizeof(value_type);

  // Proxy class to track reads and writes for a single element
  class element_proxy_t {
   public:
    element_proxy_t(value_type& ref, memop_instrumentation_wrapper_t& wrapper)
      : ref_(ref), wrapper_(wrapper)
    {
    }

    element_proxy_t& operator=(const value_type& value)
    {
      wrapper_.template record_store<value_type>();
      ref_ = value;
      return *this;
    }
    element_proxy_t& operator=(const element_proxy_t& other)
    {
      wrapper_.template record_store<value_type>();
      other.wrapper_.template record_load<value_type>();
      ref_ = other.ref_;
      return *this;
    }

    operator value_type() const
    {
      wrapper_.template record_load<value_type>();
      return ref_;
    }

    // // Allow implicit conversion to reference for functions expecting references
    // operator value_type&() { return ref_; }

    // operator const value_type&() const { return ref_; }

    // // Member access operator for structured types (e.g., type_2<f_t>)
    // value_type* operator->() { return &ref_; }

    // const value_type* operator->() const { return &ref_; }

    // Get underlying element reference (records a load)
    value_type& get()
    {
      wrapper_.template record_load<value_type>();
      return ref_;
    }

    const value_type& get() const
    {
      wrapper_.template record_load<value_type>();
      return ref_;
    }

    element_proxy_t& operator+=(const value_type& value)
    {
      wrapper_.template record_rmw<value_type>();
      ref_ += value;
      return *this;
    }
    element_proxy_t& operator-=(const value_type& value)
    {
      wrapper_.template record_rmw<value_type>();
      ref_ -= value;
      return *this;
    }
    element_proxy_t& operator*=(const value_type& value)
    {
      wrapper_.template record_rmw<value_type>();
      ref_ *= value;
      return *this;
    }
    element_proxy_t& operator/=(const value_type& value)
    {
      wrapper_.template record_rmw<value_type>();
      ref_ /= value;
      return *this;
    }
    element_proxy_t& operator++()
    {
      wrapper_.template record_rmw<value_type>();
      ++ref_;
      return *this;
    }
    element_proxy_t& operator--()
    {
      wrapper_.template record_rmw<value_type>();
      --ref_;
      return *this;
    }

    value_type operator++(int)
    {
      wrapper_.template record_rmw<value_type>();
      return ref_++;
    }
    value_type operator--(int)
    {
      wrapper_.template record_rmw<value_type>();
      return ref_--;
    }

    value_type& ref_;
    memop_instrumentation_wrapper_t& wrapper_;
  };

  // Instrumented iterator that tracks memory accesses
  template <typename IterT, bool IsConst>
  class instrumented_iterator_t {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = memop_instrumentation_wrapper_t::value_type;
    using difference_type   = std::ptrdiff_t;
    using pointer           = std::conditional_t<IsConst, const value_type*, value_type*>;
    using reference         = std::conditional_t<IsConst, value_type, element_proxy_t>;
    using wrapper_ptr       = std::conditional_t<IsConst,
                                                 const memop_instrumentation_wrapper_t*,
                                                 memop_instrumentation_wrapper_t*>;

    instrumented_iterator_t(IterT iter, wrapper_ptr wrapper) : iter_(iter), wrapper_(wrapper) {}

    // Dereference - returns proxy for non-const, tracks load for const
    auto operator*() const
    {
      if constexpr (IsConst) {
#ifdef CUOPT_ENABLE_MEMORY_INSTRUMENTATION
        wrapper_->byte_loads += sizeof(value_type);
#endif
        return *iter_;
      } else {
        return element_proxy_t(*iter_, *wrapper_);
      }
    }

    auto operator->() const { return &(*iter_); }

    instrumented_iterator_t& operator++()
    {
      ++iter_;
      return *this;
    }

    instrumented_iterator_t operator++(int)
    {
      auto tmp = *this;
      ++iter_;
      return tmp;
    }

    instrumented_iterator_t& operator--()
    {
      --iter_;
      return *this;
    }

    instrumented_iterator_t operator--(int)
    {
      auto tmp = *this;
      --iter_;
      return tmp;
    }

    instrumented_iterator_t& operator+=(difference_type n)
    {
      iter_ += n;
      return *this;
    }

    instrumented_iterator_t& operator-=(difference_type n)
    {
      iter_ -= n;
      return *this;
    }

    instrumented_iterator_t operator+(difference_type n) const
    {
      return instrumented_iterator_t(iter_ + n, wrapper_);
    }

    instrumented_iterator_t operator-(difference_type n) const
    {
      return instrumented_iterator_t(iter_ - n, wrapper_);
    }

    difference_type operator-(const instrumented_iterator_t& other) const
    {
      return iter_ - other.iter_;
    }

    auto operator[](difference_type n) const { return *(*this + n); }

    bool operator==(const instrumented_iterator_t& other) const { return iter_ == other.iter_; }
    bool operator!=(const instrumented_iterator_t& other) const { return iter_ != other.iter_; }
    bool operator<(const instrumented_iterator_t& other) const { return iter_ < other.iter_; }
    bool operator<=(const instrumented_iterator_t& other) const { return iter_ <= other.iter_; }
    bool operator>(const instrumented_iterator_t& other) const { return iter_ > other.iter_; }
    bool operator>=(const instrumented_iterator_t& other) const { return iter_ >= other.iter_; }

    IterT base() const { return iter_; }

    // Allow iterator_traits to access the underlying iterator
    friend struct std::iterator_traits<instrumented_iterator_t>;

   private:
    IterT iter_;
    wrapper_ptr wrapper_;
  };

  // Iterator type definitions (must come after instrumented_iterator_t)
  using iterator         = instrumented_iterator_t<decltype(std::declval<T>().begin()), false>;
  using const_iterator   = instrumented_iterator_t<decltype(std::declval<const T>().begin()), true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  // Constructors
  memop_instrumentation_wrapper_t() : array_() {}

  // Copy/move from underlying type
  memop_instrumentation_wrapper_t(const T& arr) : array_(arr) {}
  memop_instrumentation_wrapper_t(T&& arr) : array_(std::move(arr)) {}

  // Forwarding constructor for underlying container initialization
  // Only enabled for types that aren't the wrapper itself or the underlying type
  template <typename Arg,
            typename... Args,
            typename = std::enable_if_t<
              !std::is_same_v<std::decay_t<Arg>, memop_instrumentation_wrapper_t> &&
              !std::is_same_v<std::decay_t<Arg>, T> &&
              (sizeof...(Args) > 0 || !std::is_convertible_v<Arg, T>)>>
  explicit memop_instrumentation_wrapper_t(Arg&& arg, Args&&... args)
    : array_(std::forward<Arg>(arg), std::forward<Args>(args)...)
  {
  }

  memop_instrumentation_wrapper_t(const memop_instrumentation_wrapper_t& other)
    : memory_instrumentation_base_t(), array_(other.array_)
  {
  }

  memop_instrumentation_wrapper_t(memop_instrumentation_wrapper_t&& other) noexcept
    : memory_instrumentation_base_t(), array_(std::move(other.array_))
  {
  }

  memop_instrumentation_wrapper_t& operator=(const memop_instrumentation_wrapper_t& other)
  {
    if (this != &other) {
      reset_counters();
      array_ = other.array_;
    }
    return *this;
  }

  memop_instrumentation_wrapper_t& operator=(memop_instrumentation_wrapper_t&& other) noexcept
  {
    if (this != &other) {
      reset_counters();
      array_ = std::move(other.array_);
    }
    return *this;
  }

  element_proxy_t operator[](size_type index)
  {
    return element_proxy_t(underlying()[index], *this);
  }

  value_type operator[](size_type index) const
  {
    this->template record_load<value_type>();
    return underlying()[index];
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_front<U>::value, element_proxy_t> front()
  {
    return element_proxy_t(underlying().front(), *this);
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_front<U>::value, value_type> front() const
  {
    this->template record_load<value_type>();
    return underlying().front();
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_back<U>::value, element_proxy_t> back()
  {
    return element_proxy_t(underlying().back(), *this);
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_back<U>::value, value_type> back() const
  {
    this->template record_load<value_type>();
    return underlying().back();
  }

  // Iterators
  iterator begin() noexcept { return iterator(std::begin(underlying()), this); }
  const_iterator begin() const noexcept { return const_iterator(std::begin(underlying()), this); }
  const_iterator cbegin() const noexcept { return const_iterator(std::begin(underlying()), this); }

  iterator end() noexcept { return iterator(std::end(underlying()), this); }
  const_iterator end() const noexcept { return const_iterator(std::end(underlying()), this); }
  const_iterator cend() const noexcept { return const_iterator(std::end(underlying()), this); }

  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator(std::end(underlying()));
  }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

  // Capacity
  bool empty() const noexcept { return std::begin(underlying()) == std::end(underlying()); }
  size_type size() const noexcept
  {
    return std::distance(std::begin(underlying()), std::end(underlying()));
  }

  // Conditional methods - only available if underlying type supports them
  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_max_size<U>::value, size_type> max_size() const noexcept
  {
    return underlying().max_size();
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_capacity<U>::value, size_type> capacity() const noexcept
  {
    return underlying().capacity();
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_reserve<U>::value> reserve(size_type new_cap)
  {
    underlying().reserve(new_cap);
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_shrink_to_fit<U>::value> shrink_to_fit()
  {
    underlying().shrink_to_fit();
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_clear<U>::value> clear() noexcept
  {
    underlying().clear();
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_push_back<U>::value> push_back(const value_type& value)
  {
    // we should probably take into account possible copies done by std::vector. oh well.
    // hot loops shouldn't be doing such operations anyway
    this->template record_store<value_type>();
    underlying().push_back(value);
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_push_back<U>::value> push_back(value_type&& value)
  {
    this->template record_store<value_type>();
    underlying().push_back(std::move(value));
  }

  template <typename U = T, typename... Args>
  std::enable_if_t<type_traits_utils::has_emplace_back<U>::value> emplace_back(Args&&... args)
  {
    this->template record_store<value_type>();
    underlying().emplace_back(std::forward<Args>(args)...);
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_pop_back<U>::value> pop_back()
  {
    this->template record_load<value_type>();  // Reading the element before removal
    underlying().pop_back();
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_resize<U>::value> resize(size_type count)
  {
    size_type old_size = underlying().size();
    underlying().resize(count);
    if (count > old_size) {
      this->byte_stores += (count - old_size) * type_size;  // New elements initialized
    }
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_resize<U>::value> resize(size_type count,
                                                                   const value_type& value)
  {
    size_type old_size = underlying().size();
    underlying().resize(count, value);
    if (count > old_size) { this->byte_stores += (count - old_size) * type_size; }
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_data<U>::value, value_type*> data() noexcept
  {
    return underlying().data();
  }

  template <typename U = T>
  std::enable_if_t<type_traits_utils::has_data<U>::value, const value_type*> data() const noexcept
  {
    return underlying().data();
  }

  // Access to underlying array
  operator T&() { return underlying(); }
  operator const T&() const { return underlying(); }

  T&& release_array() { return std::move(array_); }

  T& underlying() { return array_; }
  const T& underlying() const { return array_; }

 private:
  T array_;
};

#else  // !CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Zero-overhead passthrough wrapper when instrumentation is disabled
// Provides the same interface as the instrumented version but just forwards to the underlying
// container
template <typename T>
struct memop_instrumentation_wrapper_t : public memory_instrumentation_base_t {
  using value_type             = typename T::value_type;
  using size_type              = typename T::size_type;
  using difference_type        = typename T::difference_type;
  using reference              = typename T::reference;
  using const_reference        = typename T::const_reference;
  using pointer                = typename T::pointer;
  using const_pointer          = typename T::const_pointer;
  using iterator               = typename T::iterator;
  using const_iterator         = typename T::const_iterator;
  using reverse_iterator       = typename T::reverse_iterator;
  using const_reverse_iterator = typename T::const_reverse_iterator;

  // Constructors - forward everything to the underlying container
  memop_instrumentation_wrapper_t() = default;
  memop_instrumentation_wrapper_t(const T& arr) : array_(arr) {}
  memop_instrumentation_wrapper_t(T&& arr) : array_(std::move(arr)) {}

  template <typename Arg,
            typename... Args,
            typename = std::enable_if_t<
              !std::is_same_v<std::decay_t<Arg>, memop_instrumentation_wrapper_t> &&
              !std::is_same_v<std::decay_t<Arg>, T> &&
              (sizeof...(Args) > 0 || !std::is_convertible_v<Arg, T>)>>
  explicit memop_instrumentation_wrapper_t(Arg&& arg, Args&&... args)
    : array_(std::forward<Arg>(arg), std::forward<Args>(args)...)
  {
  }

  memop_instrumentation_wrapper_t(const memop_instrumentation_wrapper_t& other)
    : array_(other.array_)
  {
  }

  memop_instrumentation_wrapper_t(memop_instrumentation_wrapper_t&& other) noexcept
    : array_(std::move(other.array_))
  {
  }

  memop_instrumentation_wrapper_t& operator=(const memop_instrumentation_wrapper_t& other)
  {
    if (this != &other) { array_ = other.array_; }
    return *this;
  }

  memop_instrumentation_wrapper_t& operator=(memop_instrumentation_wrapper_t&& other) noexcept
  {
    if (this != &other) { array_ = std::move(other.array_); }
    return *this;
  }

  // Element access - direct passthrough
  reference operator[](size_type index) { return underlying()[index]; }
  const_reference operator[](size_type index) const { return underlying()[index]; }

  reference front() { return underlying().front(); }
  const_reference front() const { return underlying().front(); }

  reference back() { return underlying().back(); }
  const_reference back() const { return underlying().back(); }

  pointer data() noexcept { return underlying().data(); }
  const_pointer data() const noexcept { return underlying().data(); }

  // Iterators - use underlying container's iterators directly
  iterator begin() noexcept { return underlying().begin(); }
  const_iterator begin() const noexcept { return underlying().begin(); }
  const_iterator cbegin() const noexcept { return underlying().cbegin(); }

  iterator end() noexcept { return underlying().end(); }
  const_iterator end() const noexcept { return underlying().end(); }
  const_iterator cend() const noexcept { return underlying().cend(); }

  reverse_iterator rbegin() noexcept { return underlying().rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return underlying().rbegin(); }
  const_reverse_iterator crbegin() const noexcept { return underlying().crbegin(); }

  reverse_iterator rend() noexcept { return underlying().rend(); }
  const_reverse_iterator rend() const noexcept { return underlying().rend(); }
  const_reverse_iterator crend() const noexcept { return underlying().crend(); }

  // Capacity
  bool empty() const noexcept { return underlying().empty(); }
  size_type size() const noexcept { return underlying().size(); }
  size_type max_size() const noexcept { return underlying().max_size(); }
  size_type capacity() const noexcept { return underlying().capacity(); }

  void reserve(size_type new_cap) { underlying().reserve(new_cap); }
  void shrink_to_fit() { underlying().shrink_to_fit(); }

  // Modifiers
  void clear() noexcept { underlying().clear(); }
  void push_back(const value_type& value) { underlying().push_back(value); }
  void push_back(value_type&& value) { underlying().push_back(std::move(value)); }

  template <typename... Args>
  void emplace_back(Args&&... args)
  {
    underlying().emplace_back(std::forward<Args>(args)...);
  }

  void pop_back() { underlying().pop_back(); }
  void resize(size_type count) { underlying().resize(count); }
  void resize(size_type count, const value_type& value) { underlying().resize(count, value); }

  // Conversion operators
  operator T&() { return underlying(); }
  operator const T&() const { return underlying(); }

  T&& release_array() { return std::move(array_); }

  T& underlying() { return array_; }
  const T& underlying() const { return array_; }

 private:
  T array_;
};

#endif  // CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Convenience alias for instrumented std::vector
template <typename T>
using ins_vector = memop_instrumentation_wrapper_t<std::vector<T>>;

}  // namespace cuopt
