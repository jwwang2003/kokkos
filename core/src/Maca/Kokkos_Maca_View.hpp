/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_View.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 11:58:39 AM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_EXPERIMENTAL_MACA_VIEW_HPP
#define KOKKOS_EXPERIMENTAL_MACA_VIEW_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_MACA)

#include <Maca/Kokkos_Maca_Runtime.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

#include <map>
#include <mutex>
#include <type_traits>

#if __has_include(<mcr/mc_channel_descriptor.h>)
#include <mcr/mc_channel_descriptor.h>
#elif __has_include(<mc/mc_channel_descriptor.h>)
#include <mc/mc_channel_descriptor.h>
#endif

#if __has_include(<mcr/mc_texture_indirect_functions.h>)
#include <mcr/mc_texture_indirect_functions.h>
#elif __has_include(<mc/mc_texture_indirect_functions.h>)
#include <mc/mc_texture_indirect_functions.h>
#endif

namespace Kokkos {
namespace Impl {

template <typename ValueType, typename AliasType>
struct MacaLDGFetch {
  const ValueType* m_ptr               = nullptr;
  const ValueType* m_texture_base_ptr  = nullptr;
  mcTextureObject_t m_texture_object   = 0;

  template <typename iType>
  KOKKOS_FUNCTION ValueType operator[](const iType& i) const {
    KOKKOS_IF_ON_DEVICE((
        if (m_texture_object != 0) {
          const int offset =
              static_cast<int>((m_ptr - m_texture_base_ptr) + i);
          AliasType v = tex1Dfetch<AliasType>(m_texture_object, offset);
          return *(reinterpret_cast<ValueType*>(&v));
        }
        return m_ptr[i];))
    KOKKOS_IF_ON_HOST((return m_ptr[i];))
  }

  KOKKOS_FUNCTION
  operator const ValueType*() const { return m_ptr; }

  KOKKOS_DEFAULTED_FUNCTION
  MacaLDGFetch() = default;

  KOKKOS_FUNCTION
  explicit MacaLDGFetch(const ValueType* arg_ptr,
                        const ValueType* arg_texture_base_ptr = nullptr,
                        mcTextureObject_t arg_texture_object = 0)
      : m_ptr(arg_ptr),
        m_texture_base_ptr(arg_texture_base_ptr ? arg_texture_base_ptr
                                                : arg_ptr),
        m_texture_object(arg_texture_object) {}

  KOKKOS_FUNCTION
  MacaLDGFetch(MacaLDGFetch const rhs, size_t offset)
      : m_ptr(rhs.m_ptr + offset),
        m_texture_base_ptr(rhs.m_texture_base_ptr),
        m_texture_object(rhs.m_texture_object) {}
};

template <typename ValueType, typename AliasType, typename MemorySpace>
struct MacaTextureObjectCache {
  using tracker_type = Kokkos::Impl::SharedAllocationTracker;

  struct CacheKey {
    const void* record = nullptr;
    const void* data   = nullptr;
    size_t size        = 0;

    bool operator<(CacheKey const& rhs) const {
      if (record != rhs.record) return record < rhs.record;
      if (data != rhs.data) return data < rhs.data;
      return size < rhs.size;
    }
  };

  struct CacheState {
    std::mutex mutex;
    std::map<CacheKey, mcTextureObject_t> textures;

    ~CacheState() {
      for (auto const& entry : textures) {
        if (entry.second != 0) {
          mcDestroyTextureObject(entry.second);
        }
      }
    }
  };

  struct TextureHandle {
    const ValueType* base_ptr = nullptr;
    mcTextureObject_t texture = 0;
  };

  static TextureHandle create(ValueType* arg_data_ptr,
                              tracker_type const& arg_tracker) {
    auto* const record = arg_tracker.template get_record<MemorySpace>();
    if (record == nullptr) return {};

    auto* const base_ptr = static_cast<const ValueType*>(record->data());
    const size_t size    = record->size();
    if (base_ptr == nullptr || arg_data_ptr < base_ptr) return {};

    const size_t size_in_bytes = (size / sizeof(AliasType)) * sizeof(AliasType);
    const size_t offset_in_bytes =
        static_cast<size_t>(reinterpret_cast<char const*>(arg_data_ptr) -
                            reinterpret_cast<char const*>(base_ptr));
    if (size_in_bytes == 0 || offset_in_bytes >= size_in_bytes) return {};

    auto& cache = state();
    std::lock_guard<std::mutex> lock(cache.mutex);
    CacheKey const key{record, base_ptr, size_in_bytes};
    auto [it, inserted] = cache.textures.emplace(key, mcTextureObject_t{0});
    if (inserted) {
      mcResourceDesc res_desc{};
      mcTextureDesc tex_desc{};
      res_desc.resType                = mcResourceTypeLinear;
      res_desc.res.linear.devPtr      = const_cast<ValueType*>(base_ptr);
      res_desc.res.linear.desc        = mcCreateChannelDesc<AliasType>();
      res_desc.res.linear.sizeInBytes = size_in_bytes;
      tex_desc.readMode               = mcReadModeElementType;
      if (mcCreateTextureObject(&it->second, &res_desc, &tex_desc, nullptr) !=
          mcSuccess) {
        cache.textures.erase(it);
        return {};
      }
    }

    return {base_ptr, it->second};
  }

 private:
  static CacheState& state() {
    static CacheState cache_state;
    return cache_state;
  }
};

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class Traits>
class ViewDataHandle<
    Traits,
    std::enable_if_t<(
        (std::is_same_v<typename Traits::memory_space, Kokkos::MacaSpace> ||
         std::is_same_v<typename Traits::memory_space,
                        Kokkos::MacaManagedSpace>)&&
        std::is_trivially_copyable_v<typename Traits::const_value_type> &&
        std::is_same_v<typename Traits::const_value_type,
                       typename Traits::value_type> &&
        (sizeof(typename Traits::const_value_type) == 4 ||
         sizeof(typename Traits::const_value_type) == 8 ||
         sizeof(typename Traits::const_value_type) == 16) &&
        (Traits::memory_traits::is_random_access != 0))>> {
 public:
  using track_type = Kokkos::Impl::SharedAllocationTracker;

  using value_type  = typename Traits::const_value_type;
  using return_type = typename Traits::const_value_type;

  using alias_type = std::conditional_t<
      (sizeof(value_type) == 4), int,
      std::conditional_t<
          (sizeof(value_type) == 8), ::int2,
          std::conditional_t<(sizeof(value_type) == 16), ::int4, void>>>;

  using handle_type = Kokkos::Impl::MacaLDGFetch<value_type, alias_type>;

  KOKKOS_INLINE_FUNCTION
  static handle_type const& assign(handle_type const& arg_handle,
                                   track_type const&) {
    return arg_handle;
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type const assign(handle_type const& arg_handle, size_t offset) {
    return handle_type(arg_handle, offset);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* arg_data_ptr,
                            track_type const& arg_tracker) {
    if (arg_data_ptr == nullptr) return handle_type();

    handle_type result(arg_data_ptr);
    KOKKOS_IF_ON_HOST((
        using cache_type = Kokkos::Impl::MacaTextureObjectCache<
            value_type, alias_type, typename Traits::memory_space>;
        auto const texture = cache_type::create(arg_data_ptr, arg_tracker);
        if (texture.texture != 0) {
          result = handle_type(arg_data_ptr, texture.base_ptr, texture.texture);
        }))
    return result;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif
#endif
