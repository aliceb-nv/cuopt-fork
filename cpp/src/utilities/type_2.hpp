/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector_types.h>

namespace cuopt {

template <typename T>
struct type_2 {
  using type = void;
};

template <>
struct type_2<int> {
  using type = int2;
};

template <>
struct type_2<float> {
  using type = float2;
};

template <>
struct type_2<double> {
  using type = double2;
};

template <typename T>
struct scalar_type {
  using type = void;
};

template <>
struct scalar_type<int2> {
  using type = int;
};

template <>
struct scalar_type<float2> {
  using type = float;
};

template <>
struct scalar_type<double2> {
  using type = double;
};

template <>
struct scalar_type<const int2> {
  using type = const int;
};

template <>
struct scalar_type<const float2> {
  using type = const float;
};

template <>
struct scalar_type<const double2> {
  using type = const double;
};

#if defined(__CUDACC__)
#define CUOPT_TYPE_2_HOST_DEVICE inline __host__ __device__
#else
#define CUOPT_TYPE_2_HOST_DEVICE inline
#endif

template <typename f_t2>
CUOPT_TYPE_2_HOST_DEVICE typename scalar_type<f_t2>::type& get_lower(f_t2& value)
{
  return value.x;
}

template <typename f_t2>
CUOPT_TYPE_2_HOST_DEVICE typename scalar_type<f_t2>::type& get_upper(f_t2& value)
{
  return value.y;
}

#undef CUOPT_TYPE_2_HOST_DEVICE

}  // namespace cuopt
