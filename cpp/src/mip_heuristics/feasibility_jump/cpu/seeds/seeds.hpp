/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../internal.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct row_repair_move_t {
  f_t effect;
  i_t var;
  f_t coeff;
  f_t new_val;
};

}  // namespace cuopt::mathematical_optimization::mip
