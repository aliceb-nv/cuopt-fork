/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/handle.hpp>

#include <string>

// test_netlib --inv-bench: explicit inverses of a pool snapshot's base factors, built on the device
// (column and row forms) vs the single-thread host build plus its upload.
int inv_bench_main(const raft::handle_t& handle, const std::string& file, int repeats, int prepare_threads);
