/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/handle.hpp>

#include <string>

// test_netlib --ftran-bench: batched base-factor FTRAN kernels on a pool snapshot's nodes.
int ftran_bench_main(const raft::handle_t& handle,
                     const std::string& file,
                     int replicas,
                     int repeats,
                     int prepare_threads);
