/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/handle.hpp>

#include <string>

// test_netlib --se-bench: exact steepest-edge weights of every stored node's basic rows from the
// device-built explicit inverses, vs the host phase2::initialize_steepest_edge_norms.
int se_bench_main(const raft::handle_t& handle, const std::string& file, int repeats, int prepare_threads,
                  int host_timed_seats);
