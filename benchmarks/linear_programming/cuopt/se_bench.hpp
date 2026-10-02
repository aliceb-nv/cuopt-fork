/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/gpu_node_pool.hpp>

#include <raft/core/handle.hpp>

#include <string>
#include <vector>

// Replaces the basic entries of nodes[seats[k]].dy_se with the exact weights of prepared[seats[k]]'s
// basis (device inverse build + v2 weight kernel). Returns the wall time in seconds.
double se_exact_weights(int m, const std::vector<cuopt::mathematical_optimization::mip::batch_seat_state_t<int, double>>& prepared,
                        const std::vector<int>& seats,
                        std::vector<cuopt::mathematical_optimization::mip::gpu_batch_node_t<int, double>>& nodes);

// test_netlib --se-bench: exact steepest-edge weights of every stored node's basic rows from the
// device-built explicit inverses, vs the host phase2::initialize_steepest_edge_norms.
int se_bench_main(const raft::handle_t& handle, const std::string& file, int repeats, int prepare_threads,
                  int host_timed_seats);
