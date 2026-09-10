/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../internal.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void apply_bipartite_matching_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.problem->nnz > fj_seed_nnz_limit) return;

  const auto started = std::chrono::steady_clock::now();
  auto timed_out     = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
           fj_matching_budget_s;
  };
  const f_t tol = 1e-6;

  struct exact_one_row_t {
    i_t begin, end;
  };
  std::vector<exact_one_row_t> rows;
  for (i_t row = 0; row < fj_cpu.problem->n_constraints; ++row) {
    if ((row & 0xFFF) == 0 && timed_out()) return;

    const f_t lb = fj_cpu.problem->cstr_lb[row];
    const f_t ub = fj_cpu.problem->cstr_ub[row];
    if (!std::isfinite(lb) || !std::isfinite(ub) || std::abs(lb - ub) > tol) continue;

    const i_t begin = fj_cpu.problem->offsets[row];
    const i_t end   = fj_cpu.problem->offsets[row + 1];
    if (begin == end || end - begin > fj_matching_max_row_width) continue;

    const f_t scale = fj_cpu.problem->coefficients[begin];
    if (!std::isfinite(scale) || std::abs(scale) <= tol || std::abs(lb / scale - 1) > 1e-5) continue;

    bool uniform_binary = true;
    for (i_t p = begin; p < end && uniform_binary; ++p) {
      const f_t coeff     = fj_cpu.problem->coefficients[p];
      const f_t agreement = tol * std::max((f_t)1, std::abs(scale));
      uniform_binary = fj_cpu.h_is_binary_variable[fj_cpu.problem->variables[p]] &&
                       std::abs(coeff - scale) <= agreement;
    }
    if (uniform_binary) rows.push_back({begin, end});
  }
  if (rows.size() < 2 || timed_out()) return;

  const i_t n_rows      = (i_t)rows.size();
  const i_t n_variables = fj_cpu.problem->n_variables;
  std::vector<i_t> degree(n_variables, 0);
  std::vector<i_t> endpoint_a(n_variables, -1);
  std::vector<i_t> endpoint_b(n_variables, -1);
  for (i_t row = 0; row < n_rows; ++row) {
    for (i_t p = rows[row].begin; p < rows[row].end; ++p) {
      const i_t var = fj_cpu.problem->variables[p];
      if (degree[var] == 0) endpoint_a[var] = row;
      else if (degree[var] == 1) endpoint_b[var] = row;
      ++degree[var];
    }
  }

  struct edge_t {
    int to, reverse, capacity;
    f_t cost;
    i_t var;
  };
  auto add_edge = [](std::vector<std::vector<edge_t>>& graph, int from, int to, f_t cost, i_t var) {
    const int back = (int)graph[to].size();
    graph[from].push_back({to, back, 1, cost, var});
    graph[to].push_back({from, (int)graph[from].size() - 1, 0, -cost, -1});
  };

  std::vector<int8_t> color(n_rows, -1);
  std::vector<int8_t> state(n_variables, -1);
  std::vector<i_t> side_index(n_rows, -1);
  std::vector<i_t> component_rows, component_vars, left, right;
  std::queue<i_t> pending;
  bool installed = false;

  for (i_t root = 0; root < n_rows && !timed_out(); ++root) {
    if (color[root] >= 0) continue;

    component_rows.clear();
    component_vars.clear();
    color[root] = 0;
    pending.push(root);
    bool valid = true;
    while (!pending.empty()) {
      const i_t row = pending.front();
      pending.pop();
      component_rows.push_back(row);
      for (i_t p = rows[row].begin; p < rows[row].end; ++p) {
        const i_t var = fj_cpu.problem->variables[p];
        // A variable outside exactly two rows is not an edge, and a self-loop cannot be 2-coloured.
        if (degree[var] != 2 || endpoint_a[var] == endpoint_b[var]) {
          valid = false;
          continue;
        }
        if (endpoint_a[var] == row) component_vars.push_back(var);
        const i_t other = endpoint_a[var] == row ? endpoint_b[var] : endpoint_a[var];
        if (color[other] < 0) {
          color[other] = 1 - color[row];
          pending.push(other);
        } else if (color[other] == color[row]) {
          valid = false;
        }
      }
      if ((component_rows.size() & 0x3FF) == 0 && timed_out()) return;
    }
    if (!valid || component_vars.empty()) continue;

    left.clear();
    right.clear();
    for (i_t row : component_rows)
      (color[row] == 0 ? left : right).push_back(row);
    if (left.size() != right.size()) continue;
    for (i_t k = 0; k < (i_t)left.size(); ++k)
      side_index[left[k]] = k;
    for (i_t k = 0; k < (i_t)right.size(); ++k)
      side_index[right[k]] = k;

    const int side   = (int)left.size();
    const int source = 2 * side;
    const int sink   = source + 1;
    std::vector<std::vector<edge_t>> graph(sink + 1);
    for (int k = 0; k < side; ++k) {
      add_edge(graph, source, k, 0, -1);
      add_edge(graph, side + k, sink, 0, -1);
    }
    // Every perfect matching uses exactly one variable edge per row, so shifting all of them by a
    // constant moves every matching's cost equally and leaves the cheapest one unchanged. Shifting
    // the negatives away is what lets the potentials below start at zero.
    f_t cheapest = 0;
    for (i_t var : component_vars) {
      const f_t cost = fj_cpu.problem->h_obj_coeffs[var];
      if (!std::isfinite(cost)) {
        valid = false;
        break;
      }
      cheapest = std::min(cheapest, cost);
    }
    if (!valid) continue;
    const f_t shift = -cheapest;

    for (i_t var : component_vars) {
      i_t a = endpoint_a[var];
      i_t b = endpoint_b[var];
      if (color[a] == 1) std::swap(a, b);
      add_edge(graph, side_index[a], side + side_index[b], fj_cpu.problem->h_obj_coeffs[var] + shift, var);
    }

    // Node potentials hold every reduced cost at or above zero, which is what makes Dijkstra
    // applicable. All shifted costs start non-negative, so the potentials start at zero. Rounding
    // can still leave a tree edge fractionally negative once the potentials move, so relaxation
    // below skips settled nodes: that keeps every predecessor older than its successor in
    // settlement order, which is what makes the retrace terminate.
    int flow = 0;
    std::vector<f_t> potential(graph.size(), 0);
    std::vector<f_t> distance(graph.size());
    std::vector<int> previous_node(graph.size());
    std::vector<int> previous_edge(graph.size());
    std::vector<uint8_t> settled(graph.size());
    using heap_entry_t = std::pair<f_t, int>;

    while (flow < side && !timed_out()) {
      std::fill(distance.begin(), distance.end(), std::numeric_limits<f_t>::infinity());
      std::fill(previous_node.begin(), previous_node.end(), -1);
      std::fill(settled.begin(), settled.end(), 0);
      distance[source] = 0;
      std::priority_queue<heap_entry_t, std::vector<heap_entry_t>, std::greater<heap_entry_t>> heap;
      heap.push({0, source});

      while (!heap.empty()) {
        const auto [reached_at, from] = heap.top();
        heap.pop();
        if (settled[from]) continue;
        settled[from] = 1;
        for (int e = 0; e < (int)graph[from].size(); ++e) {
          const auto& edge = graph[from][e];
          if (!edge.capacity || settled[edge.to]) continue;
          const f_t reduced = edge.cost + potential[from] - potential[edge.to];
          cuopt_assert(reduced >= -1e-9 * std::max((f_t)1, std::abs(edge.cost)),
                       "potentials failed to keep the reduced cost non-negative");
          if (reached_at + reduced >= distance[edge.to]) continue;
          distance[edge.to]      = reached_at + reduced;
          previous_node[edge.to] = from;
          previous_edge[edge.to] = e;
          heap.push({distance[edge.to], edge.to});
        }
      }
      if (previous_node[sink] < 0) break;

      for (int node = 0; node < (int)graph.size(); ++node)
        if (std::isfinite(distance[node])) potential[node] += distance[node];

      for (int node = sink; node != source; node = previous_node[node]) {
        cuopt_assert(previous_node[node] >= 0, "augmenting path is broken");
        auto& edge = graph[previous_node[node]][previous_edge[node]];
        --edge.capacity;
        ++graph[node][edge.reverse].capacity;
      }
      ++flow;
    }
    if (flow != side) continue;

    for (i_t var : component_vars)
      state[var] = 0;
    for (int node = 0; node < side; ++node)
      for (const auto& edge : graph[node])
        if (edge.var >= 0 && edge.capacity == 0) state[edge.var] = 1;
    installed = true;
  }
  if (!installed) return;

  recompute_lhs(fj_cpu);
  const i_t baseline = fj_cpu.violated_constraints.size();
  const auto anchor  = fj_cpu.h_assignment;
  for (i_t var = 0; var < n_variables; ++var)
    if (state[var] >= 0) fj_cpu.h_assignment[var] = state[var];

  recompute_lhs(fj_cpu);
  const i_t candidate = fj_cpu.violated_constraints.size();
  // Kept when it reaches feasibility outright, otherwise only on a strict gain.
  if (candidate != 0 && candidate >= baseline) {
    fj_cpu.h_assignment = anchor;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}


}  // namespace cuopt::mathematical_optimization::mip
