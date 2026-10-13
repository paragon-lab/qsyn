/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Proxy functions and simulated annealing to optimize ternary tree ]
  Author       [ April Wang (april864) ]
*/

#include "tree_optimizations.hpp"

#include <fmt/format.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "device/device.hpp"
#include "device/device_analysis.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "hamiltonian/qubit_hamiltonian.hpp"
#include "ternary_tree.hpp"
#include "tree_rotations.hpp"
#include "util/simulated_annealing.hpp"

namespace qsyn::hamiltonian {

void TreeOracle::_dfs(TernaryTree const& tree, dvlab::APSPResult<QubitIdType> const& apsp, int& timer,
                      size_t v, size_t p, int current_depth, double cost_from_parent) {
    _parent[v]       = p;
    _depth[v]        = current_depth;
    _edge_cost_up[v] = cost_from_parent;
    _dfs_start[v]    = ++timer;

    auto v_node = dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(v));
    if (v_node && v_node->qubit_label.has_value()) {
        size_t phys_v = v_node->qubit_label.value();

        for (auto branch : {BranchType::left, BranchType::mid, BranchType::right}) {
            auto* child = v_node->get_child(branch);
            if (child && !child->is_leg()) {
                auto child_node = dynamic_cast<TernaryQubitNode*>(child);
                if (child_node && child_node->qubit_label.has_value()) {
                    size_t next      = child_node->id;
                    size_t phys_next = child_node->qubit_label.value();

                    double weight = apsp.distance[phys_v][phys_next];
                    _dfs(tree, apsp, timer, next, v, current_depth + 1, weight);
                }
            }
        }
    }
}

double TreeOracle::_calc_branch_tail(dvlab::APSPResult<QubitIdType> const& apsp, TernaryQubitNode* start_node, BranchType initial_branch) {
    TernaryEdge* edge = start_node->get_edge(initial_branch);
    if (!edge || !edge->target || edge->target->is_leg()) return 0.0;

    TernaryNode* current = edge->target.get();
    auto curr_qnode      = dynamic_cast<TernaryQubitNode*>(current);
    size_t phys_curr     = curr_qnode->qubit_label.value();
    size_t phys_prev     = start_node->qubit_label.value();

    double cost = apsp.distance[phys_prev][phys_curr];

    while (!current->is_leg()) {
        TernaryNode* next = current->get_right_child();
        if (next->is_leg()) break;

        auto next_qnode  = dynamic_cast<TernaryQubitNode*>(next);
        size_t phys_next = next_qnode->qubit_label.value();

        cost += apsp.distance[phys_curr][phys_next];
        current   = next;
        phys_curr = phys_next;
    }
    return cost;
}

TreeOracle::TreeOracle(TernaryTree const& tree, dvlab::APSPResult<QubitIdType> const& apsp) : _n_modes(tree.num_qubits()) {
    if (_n_modes == 0) return;

    _parent.assign(_n_modes, 0);
    _depth.assign(_n_modes, 0);
    _edge_cost_up.assign(_n_modes, 0.0);
    _dfs_start.assign(_n_modes, 0);
    _tail_weight.assign(_n_modes, 0.0);

    auto root_node = dynamic_cast<TernaryQubitNode*>(tree.get_root());
    if (!root_node) throw std::runtime_error("Root is not a qubit node");
    int timer = 0;
    _dfs(tree, apsp, timer, root_node->id, root_node->id, 0, 0.0);

    for (size_t u = 0; u < _n_modes; ++u) {
        auto u_node = dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(u));
        if (u_node && u_node->qubit_label.has_value()) {
            double x_cost   = _calc_branch_tail(apsp, u_node, BranchType::left);
            double y_cost   = _calc_branch_tail(apsp, u_node, BranchType::mid);
            _tail_weight[u] = (x_cost + y_cost) / 2.0;
        }
    }
}

double TreeOracle::get_tree_distance(size_t u, size_t v) const {
    double dist = 0.0;

    while (_depth[u] > _depth[v]) {
        dist += _edge_cost_up[u];
        u = _parent[u];
    }
    while (_depth[v] > _depth[u]) {
        dist += _edge_cost_up[v];
        v = _parent[v];
    }

    while (u != v) {
        dist += _edge_cost_up[u] + _edge_cost_up[v];
        u = _parent[u];
        v = _parent[v];
    }
    return dist;
}

double TreeOracle::get_subtree_weight(std::vector<size_t> const& nodes) const {
    if (nodes.empty()) return 0.0;
    if (nodes.size() == 1) return 0.0;
    if (nodes.size() == 2) return get_tree_distance(nodes[0], nodes[1]);

    // Weight of the unique Steiner subtree connecting ``nodes`` on this tree.
    //
    // Sort terminals by DFS discovery time so they appear in Euler-tour /
    // circular order around that subtree. Then
    //
    //   w(Steiner) = (1/2) * sum_i dist(t_i, t_{i+1}),   t_{k+1} := t_1.
    //
    // This is the same DFS-doubling identity used in the classical MST
    // 2-approximation for metric Steiner tree (there the host graph is a
    // metric, so shortcutting only gives an upper bound; here the host is
    // already a tree, so the identity is exact). See e.g.:
    //   L. Kou, G. Markowsky, L. Berman, "A fast algorithm for Steiner trees",
    //   Acta Informatica 15 (1981) 141-145;
    //   V. V. Vazirani, Approximation Algorithms, Springer, 2001, Ch. 3.
    std::vector<size_t> sorted_nodes = nodes;
    std::ranges::sort(sorted_nodes, std::less<>{}, [this](size_t node) { return _dfs_start[node]; });
    double total_weight = 0.0;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        size_t u = sorted_nodes[i];
        size_t v = sorted_nodes[(i + 1) % sorted_nodes.size()];
        total_weight += get_tree_distance(u, v);
    }

    return total_weight / 2.0;
}

ApspPathCost::ApspPathCost(FermionHamiltonian const& f_ham, dvlab::APSPResult<QubitIdType> const& apsp)
    : _f_ham(f_ham), _apsp(apsp) {}

double ApspPathCost::evaluate(TernaryTree const& tree) const {
    TreeOracle oracle(tree, _apsp);
    double total = 0.0;

    TernaryTreeMapping mapping(tree);
    QubitHamiltonian q_ham = qubitize(_f_ham, mapping);

    for (auto const& term : q_ham) {
        total += oracle.get_subtree_weight(term.non_identity_qubits());
    }

    return total;
}

PauliWeightCost::PauliWeightCost(FermionHamiltonian const& f_ham) : _f_ham(f_ham) {}

double PauliWeightCost::evaluate(TernaryTree const& tree) const {
    TernaryTreeMapping mapping(tree);
    QubitHamiltonian q_ham = qubitize(_f_ham, mapping);

    double total_weight = 0;
    for (auto const& term : q_ham) {
        total_weight += static_cast<double>(term.weight());
    }
    return total_weight;
}

TernaryTree optimize_ternary_tree_mapping(
    TernaryTree const& initial_tree,
    TernaryTreeCost const& cost,
    device::Device const* device,
    std::optional<std::uint32_t> seed) {
    using util::SimulatedAnnealing;
    using MutateFn = SimulatedAnnealing<TernaryTree, double>::MutateFn;

    // Derive independent streams for mutations vs. SA accept/mutate selection.
    std::uint32_t rotator_seed = 0;
    std::uint32_t sa_seed      = 0;
    if (seed.has_value()) {
        std::seed_seq seq{*seed};
        std::array<std::uint32_t, 2> seeds{};
        seq.generate(seeds.begin(), seeds.end());
        rotator_seed = seeds[0];
        sa_seed      = seeds[1];
        fmt::println("SA seed: {} (rotator={}, accept={})", *seed, rotator_seed, sa_seed);
    } else {
        std::random_device rd;
        rotator_seed = rd();
        sa_seed      = rd();
    }

    TreeRotator rotator{rotator_seed};
    auto const mutate_fns = std::vector<MutateFn>{
        [&](TernaryTree& tree) {
            if (device) {
                rotator.cp_leaf_move(&tree, *device);
            } else {
                rotator.ncp_leaf_move(&tree);
            }
        },
        [&](TernaryTree& tree) { rotator.root_change(&tree); },
        [&](TernaryTree& tree) { rotator.pauli_shuffle(&tree); },
        [&](TernaryTree& tree) { rotator.mode_association_swap(&tree); },
        [&](TernaryTree& tree) { rotator.majorana_braiding_change(&tree); },
    };

    auto const sa = SimulatedAnnealing<TernaryTree, double>(
        /* init_temp    = */ 20.0,
        /* cooling_rate = */ 0.99995,
        /* min_temp     = */ 0.3,
        /* cost_fn      = */ [&](TernaryTree const& tree) { return cost.evaluate(tree); },
        /* mutate_fns   = */ mutate_fns,
        /* seed         = */ sa_seed);

    auto const [best_tree, best_cost] = sa.run(initial_tree);
    fmt::println("Final optimized cost: {}\n", best_cost);
    return best_tree;
}

}  // namespace qsyn::hamiltonian
