/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Proxy functions and simulated annealing to optimize ternary tree ]
  Author       [ April Wang (april864) ]
*/

#include "tree_optimizations.hpp"

#include <fmt/format.h>

#include <algorithm>
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

void TreeOracle::dfs(const TernaryTree& tree, const dvlab::APSPResult<QubitIdType>& apsp, int& timer,
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
                    dfs(tree, apsp, timer, next, v, current_depth + 1, weight);
                }
            }
        }
    }
};

double TreeOracle::calc_branch_tail(const dvlab::APSPResult<QubitIdType>& apsp, TernaryQubitNode* start_node, BranchType initial_branch) {
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
};

TreeOracle::TreeOracle(const TernaryTree& tree, const dvlab::APSPResult<QubitIdType>& apsp) {
    _n_modes = tree.num_qubits();
    if (_n_modes == 0) return;

    _parent.assign(_n_modes, 0);
    _depth.assign(_n_modes, 0);
    _edge_cost_up.assign(_n_modes, 0.0);
    _dfs_start.assign(_n_modes, 0);
    _tail_weight.assign(_n_modes, 0.0);

    // Run dfs from root to fill _parent, _depth, _edge_cost_up, _dfs_start
    auto root_node = dynamic_cast<TernaryQubitNode*>(tree.get_root());
    if (!root_node) throw std::runtime_error("Root is not a qubit node");
    int timer = 0;
    dfs(tree, apsp, timer, root_node->id, root_node->id, 0, 0.0);

    // Fill _tail_weight
    for (size_t u = 0; u < _n_modes; ++u) {
        auto u_node = dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(u));
        if (u_node && u_node->qubit_label.has_value()) {
            double x_cost   = calc_branch_tail(apsp, u_node, BranchType::left);
            double y_cost   = calc_branch_tail(apsp, u_node, BranchType::mid);
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

double TreeOracle::get_subtree_weight(const std::vector<size_t>& nodes) const {
    if (nodes.empty()) return 0.0;
    if (nodes.size() == 1) return 0.0;
    if (nodes.size() == 2) return get_tree_distance(nodes[0], nodes[1]);

    std::vector<size_t> sorted_nodes = nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), [this](size_t a, size_t b) {
        return _dfs_start[a] < _dfs_start[b];
    });

    // E.g. for nodes a, b, c, d in sorted order, calculate
    // dist(a,b) + dist(b,c) + dist(c,d) + dist(d,a)
    double total_weight = 0.0;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        size_t u = sorted_nodes[i];
        size_t v = sorted_nodes[(i + 1) % sorted_nodes.size()];
        total_weight += get_tree_distance(u, v);
    }

    // Divide by 2 to fix double counting
    return total_weight / 2.0;
}

// Best proxy: infidelity_cost. Approximates infidelity by multiplying infidelities of two-qubit
// gates along the interaction path in the ternary tree.
// TODO: check use of TreeOracle
double infidelity_cost(const TernaryTree& tree, const FermionHamiltonian& f_ham, const dvlab::APSPResult<QubitIdType>& apsp) {
    TreeOracle oracle(tree, apsp);
    double total_infidelity = 0.0;

    TernaryTreeMapping mapping(tree);
    QubitHamiltonian q_ham = qubitize(f_ham, mapping);

    for (const auto& term : q_ham) {
        std::vector<size_t> active_nodes;
        for (size_t i = 0; i < term.n_qubits(); ++i) {
            if (!term.is_i(i)) {
                active_nodes.push_back(i);
            }
        }
        
        // Note: sum because apsp is log fidelities
        total_infidelity += oracle.get_subtree_weight(active_nodes);
    }
    
    return total_infidelity;
}

// Approximates CNOT count across interaction paths in ternary tree
double cnot_cost(const TernaryTree& tree, const FermionHamiltonian& f_ham, const dvlab::APSPResult<QubitIdType>& apsp) {
    TreeOracle oracle(tree, apsp);
    double total_cost = 0.0;

    TernaryTreeMapping mapping(tree);
    QubitHamiltonian q_ham = qubitize(f_ham, mapping);

    for (const auto& term : q_ham) {
        std::vector<size_t> active_nodes;
        for (size_t i = 0; i < term.n_qubits(); ++i) {
            if (!term.is_i(i)) {
                active_nodes.push_back(i);
            }
        }
        
        total_cost += oracle.get_subtree_weight(active_nodes);
    }
    
    return total_cost;
}

// Counts number of Pauli gates along path in ternary tree.
double pauli_weight_cost(const TernaryTree& tt, const FermionHamiltonian& f_ham) {
    TernaryTreeMapping mapping(tt);
    QubitHamiltonian q_ham = qubitize(f_ham, mapping);

    double total_weight = 0;
    for (const auto& term : q_ham) {
        for (size_t i = 0; i < term.n_qubits(); ++i) {
            if (!term.is_i(i)) {
                total_weight += 1.0;
            }
        }
    }
    return total_weight;
}

// TODO: combine SA functions below
/**
 * @brief Optimizes a fermion-to-qubit mapping to minimize proxy infidelity cost.
 * @param initial_tree Initial mapping.
 * @param f_ham Fermionic Hamiltonian to be mapped.
 * @param device Hardware device (required for noise data).
 * @return Optimized TernaryTree mapping.
 */
TernaryTree infidelity_proxy_optimize_mapping(
    TernaryTree const& initial_tree,
    const FermionHamiltonian& f_ham,
    const qsyn::device::Device* device) {
    
    using util::SimulatedAnnealing;

    if (!device) {
        throw std::runtime_error("Device is required for infidelity proxy optimization");
    }

    // For noise-aware APSP
    auto noise_aware_cost = [](qsyn::device::Device::QubitPair const& edge, qsyn::device::Device const& dev) -> float {
        auto const& edge_gates = dev.get_gate_info(edge);
        auto const& gate_set = dev.get_gate_set();
        float error_rate = 1.0f;
        
        for (auto const& info : edge_gates) {
            std::string gate_name = dvlab::str::tolower_string(gate_set[info.gate_idx]);
            if (gate_name == "cx" || gate_name == "cnot" || gate_name == "ecr" || gate_name == "cz") {
                error_rate = info.error; 
                break; 
            }   
        }
        
        if (error_rate >= 1.0f || error_rate < 0.0f) {
            return std::numeric_limits<float>::infinity();
        }
        return -std::log(1.0f - error_rate); 
    };

    auto const fidelity_apsp = device::floyd_warshall(*device, noise_aware_cost);

    std::function<double(TernaryTree const&)> const wrapped_cost_fn =
        [&](TernaryTree const& tree) -> double { return infidelity_cost(tree, f_ham, fidelity_apsp); };
        
    TreeRotator rotator;

    using MutateFn = SimulatedAnnealing<TernaryTree, double>::MutateFn;

    auto const mutate_fns = std::vector<MutateFn>{
        [&](TernaryTree& tree) {
            rotator.cp_leaf_move(&tree, *device);
        },
        [&](TernaryTree& tree) {
            rotator.root_change(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.pauli_shuffle(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.mode_association_swap(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.majorana_braiding_change(&tree);
        },
    };

    auto const sa = SimulatedAnnealing<TernaryTree, double>(
        /* init_temp    = */ 20.0,
        /* cooling_rate = */ 0.99995,
        /* min_temp     = */ 0.3,
        /* cost_fn      = */ wrapped_cost_fn,
        /* mutate_fns   = */ mutate_fns);

    auto const [best_tree, best_cost] = sa(initial_tree);

    fmt::println("Final optimized infidelity cost: {} \n", best_cost);

    return best_tree;
}

/**
 * @brief Optimizes a fermion-to-qubit mapping to minimize proxy cnot count as
   computed by cnot_cost.
 * @param initial_tree Initital mapping.
 * @param f_ham Fermionic Hamiltonian to be mapped.
 * @param device Hardware device.
 * @return Optimized TernaryTree mapping.
 */
TernaryTree cnot_proxy_optimize_mapping(
    TernaryTree const& initial_tree,
    const FermionHamiltonian& f_ham,
    const qsyn::device::Device* device) {
    using util::SimulatedAnnealing;

    if (!device) {
        throw std::runtime_error("Device is required for cnot proxy optimization");
    }

    auto apsp = device::floyd_warshall(*device);
    auto const wrapped_cost_fn =
        [&](TernaryTree const& tree) { return cnot_cost(tree, f_ham, apsp); };
    TreeRotator rotator;

    using MutateFn = SimulatedAnnealing<TernaryTree, double>::MutateFn;

    auto const mutate_fns = std::vector<MutateFn>{
        [&](TernaryTree& tree) {
            rotator.cp_leaf_move(&tree, *device);
        },
        [&](TernaryTree& tree) {
            rotator.root_change(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.pauli_shuffle(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.mode_association_swap(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.majorana_braiding_change(&tree);
        },
    };

    auto const sa = SimulatedAnnealing<TernaryTree, double>(
        /* init_temp    = */ 20.0,
        /* cooling_rate = */ 0.99995,
        /* min_temp     = */ 0.3,
        /* cost_fn      = */ wrapped_cost_fn,
        /* mutate_fns   = */ mutate_fns);

    auto const [best_tree, best_cost] = sa(initial_tree);

    fmt::println("Final optimized 'CNOT' count: {} \n", best_cost);

    return best_tree;
}

/**
 * @brief Optimizes a fermion-to-qubit mapping to minimize Pauli weight.
 * @param initial_tree Initital mapping.
 * @param f_ham Fermionic Hamiltonian to be mapped.
 * @param device (Optional) Hardware device.
 * @return Optimized TernaryTree mapping.
 */
TernaryTree pauli_weight_optimize_mapping(
    TernaryTree const& initial_tree,
    const FermionHamiltonian& f_ham,
    const qsyn::device::Device* device) {
    using util::SimulatedAnnealing;
    auto const wrapped_cost_fn =
        [&](TernaryTree const& tree) { return pauli_weight_cost(tree, f_ham); };
    TreeRotator rotator;

    using MutateFn = SimulatedAnnealing<TernaryTree, double>::MutateFn;

    auto const mutate_fns = std::vector<MutateFn>{
        [&](TernaryTree& tree) {
            if (device) {
                rotator.cp_leaf_move(&tree, *device);
            } else {
                rotator.ncp_leaf_move(&tree);
            }
        },
        [&](TernaryTree& tree) {
            rotator.root_change(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.pauli_shuffle(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.mode_association_swap(&tree);
        },
        [&](TernaryTree& tree) {
            rotator.majorana_braiding_change(&tree);
        },
    };

    auto const sa = SimulatedAnnealing<TernaryTree, double>(
        /* init_temp    = */ 20.0,
        /* cooling_rate = */ 0.99995,
        /* min_temp     = */ 0.3,
        /* cost_fn      = */ wrapped_cost_fn,
        /* mutate_fns   = */ mutate_fns);

    auto const [best_tree, best_cost] = sa(initial_tree);

    fmt::println("Final optimized Pauli weight: {} \n", best_cost);

    return best_tree;
}

}  // namespace qsyn::hamiltonian
