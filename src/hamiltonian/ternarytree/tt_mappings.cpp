/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Ternary tree mappings ]
  Author       [ April Wang (april864) ]
*/

#include "tt_mappings.hpp"

#include <set>
#include <unordered_map>
#include <vector>

#include "hamiltonian/qubit_hamiltonian.hpp"
#include "tableau/pauli_product_trait.hpp"
#include "tableau/stabilizer_tableau.hpp"
#include "ternary_tree.hpp"

namespace qsyn::hamiltonian {

using Pauli = qsyn::tableau::Pauli;

namespace {
Pauli to_pauli(BranchType branch) {
    switch (branch) {
        case BranchType::left:
            return Pauli::x;
        case BranchType::mid:
            return Pauli::y;
        case BranchType::right:
            return Pauli::z;
    }
    DVLAB_UNREACHABLE("Every branch type should be handled in the switch-case");
}
}  // namespace

/**
 * @brief Basic method of assigning qubits to nodes.
 *        Assigns qubit i to node i.
 */
void TTMapper::_basic_assign_qubits() {
    for (size_t i = 0; i < _tree.num_qubits(); i++) {
        _tree.assign_qubit(i, i);
    }
}

/**
 * @brief Basic method of getting fermionic operators from Bonsai paper.
 */
void TTMapper::_basic_load_pauli_strs() {
    for (TernaryLeg* leg : _tree.get_legs()) {
        std::vector<Pauli> pauli_str(_tree.num_qubits(), Pauli::i);

        TernaryNode* curr = leg;
        while (curr->incoming_edge) {
            TernaryEdge* edge   = curr->incoming_edge;
            TernaryNode* parent = edge->source;

            auto const parent_qubit_node = dynamic_cast<TernaryQubitNode*>(parent);
            assert(parent_qubit_node);

            auto const parent_id = parent_qubit_node->id;

            assert(parent_id < _tree.num_qubits());

            pauli_str[parent_id] = to_pauli(edge->branch);

            curr = parent;
        }

        _pauli_strs.emplace(leg, std::move(pauli_str));
    }
}

void TTMapper::_pair_legs() {
    for (size_t i = 0; i < _tree.num_qubits(); i++) {
        TernaryNode* curr = _tree.get_node_by_index(i);

        TernaryNode* left_path = curr->get_left_child();
        while (!left_path->is_leg()) {
            left_path = left_path->get_right_child();
        }

        TernaryNode* right_path = curr->get_mid_child();
        while (!right_path->is_leg()) {
            right_path = right_path->get_right_child();
        }

        assert(left_path && left_path->is_leg());
        assert(right_path && right_path->is_leg());

        auto left_leg  = dynamic_cast<TernaryLeg*>(left_path);
        auto right_leg = dynamic_cast<TernaryLeg*>(right_path);
        DVLAB_ASSERT(left_leg, "left_path is not a leg");
        DVLAB_ASSERT(right_path, "right_path is not a leg");
        _leg_pairs.emplace(i, std::pair{left_leg, right_leg});
    }
}

/**
 * @brief Pauli-string product P * Q for the strings mapped from two legs.
 *
 * Paired legs in the ternary-tree mapping anticommute, so the product is
 * ±i times a Pauli string (not a real ±1). The phase uses the same rules
 * as ComplexPauliTerm multiplication (e.g. for exp(-π/4 · P Q)).
 */
ComplexPauliTerm TTMapper::leg_pauli_product(TernaryLeg* leg_p, TernaryLeg* leg_q) const {
    auto const& p_str = _pauli_strs.at(leg_p);
    auto const& q_str = _pauli_strs.at(leg_q);
    return ComplexPauliTerm(p_str, std::complex<double>(1.0, 0.0)) *
           ComplexPauliTerm(q_str, std::complex<double>(1.0, 0.0));
}

void TTMapper::_load_ferm_ops() {
    for (size_t i = 0; i < _tree.num_qubits(); ++i) {
        auto left_str  = _pauli_strs.at(_leg_pairs.at(i).first);
        auto right_str = _pauli_strs.at(_leg_pairs.at(i).second);

        auto* qubit_node = dynamic_cast<TernaryQubitNode*>(_tree.get_node_by_index(i));
        assert(qubit_node);

        FermionOps ops;
        if (!qubit_node->is_braided) {
            // Bonsai: a_j^ = 0.5 Sx - 0.5i Sy
            // Treespilation (USE THIS ONE): a_j^ = 0.5 Sx + 0.5i Sy
            ops.creation = {ComplexPauliTerm(left_str, std::complex<double>(0.5, 0)),
                            ComplexPauliTerm(right_str, std::complex<double>(0, 0.5))};
            // Bonsai: a_j = 0.5 Sx + 0.5i Sy
            // Treespilation (USE THIS ONE): a_j = 0.5 Sx - 0.5i Sy
            ops.annihilation = {ComplexPauliTerm(left_str, std::complex<double>(0.5, 0)),
                                ComplexPauliTerm(right_str, std::complex<double>(0, -0.5))};
        } else {
            // a_j^ = -0.5 Sy + 0.5i Sx
            ops.creation = {ComplexPauliTerm(left_str, std::complex<double>(0, 0.5)),
                            ComplexPauliTerm(right_str, std::complex<double>(-0.5, 0))};
            // a_j = -0.5 Sy - 0.5i Sx
            ops.annihilation = {ComplexPauliTerm(left_str, std::complex<double>(0, -0.5)),
                                ComplexPauliTerm(right_str, std::complex<double>(-0.5, 0))};
        }

        _mode_to_ferm_ops.emplace(i, std::move(ops));
    }
}

namespace {

TernaryQubitNode* as_qubit_node(TernaryNode* node) {
    return dynamic_cast<TernaryQubitNode*>(node);
}

TernaryQubitNode* qubit_child(TernaryNode* parent, BranchType branch) {
    auto* child = parent->get_child(branch);
    if (!child || child->is_leg()) {
        return nullptr;
    }
    return as_qubit_node(child);
}

TernaryNode* get_node_by_index(TernaryTree const& tree, size_t index) {
    return tree.get_node_by_index(index);
}

void debraid_and_append_gates(
    TTMapper const& mapper,
    size_t mode,
    std::pair<TernaryLeg*, TernaryLeg*> const& leg_pair,
    std::vector<tableau::CliffordOperator>& clifford_circuit_adjoint) {
    auto* qubit_node = as_qubit_node(get_node_by_index(mapper.tree(), mode));
    if (!qubit_node->is_braided) {
        return;
    }

    // Recall that for mode $i$,
    // not braided: $a_i =  0.5 S_{2i}   + 0.5i S_{2i+1}$
    //     braided: $a_i = -0.5 S_{2i+1} + 0.5i S_{2i}$
    // To debraid, it suffices to apply
    // $$U_{2i, 2i+1} = \exp(-pi/4 S_{2i} S_{2i+1})$$
    // to each braided leg pair.

    auto [left_leg, right_leg] = leg_pair;
    auto pauli_product         = mapper.leg_pauli_product(left_leg, right_leg);

    constexpr std::complex<double> plus_i(0, 1);
    constexpr std::complex<double> minus_i(0, -1);

    DVLAB_ASSERT(pauli_product.coeff() == plus_i || pauli_product.coeff() == minus_i,
                 "Coefficient must be ±i");

    auto const phase_gate = pauli_product.coeff() == plus_i
                                ? tableau::CliffordOperatorType::s
                                : tableau::CliffordOperatorType::sdg;

    // REVIEW: maybe make extract_clifford_operators work on PauliProduct directly?
    auto const pauli_rotation = tableau::PauliRotation(
        pauli_product.pauli_product(),
        dvlab::Phase(1));  // this phase is just a placeholder

    auto [ops, qubit] = tableau::extract_clifford_operators(pauli_rotation);

    clifford_circuit_adjoint.insert(clifford_circuit_adjoint.end(), ops.begin(), ops.end());
    clifford_circuit_adjoint.push_back({phase_gate, {qubit, 0}});
    tableau::adjoint_inplace(ops);
    clifford_circuit_adjoint.insert(clifford_circuit_adjoint.end(), ops.begin(), ops.end());
}

std::vector<size_t> pre_order_qubit_indices(TernaryTree const& tree) {
    std::vector<size_t> traversal;
    std::vector<TernaryQubitNode*> stack;
    stack.push_back(as_qubit_node(tree.get_root()));
    while (!stack.empty()) {
        auto const v = stack.back();
        stack.pop_back();
        traversal.push_back(v->id);
        for (auto const& branch : {BranchType::left, BranchType::mid, BranchType::right}) {
            auto const child = v->get_child(branch);
            if (child && !child->is_leg()) {
                stack.push_back(as_qubit_node(child));
            }
        }
    }
    return traversal;
}

std::vector<size_t> post_order_qubit_indices(TernaryTree const& tree) {
    auto traversal = pre_order_qubit_indices(tree);
    std::ranges::reverse(traversal);
    return traversal;
}

void append_cx_from_descendants(
    std::set<size_t> const& control_indices,
    size_t target_idx,
    std::vector<tableau::CliffordOperator>& clifford_circuit_adjoint) {
    for (auto const ctrl_idx : control_indices) {
        clifford_circuit_adjoint.push_back(
            {tableau::CliffordOperatorType::cx, {ctrl_idx, target_idx}});
    }
}

/**
 * @brief Inorder traversal of the qubit indices in the ternary tree.
   The left (X) branch is visited before the node itself, then the Y and Z branches.
 * @param tree Ternary tree
 * @return Vector of qubit indices in inorder traversal
 */
std::vector<size_t> inorder_qubit_indices(TernaryTree const& tree) {
    std::vector<size_t> traversal;
    // phase 0: left subtree; 1: visit node; 2: right subtree (after mid)
    std::vector<std::pair<TernaryQubitNode*, uint8_t>> stack;
    stack.emplace_back(as_qubit_node(tree.get_root()), 0);

    while (!stack.empty()) {
        auto [node, phase] = stack.back();
        stack.pop_back();

        if (phase == 0) {
            stack.emplace_back(node, 1);
            if (auto* left = qubit_child(node, BranchType::left)) {
                stack.emplace_back(left, 0);
            }
        } else if (phase == 1) {
            traversal.push_back(node->id);
            stack.emplace_back(node, 2);
            if (auto* mid = qubit_child(node, BranchType::mid)) {
                stack.emplace_back(mid, 0);
            }
        } else {
            if (auto* right = qubit_child(node, BranchType::right)) {
                stack.emplace_back(right, 0);
            }
        }
    }
    return traversal;
}

/**
 * @brief Inverse mapping of the qubit indices. For example, if the mapping is
   [4, 1, 0, 2, 3], then the inverse mapping is [2, 1, 3, 4, 0].
 * @param mapping Mapping of the qubit indices. Should be a permutation of [0, 1, 2, ..., n-1].
 * @return Inverse mapping of the qubit indices
 */
std::vector<size_t> inverse_permutation(std::vector<size_t> const& mapping) {
    std::vector<size_t> inverse(mapping.size(), 0);
    for (size_t i = 0; i < mapping.size(); ++i) {
        inverse[mapping[i]] = i;
    }
    return inverse;
}

std::vector<std::vector<size_t>> get_descendants(TernaryTree const& tree) {
    std::vector<std::vector<size_t>> descendants(tree.num_qubits());

    for (auto const& idx : post_order_qubit_indices(tree)) {
        auto* qubit_node  = as_qubit_node(get_node_by_index(tree, idx));
        auto* left_child  = qubit_child(qubit_node, BranchType::left);
        auto* mid_child   = qubit_child(qubit_node, BranchType::mid);
        auto* right_child = qubit_child(qubit_node, BranchType::right);

        descendants[idx].push_back(idx);
        auto append_child_descendants = [&](TernaryQubitNode* child) {
            auto const& child_desc = descendants[child->id];
            descendants[idx].insert(
                descendants[idx].end(),
                child_desc.begin(),
                child_desc.end());
        };
        if (left_child) {
            append_child_descendants(left_child);
        }
        if (mid_child) {
            append_child_descendants(mid_child);
        }
        if (right_child) {
            append_child_descendants(right_child);
        }
    }

    return descendants;
}

/**
 * @brief Get the depth of the Y branch for each qubit. That is, how many
 * Y-edges are on the path from the root to a node
 * @param tree Ternary tree
 * @return Vector of depths
 */
std::vector<size_t> get_y_branch_depths(TernaryTree const& tree) {
    // indexed by tree node index
    std::vector<size_t> depths(tree.num_qubits(), 0);

    std::vector<std::pair<TernaryQubitNode*, size_t>> stack;
    stack.emplace_back(as_qubit_node(tree.get_root()), 0);
    while (!stack.empty()) {
        auto [node, y_depth] = stack.back();
        stack.pop_back();
        depths[node->id] = y_depth;
        for (auto const branch : {BranchType::left, BranchType::mid, BranchType::right}) {
            if (auto* child = qubit_child(node, branch)) {
                stack.emplace_back(child, y_depth + (branch == BranchType::mid));
            }
        }
    }

    return depths;
}
}  // namespace

/**
 * @brief Convert a ternary-tree mapping to a Clifford tableau.
 *        The Clifford tableau represents an operator C such that
 *        \Phi_{T} = C \Phi_{\text{JW}} represents the F2Q mapping
 *        associated with the ternary tree.
 * @param mapper Ternary-tree mapper with leg pairing and Pauli strings loaded.
 * @return Stabilizer tableau
 */
tableau::StabilizerTableau to_clifford(TTMapper const& mapper) {
    using CliffordOperator     = tableau::CliffordOperator;
    using CliffordOperatorType = tableau::CliffordOperatorType;
    using dvlab::iterator::next;

    auto permutation = inorder_qubit_indices(mapper.tree());

    auto const inorder_indices_inv = inverse_permutation(permutation);
    auto const braiding_counts     = get_y_branch_depths(mapper.tree());
    auto const descendants         = get_descendants(mapper.tree());

    auto tableau = tableau::StabilizerTableau(mapper.tree().num_qubits());

    auto reverse_ranges = std::vector<std::pair<size_t, size_t>>();

    for (auto const& idx : pre_order_qubit_indices(mapper.tree())) {
        auto const qubit_node = as_qubit_node(get_node_by_index(mapper.tree(), idx));

        auto const x_child = as_qubit_node(qubit_node->get_child(BranchType::left));
        auto const y_child = as_qubit_node(qubit_node->get_child(BranchType::mid));

        // Record the permutation that should happen to the Y-descendants.
        auto const y_descendant_idx_begin =
            inorder_indices_inv[idx] + 1;
        auto const y_descendant_idx_end =
            y_descendant_idx_begin +
            (y_child ? descendants[y_child->id].size() : 0);
        reverse_ranges.emplace_back(y_descendant_idx_begin, y_descendant_idx_end);

        // NOTE: the is_braided flag removes one S gate.
        // We writes +3 instead of -1 to avoid footguns if we ever decide
        // to change the type of braiding_counts to a signed integer.
        auto const n_braidings = (braiding_counts[idx] + (qubit_node->is_braided ? 3 : 0)) % 4;

        switch (n_braidings) {
            case 0:
                break;
            case 1:
                tableau.s(idx);
                break;
            case 2:
                tableau.z(idx);
                break;
            case 3:
                tableau.sdg(idx);
                break;
            default:
                DVLAB_UNREACHABLE("Invalid number of braidings");
        }

        if (x_child) {
            for (auto const x_descendant : descendants[x_child->id]) {
                tableau.cx(x_descendant, idx);
            }
        }

        if (y_child) {
            for (auto const y_descendant : descendants[y_child->id]) {
                tableau.cx(y_descendant, idx);
            }
        }
    }

    // Each pre-order node records an inorder-index range whose Y-descendant
    // block must be reversed. Apply those reversals now, in reverse pre-order,
    // so inner (deeper) ranges are handled before outer ones.
    //
    // The ranges nest along Y-edges, and the reversals do not commute.
    // For example, consider a Y-chain 0 → 1 → 2 → 3. The edge 0 → 1 reverses
    // inorder positions between nodes 1 and 3, while the edge 1 → 2 reverses
    // inorder positions between nodes 2 and 3. Composed, the result is
    // [0, 2, 3, 1]. Reversing the ranges inner-to-outer reproduces that
    // composition without needing to update inorder_indices_inv during the
    // tree-gate loop above.
    for (auto const& [begin, end] : reverse_ranges | std::views::reverse) {
        std::reverse(next(permutation.begin(), begin), next(permutation.begin(), end));
    }

    // prepend permutation CZ gates to the tree gates
    for (size_t i = 0; i < permutation.size(); ++i) {
        for (size_t j = i + 1; j < permutation.size(); ++j) {
            if (permutation[i] > permutation[j]) {
                tableau.prepend_cz(permutation[i], permutation[j]);
            }
        }
    }

    return tableau;
}

}  // namespace qsyn::hamiltonian
