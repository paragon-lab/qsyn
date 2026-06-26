/*
  Shared fixtures for TTMapper / to_clifford unit tests.
*/

#pragma once

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "hamiltonian/f2q_majorana.hpp"
#include "tableau/stabilizer_tableau.hpp"

namespace qsyn::hamiltonian::tt_mappings_test {

using qsyn::tableau::StabilizerTableau;

// --- Chain trees ---

inline TernaryTree make_jw_chain_tree(size_t num_qubits) {
    TernaryTree tree;
    TernaryNode* curr = tree.get_root();
    for (size_t i = 1; i < num_qubits; ++i) {
        tree.add_qubit_node(curr, BranchType::right);
        curr = curr->get_right_child();
    }
    tree.append_legs_to_tree();
    for (size_t i = 0; i < num_qubits; ++i) {
        tree.assign_qubit(i, i);
    }
    return tree;
}

inline TernaryTree make_parity_chain_tree(size_t num_qubits) {
    TernaryTree tree;
    TernaryNode* curr = tree.get_root();
    for (size_t i = 1; i < num_qubits; ++i) {
        tree.add_qubit_node(curr, BranchType::left);
        curr = curr->get_left_child();
    }
    tree.append_legs_to_tree();
    for (size_t i = 0; i < num_qubits / 2; ++i) {
        tree.swap_indices(i, num_qubits - 1 - i);
    }
    for (size_t i = 0; i < num_qubits; ++i) {
        tree.assign_qubit(i, i);
    }
    return tree;
}

// 4 -> (1, 7, 8); 1 -> (0, 2, 3); 7 -> (5, leg, leg); 5 -> (leg, leg, 6)
inline TernaryTree make_branching_example_tree() {
    TernaryTree tree;
    auto* root = dynamic_cast<TernaryQubitNode*>(tree.get_root());
    std::vector<TernaryQubitNode*> nodes;
    nodes.push_back(root);

    size_t const i1 = tree.add_qubit_node(root, BranchType::left);
    size_t const i7 = tree.add_qubit_node(root, BranchType::mid);
    size_t const i8 = tree.add_qubit_node(root, BranchType::right);
    nodes.push_back(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(i1)));
    nodes.push_back(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(i7)));
    nodes.push_back(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(i8)));

    auto* n1        = nodes[1];
    size_t const i0 = tree.add_qubit_node(n1, BranchType::left);
    size_t const i2 = tree.add_qubit_node(n1, BranchType::mid);
    size_t const i3 = tree.add_qubit_node(n1, BranchType::right);
    nodes.push_back(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(i0)));
    nodes.push_back(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(i2)));
    nodes.push_back(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(i3)));

    auto* n7        = nodes[2];
    size_t const i5 = tree.add_qubit_node(n7, BranchType::left);
    nodes.push_back(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(i5)));

    auto* n5        = nodes[7];
    size_t const i6 = tree.add_qubit_node(n5, BranchType::right);
    nodes.push_back(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(i6)));

    tree.append_legs_to_tree();

    std::array<size_t, 9> const slot_of_label = {4, 1, 5, 6, 0, 7, 8, 2, 3};
    tree.remap_node_ids(slot_of_label);
    for (size_t label = 0; label < slot_of_label.size(); ++label) {
        tree.assign_qubit(label, label);
    }
    return tree;
}

// --- Braiding helpers ---

inline void set_braided(TernaryTree& tree, std::vector<size_t> const& node_indices) {
    for (size_t const index : node_indices) {
        auto* qnode = dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(index));
        REQUIRE(qnode != nullptr);
        qnode->is_braided = true;
    }
}

// exp(-i pi/4 Z_j) from +i Z_j leg pair (JW / parity mode 0).
inline void apply_single_z_debraid(StabilizerTableau& clifford, size_t qubit) {
    clifford.sdg(qubit);
}

// exp(-i pi/4 Z_{j-1} Z_j) from +i Z_{j-1} Z_j leg pair (parity mode j > 0).
inline void apply_zz_debraid(StabilizerTableau& clifford, size_t j) {
    clifford.cx(j - 1, j).sdg(j).cx(j - 1, j);
}

// --- Expected Cliffords ---

inline StabilizerTableau expected_parity_clifford(size_t num_qubits) {
    StabilizerTableau clifford(num_qubits);
    for (size_t i = num_qubits; i-- > 0;) {
        for (size_t j = num_qubits; j-- > i + 1;) {
            clifford.cx(i, j);
        }
    }
    return clifford;
}

inline StabilizerTableau expected_debraid_clifford(
    size_t num_qubits, std::vector<size_t> const& braided_modes) {
    StabilizerTableau clifford(num_qubits);
    for (size_t const mode : braided_modes) {
        apply_single_z_debraid(clifford, mode);
    }
    return clifford;
}

inline StabilizerTableau expected_parity_with_braiding(
    size_t num_qubits, std::vector<size_t> const& braided_modes) {
    StabilizerTableau clifford = expected_parity_clifford(num_qubits);
    for (size_t const mode : braided_modes) {
        if (mode == 0) {
            apply_single_z_debraid(clifford, 0);
        } else {
            apply_zz_debraid(clifford, mode);
        }
    }
    return clifford;
}

inline StabilizerTableau expected_branching_example_clifford() {
    constexpr size_t num_qubits = 9;
    StabilizerTableau clifford(num_qubits);
    constexpr std::array<std::pair<size_t, size_t>, 11> cx_gates = {{
        {0, 4},
        {1, 4},
        {2, 4},
        {3, 4},
        {5, 4},
        {6, 4},
        {7, 4},
        {2, 1},
        {0, 1},
        {6, 7},
        {5, 7},
    }};
    clifford.cz(5, 6).cz(5, 7).cz(6, 7);
    clifford.s(2).s(5).s(6).s(7);
    for (auto const& [ctrl, targ] : cx_gates) {
        clifford.cx(ctrl, targ);
    }
    return clifford;
}

inline StabilizerTableau expected_branching_braided_4_7_clifford() {
    constexpr size_t num_qubits = 9;
    StabilizerTableau clifford(num_qubits);
    constexpr std::array<std::pair<size_t, size_t>, 11> cx_gates = {{
        {0, 4},
        {1, 4},
        {2, 4},
        {3, 4},
        {5, 4},
        {6, 4},
        {7, 4},
        {2, 1},
        {0, 1},
        {6, 7},
        {5, 7},
    }};
    clifford.cz(5, 6).cz(5, 7).cz(6, 7);
    clifford.s(2).s(5).s(6).s(7);
    clifford.sdg(4).sdg(7);
    for (auto const& [ctrl, targ] : cx_gates) {
        clifford.cx(ctrl, targ);
    }
    return clifford;
}

inline TernaryTree make_noncanonical_qubit_indexing_tree() {
    // 0 -> (1, 2, 3); 1 -> (4, leaf, leaf)
    TernaryTree tree;
    auto* root = dynamic_cast<TernaryQubitNode*>(tree.get_root());

    tree.add_qubit_node(root, BranchType::left);
    tree.add_qubit_node(root, BranchType::mid);
    tree.add_qubit_node(root, BranchType::right);

    auto* n1 = dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(1));
    tree.add_qubit_node(n1, BranchType::left);

    tree.append_legs_to_tree();

    for (size_t label = 0; label < 5; ++label) {
        tree.assign_qubit(label, label);
    }
    return tree;
}

inline StabilizerTableau expected_noncanonical_qubit_indexing_clifford() {
    StabilizerTableau clifford(5);

    clifford.cz(0, 1).cz(0, 4).cz(1, 4).cz(2, 4).cz(3, 4);
    // inorder qubit indices: (4, 1, 0, 2, 3)
    clifford.s(2);
    clifford.cx(1, 0).cx(4, 0).cx(2, 0);
    clifford.cx(4, 1);

    return clifford;
}

inline TernaryTree y_chain_tree(size_t num_qubits) {
    // 0 --> 1 --> 2 --> ... --> num_qubits - 1
    // along the Y-branch only
    TernaryTree tree;
    auto* root = dynamic_cast<TernaryQubitNode*>(tree.get_root());

    TernaryNode* curr = root;
    for (size_t _ = 1; _ < num_qubits; ++_) {
        tree.add_qubit_node(curr, BranchType::mid);
        curr = curr->get_mid_child();
    }

    tree.append_legs_to_tree();

    for (size_t label = 0; label < num_qubits; ++label) {
        tree.assign_qubit(label, label);
    }

    return tree;
}

inline StabilizerTableau expected_y_chain_clifford(size_t num_qubits) {
    StabilizerTableau clifford(num_qubits);
    for (size_t targ = 0; targ < num_qubits; ++targ) {
        // CZ gates between targ+1 and targ+2, ..., num_qubits-1
        for (size_t i = targ + 1; i < num_qubits; ++i) {
            for (size_t j = i + 1; j < num_qubits; ++j) {
                clifford.cz(i, j);
            }
        }

        // S gates on targ+1, targ+2, ..., num_qubits-1
        for (size_t i = targ + 1; i < num_qubits; ++i) {
            clifford.s(i);
        }

        // CX gates between targ and targ+1, targ+1 and targ+2, ..., num_qubits-1 and targ
        for (size_t i = targ + 1; i < num_qubits; ++i) {
            clifford.cx(i, targ);
        }
    }
    return clifford;
}

// --- Assertions / diagnostics ---

inline void require_iz_pauli_product(
    ComplexPauliTerm const& product,
    std::vector<size_t> const& z_qubits) {
    REQUIRE(product.coeff() == std::complex<double>(0, 1));
    for (size_t const q : z_qubits) {
        REQUIRE(product.get_pauli_type(q) == qsyn::tableau::Pauli::z);
    }
    for (size_t q = 0; q < product.n_qubits(); ++q) {
        if (std::ranges::find(z_qubits, q) == z_qubits.end()) {
            REQUIRE(product.is_i(q));
        }
    }
}

using PauliExpansion = std::unordered_map<qsyn::tableau::PauliProduct, std::complex<double>>;

inline std::vector<ComplexPauliTerm> multiply_expansions(
    std::vector<ComplexPauliTerm> const& lhs,
    std::vector<ComplexPauliTerm> const& rhs) {
    std::vector<ComplexPauliTerm> result;
    for (auto const& l : lhs) {
        for (auto const& r : rhs) {
            result.emplace_back(l * r);
        }
    }
    return result;
}

inline PauliExpansion to_pauli_expansion(std::vector<ComplexPauliTerm> const& terms) {
    PauliExpansion expansion;
    for (auto const& term : terms) {
        expansion[term.pauli_product()] += term.coeff();
    }
    return expansion;
}

inline std::vector<ComplexPauliTerm> from_pauli_expansion(PauliExpansion const& expansion) {
    std::vector<ComplexPauliTerm> terms;
    for (auto const& [pauli, coeff] : expansion) {
        if (std::abs(coeff) > 0) {
            terms.emplace_back(pauli, coeff);
        }
    }
    return terms;
}

enum class CliffordConjugateDirection : std::uint8_t {
    forward,  // P -> C P C†  (apply extract_clifford_operators(C) to P)
    adjoint,  // P -> C† P C  (apply adjoint circuit to P)
};

inline ComplexPauliTerm conjugate_pauli_term(
    ComplexPauliTerm const& term,
    StabilizerTableau const& clifford,
    CliffordConjugateDirection direction) {
    if (direction == CliffordConjugateDirection::forward) {
        return qsyn::hamiltonian::conjugate_pauli_term(term, clifford);
    }
    auto ops = qsyn::tableau::extract_clifford_operators(clifford);
    qsyn::tableau::adjoint_inplace(ops);
    auto pauli = term.pauli_product();
    pauli.apply(ops);
    return ComplexPauliTerm(pauli, term.coeff());
}

inline PauliExpansion conjugate_pauli_expansion(
    std::vector<ComplexPauliTerm> const& terms,
    StabilizerTableau const& clifford,
    CliffordConjugateDirection direction) {
    PauliExpansion conjugated;
    for (auto const& term : terms) {
        auto const mapped = conjugate_pauli_term(term, clifford, direction);
        conjugated[mapped.pauli_product()] += mapped.coeff();
    }
    return conjugated;
}

inline bool pauli_expansions_equal(
    PauliExpansion const& lhs,
    PauliExpansion const& rhs,
    double tolerance = 1e-10) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    return std::ranges::all_of(lhs, [&](auto const& item) {
        auto const& [pauli, coeff] = item;

        auto it = rhs.find(pauli);
        if (it == rhs.end()) {
            return false;
        }
        return std::abs(coeff - it->second) <= tolerance;
    });
}

inline CliffordConjugateDirection conjugation_direction_matching_tt(
    std::vector<ComplexPauliTerm> const& jw_terms,
    std::vector<ComplexPauliTerm> const& tt_terms,
    StabilizerTableau const& clifford) {
    auto const expected = to_pauli_expansion(tt_terms);
    if (pauli_expansions_equal(
            conjugate_pauli_expansion(jw_terms, clifford, CliffordConjugateDirection::forward),
            expected)) {
        return CliffordConjugateDirection::forward;
    }
    if (pauli_expansions_equal(
            conjugate_pauli_expansion(jw_terms, clifford, CliffordConjugateDirection::adjoint),
            expected)) {
        return CliffordConjugateDirection::adjoint;
    }
    return CliffordConjugateDirection::forward;
}

inline std::string format_pauli_expansion(PauliExpansion const& expansion) {
    std::string out;
    for (auto const& [pauli, coeff] : expansion) {
        out += fmt::format("{} * {}\n", coeff, pauli.to_string());
    }
    if (out.empty()) {
        out = "(empty)\n";
    }
    return out;
}

inline std::optional<CliffordConjugateDirection> matching_conjugation_direction(
    std::vector<ComplexPauliTerm> const& jw_terms,
    std::vector<ComplexPauliTerm> const& tt_terms,
    StabilizerTableau const& clifford) {
    auto const expected = to_pauli_expansion(tt_terms);
    for (auto const direction :
         {CliffordConjugateDirection::forward, CliffordConjugateDirection::adjoint}) {
        if (pauli_expansions_equal(
                conjugate_pauli_expansion(jw_terms, clifford, direction),
                expected)) {
            return direction;
        }
    }
    return std::nullopt;
}

inline void require_jw_conjugated_matches_tt(
    std::vector<ComplexPauliTerm> const& jw_terms,
    std::vector<ComplexPauliTerm> const& tt_terms,
    StabilizerTableau const& clifford,
    CliffordConjugateDirection direction) {
    auto const expected = to_pauli_expansion(tt_terms);
    auto const conjugated =
        conjugate_pauli_expansion(jw_terms, clifford, direction);
    if (!pauli_expansions_equal(conjugated, expected)) {
        INFO("Expected (TT):\n"
             << format_pauli_expansion(expected));
        INFO("Got (C conjugated JW):\n"
             << format_pauli_expansion(conjugated));
    }
    REQUIRE(pauli_expansions_equal(conjugated, expected));
}

// JW Pauli monomial Z_0 ... Z_{p-1} P_p (Majorana / JW string on mode p).
inline std::vector<qsyn::tableau::Pauli> jw_pauli_string(
    size_t n_qubits,
    size_t mode,
    qsyn::tableau::Pauli pauli_on_mode) {
    std::vector<qsyn::tableau::Pauli> paulis(n_qubits, qsyn::tableau::Pauli::i);
    for (size_t q = 0; q < mode; ++q) {
        paulis[q] = qsyn::tableau::Pauli::z;
    }
    paulis[mode] = pauli_on_mode;
    return paulis;
}

inline ComplexPauliTerm jw_pauli_monomial(
    size_t n_qubits,
    size_t mode,
    qsyn::tableau::Pauli pauli_on_mode,
    std::complex<double> coeff = {1.0, 0.0}) {
    return ComplexPauliTerm(jw_pauli_string(n_qubits, mode, pauli_on_mode), coeff);
}

inline std::string format_synthesized_clifford_circuit(StabilizerTableau const& clifford) {
    using qsyn::tableau::CliffordOperatorType;
    using qsyn::tableau::to_string;
    std::string out;
    for (auto const& [type, qubits] : qsyn::tableau::extract_clifford_operators(clifford)) {
        out += to_string(type);
        switch (type) {
            case CliffordOperatorType::cx:
            case CliffordOperatorType::cz:
            case CliffordOperatorType::swap:
            case CliffordOperatorType::ecr:
                out += " " + std::to_string(qubits[0]) + " " + std::to_string(qubits[1]);
                break;
            default:
                out += " " + std::to_string(qubits[0]);
                break;
        }
        out += '\n';
    }
    if (out.empty()) {
        out = "(no Clifford gates synthesized)\n";
    }
    return out;
}

}  // namespace qsyn::hamiltonian::tt_mappings_test
