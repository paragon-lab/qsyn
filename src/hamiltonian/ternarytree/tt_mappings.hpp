/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Ternary tree mappings ]
  Author       [ April Wang (april864) ]
*/

#pragma once

#include <unordered_map>
#include <vector>

#include "hamiltonian/qubit_hamiltonian.hpp"
#include "tableau/stabilizer_tableau.hpp"
#include "ternary_tree.hpp"

namespace qsyn::hamiltonian {

using Pauli = qsyn::tableau::Pauli;

struct FermionOps {
    std::vector<ComplexPauliTerm> creation;
    std::vector<ComplexPauliTerm> annihilation;
};

/**
 * @brief Mapper for Ternary Tree fermionic operators to qubit operators.
 * @param num_qubits The number of qubits in the Ternary Tree.
 */
class TTMapper {
public:
    TTMapper(size_t num_qubits)
        : _tree(TernaryTree(num_qubits)) {
        _basic_assign_qubits();
        _basic_load_pauli_strs();
        _pair_legs();
        _load_ferm_ops();
    }

    TTMapper(TernaryTree tree)
        : _tree(std::move(tree)) {
        _basic_load_pauli_strs();
        _pair_legs();
        _load_ferm_ops();
    }

    std::vector<ComplexPauliTerm> get_pauli_str(size_t mode, bool is_creation) const {
        return is_creation ? _mode_to_ferm_ops.at(mode).creation : _mode_to_ferm_ops.at(mode).annihilation;
    }

    [[nodiscard]] ComplexPauliTerm leg_pauli_product(TernaryLeg* leg_p, TernaryLeg* leg_q) const;

    TernaryTree const& tree() const { return _tree; }
    std::unordered_map<TernaryLeg*, std::vector<Pauli>> const& pauli_strs() const { return _pauli_strs; }
    std::unordered_map<size_t, std::pair<TernaryLeg*, TernaryLeg*>> const& leg_pairs() const {
        return _leg_pairs;
    }

private:
    TernaryTree _tree;
    std::unordered_map<TernaryLeg*, std::vector<Pauli>> _pauli_strs;
    std::unordered_map<size_t, std::pair<TernaryLeg*, TernaryLeg*>> _leg_pairs;
    std::unordered_map<size_t, FermionOps> _mode_to_ferm_ops;

    void _basic_assign_qubits();
    void _basic_load_pauli_strs();
    void _pair_legs();
    void _load_ferm_ops();
};

[[nodiscard]]
tableau::StabilizerTableau to_clifford(TTMapper const& mapper);

}  // namespace qsyn::hamiltonian
