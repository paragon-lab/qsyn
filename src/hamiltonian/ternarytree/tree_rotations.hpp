/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Ternary tree rotations ]
  Author       [ April Wang (april864)]
*/

#pragma once

#include <random>

#include "device/device.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "ternary_tree.hpp"

namespace qsyn::hamiltonian {

class TreeRotator {
public:
    /// Construct with a nondeterministic seed from ``std::random_device``.
    TreeRotator() : _rng(std::random_device{}()) {}

    /// Construct with a fixed RNG seed (for reproducible SA / mutations).
    explicit TreeRotator(std::uint32_t seed) : _rng(seed) {}

    /// Construct with an existing engine (copied).
    explicit TreeRotator(std::mt19937 rng) : _rng(std::move(rng)) {}

    void seed(std::uint32_t seed) { _rng.seed(seed); }

    // For non-connectivity preserving (NCP), choose a random terminal node-v with
    // three legs and attach it to the free leg of another node-w that is neither
    // v nor its parent. For CP random leaf move, choose random move that preserves
    // connectivity. For CP leaf move, choose moves more likely to improve error.
    void ncp_leaf_move(TernaryTree* tt);
    void cp_random_leaf_move(TernaryTree* tt, qsyn::device::Device const& device);
    void cp_leaf_move(TernaryTree* tt, qsyn::device::Device const& device);
    // A node v different from the root with out-degree at most 2 is chosen
    // as a new root. The path from root to v is identified, and the tree is
    // updated so that child and parent designations are swapped along the path.
    void root_change(TernaryTree* tt);
    // A node with an out-degree of at least 1 is chosen, and the Pauli operators
    // associated with the links are changed.
    void pauli_shuffle(TernaryTree* tt);
    // For nodes with labels (i, u, b) and (i', u', b') the labels are changed to
    // (i', u, b') and (i, u', b) respectively.
    void mode_association_swap(TernaryTree* tt);
    // For a node with label (i, u, b), the braiding b is changed to the
    // opposite one, i.e., '+' is changed to '-' and vice versa.
    void majorana_braiding_change(TernaryTree* tt);

private:
    std::mt19937 _rng;
};

}  // namespace qsyn::hamiltonian
