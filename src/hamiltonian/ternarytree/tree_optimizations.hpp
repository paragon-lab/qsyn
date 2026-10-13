/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Proxy functions and simulated annealing to optimize ternary tree ]
  Author       [ April Wang (april864) ]
*/

#pragma once

#include <cstdint>
#include <optional>

#include "device/device.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "ternary_tree.hpp"
#include "util/graph/floyd_warshall.hpp"

namespace qsyn::hamiltonian {

// Oracle for APSP-based path costs. Computes tree distances and tail weights.
class TreeOracle {
public:
    TreeOracle(TernaryTree const& tree, dvlab::APSPResult<QubitIdType> const& apsp);

    double get_tree_distance(size_t u, size_t v) const;
    double get_tail_weight(size_t u) const { return _tail_weight[u]; }
    double get_subtree_weight(std::vector<size_t> const& nodes) const;

private:
    size_t _n_modes;
    std::vector<size_t> _parent;
    std::vector<int> _depth;
    std::vector<double> _edge_cost_up;
    std::vector<int> _dfs_start;
    std::vector<double> _tail_weight;

    void _dfs(TernaryTree const& tree, dvlab::APSPResult<QubitIdType> const& apsp, int& timer,
              size_t v, size_t p, int current_depth, double cost_from_parent);

    double _calc_branch_tail(dvlab::APSPResult<QubitIdType> const& apsp, TernaryQubitNode* start_node,
                             BranchType initial_branch);
};

/**
 * Polymorphic cost model for ternary-tree encodings.
 * Subclass this to plug a new objective into ``optimize_ternary_tree_mapping``.
 */
class TernaryTreeCost {
public:
    virtual ~TernaryTreeCost() = default;

    double operator()(TernaryTree const& tree) const { return evaluate(tree); }

    virtual double evaluate(TernaryTree const& tree) const = 0;
};

/**
 * Sum of APSP-weighted interaction paths over Pauli supports (TreeOracle).
 * Edge weights in ``apsp`` determine the semantics (e.g. hop count vs log-infidelity).
 */
class ApspPathCost : public TernaryTreeCost {
public:
    ApspPathCost(FermionHamiltonian const& f_ham, dvlab::APSPResult<QubitIdType> const& apsp);

    double evaluate(TernaryTree const& tree) const override;

protected:
    FermionHamiltonian const& _f_ham;
    dvlab::APSPResult<QubitIdType> const& _apsp;
};

/// Total non-identity Pauli weight after qubitization.
class PauliWeightCost : public TernaryTreeCost {
public:
    explicit PauliWeightCost(FermionHamiltonian const& f_ham);

    double evaluate(TernaryTree const& tree) const override;

private:
    FermionHamiltonian const& _f_ham;
};

/**
 * Simulated annealing over ternary-tree encodings with a polymorphic cost.
 * When ``device`` is non-null, connectivity-preserving leaf moves are enabled.
 * ``seed`` fixes SA + mutation RNGs for reproducibility; omit for nondeterministic.
 */
TernaryTree optimize_ternary_tree_mapping(
    TernaryTree const& initial_tree,
    TernaryTreeCost const& cost,
    device::Device const* device         = nullptr,
    std::optional<std::uint32_t> seed    = std::nullopt);

}  // namespace qsyn::hamiltonian
