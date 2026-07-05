/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Implement the Treespilation mapping from fermionic Hamiltonian directly to mapped quantum circuits ]
  Author       [ Mu-Te (Joshua) Lau (joshmtlau) ]
*/

#pragma once

#include <tl/expected.hpp>
#include <vector>

#include "device/device.hpp"
#include "device/device_analysis.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "qcir/qcir.hpp"
#include "ternary_tree.hpp"

namespace qsyn::hamiltonian {

enum class TreespileFailReason : uint8_t {
    empty_hamiltonian,
    device_too_small,
    invalid_n_trotterization_steps,
    tt_build_failed_not_enough_qubits,
};

struct TreespileResult {
    qcir::QCir circuit;
    /// Populated when ``use_logical_indices`` is true: ``physical_qubits[logical]`` on the parent device.
    std::vector<QubitIdType> physical_qubits;
    /// Fermion-to-qubit encoding used for this treespile (stored on the FHam workspace).
    std::unique_ptr<FermionToQubitMapping> encoding;

    TreespileResult()                                      = default;
    TreespileResult(TreespileResult&&) noexcept            = default;
    TreespileResult& operator=(TreespileResult&&) noexcept = default;
    TreespileResult(TreespileResult const&)                = delete;
    TreespileResult& operator=(TreespileResult const&)     = delete;
    ~TreespileResult()                                     = default;
};

tl::expected<TreespileResult, TreespileFailReason>
treespile(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    double time,
    size_t n_trotterization_steps,
    device::APSPCostFnType const& cost_fn = device::default_floyd_warshall_cost,
    bool optimize1                        = false,
    bool optimize2                        = false,
    bool exhaustive                       = false,
    bool use_logical_indices              = false);

}  // namespace qsyn::hamiltonian
