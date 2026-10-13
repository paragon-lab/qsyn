/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Implement the Treespilation mapping from fermionic Hamiltonian directly to mapped quantum circuits ]
  Author       [ Mu-Te (Joshua) Lau (joshmtlau) ]
*/

#pragma once

#include <cstdint>
#include <tl/expected.hpp>
#include <optional>
#include <vector>

#include "device/device.hpp"
#include "device/device_analysis.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "qcir/qcir.hpp"
#include "tableau/pauli_rotation.hpp"
#include "ternary_tree.hpp"

namespace qsyn::hamiltonian {

enum class TreespileFailReason : uint8_t {
    empty_hamiltonian,
    device_too_small,
    invalid_n_trotterization_steps,
    tt_build_failed_not_enough_qubits,
    not_routable,
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

struct SynthesizedCircuit {
    qcir::QCir circuit;
    /// Populated when ``use_logical_indices`` is true.
    std::vector<QubitIdType> physical_qubits;
};

/// Synthesize a fermionic Hamiltonian on an existing ternary-tree embedding.
/// Equivalent to the synthesis stage of ``treespile`` (Trotter step of ``time / n``).
tl::expected<SynthesizedCircuit, TreespileFailReason>
synthesize_hamiltonian_on_tree(
    FermionHamiltonian const& hamiltonian,
    TernaryTree const& tree,
    device::Device const& device,
    device::APSPResult const& apsp,
    double time,
    size_t n_trotterization_steps,
    bool use_logical_indices = false);

/// Append a single Pauli rotation onto ``qcir``. Returns false if routing fails.
bool try_synthesize_term(
    tableau::PauliRotation const& rotation,
    TernaryTree const& tree,
    device::APSPResult const& apsp,
    device::Device const& device,
    qcir::QCir& qcir);

tl::expected<TreespileResult, TreespileFailReason>
treespile(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    double time,
    size_t n_trotterization_steps,
    device::APSPCostFnType const& cost_fn = device::default_floyd_warshall_cost,
    unsigned optimize_level               = 0,
    bool exhaustive                       = false,
    bool use_logical_indices              = false,
    std::optional<std::uint32_t> seed     = std::nullopt);

}  // namespace qsyn::hamiltonian
