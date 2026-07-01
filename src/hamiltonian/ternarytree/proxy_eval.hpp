/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Evaluates proxy cost functions for ternary tree optimization ]
  Author       [ April Wang (april864) ]
*/

#pragma once

#include <filesystem>
#include <tl/expected.hpp>
#include <vector>

#include "device/device.hpp"
#include "device/device_analysis.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "qcir/qcir.hpp"
#include "ternary_tree.hpp"

namespace qsyn::hamiltonian {

// Evaluates infidelity_cost proxy function to compare the proxy infidelity with the
// actual infidelity
void evaluate_proxy_cost(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    std::filesystem::path const& output_dir,
    std::string const& output_csv_name,
    size_t samples);

void evaluate_proxy_termwise_depth(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    std::string const& output_csv,
    size_t samples);

void evaluate_proxy_termwise_fidelity(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    std::string const& output_csv,
    size_t samples);

}  // namespace qsyn::hamiltonian
