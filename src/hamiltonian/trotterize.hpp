/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Define Trotterization functions ]
  Author       [ Mu-Te Lau (joshmtlau) ]
*/

#pragma once

#include "./qubit_hamiltonian.hpp"
#include "tableau/pauli_rotation.hpp"

namespace qsyn::hamiltonian {

qsyn::tableau::PauliRotationTableau trotterize_single_step(
    QubitHamiltonian const& hamiltonian, double dt) noexcept;

qsyn::tableau::PauliRotationTableau trotterize(
    QubitHamiltonian const& hamiltonian, double time, size_t n_steps) noexcept;

}  // namespace qsyn::hamiltonian
