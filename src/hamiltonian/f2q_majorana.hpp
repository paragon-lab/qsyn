/*
  PackageName  [ hamiltonian ]
  Synopsis     [ JW Majorana Paulis and Clifford conjugation (stabilizer formalism) ]
*/

#pragma once

#include <cstddef>
#include <string>

#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/qubit_hamiltonian.hpp"
#include "tableau/stabilizer_tableau.hpp"

namespace qsyn::hamiltonian {

/// JW Majorana Pauli \(\gamma_j\) with \(j \in [0, 2n)\): \(\gamma_{2p} = Z^{\otimes p} X\), \(\gamma_{2p+1} = Z^{\otimes p} Y\).
ComplexPauliTerm jw_majorana_term(std::size_t n_modes, std::size_t index);

std::string majorana_label(std::size_t index);

/// Conjugate a Pauli operator by Clifford \(C\): \(P \mapsto C P C^\dagger\) (Heisenberg / `PauliProduct::apply`).
ComplexPauliTerm conjugate_pauli_term(
    ComplexPauliTerm const& term,
    qsyn::tableau::StabilizerTableau const& clifford);

void print_majorana_encoding_correspondence(FermionToQubitMapping const& mapping);

}  // namespace qsyn::hamiltonian
