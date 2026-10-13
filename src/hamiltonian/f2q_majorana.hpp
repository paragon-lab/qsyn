/*
  PackageName  [ hamiltonian ]
  Synopsis     [ JW Majorana Paulis and Clifford conjugation (stabilizer formalism) ]
*/

#pragma once

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "hamiltonian/majorana_hamiltonian.hpp"
#include "hamiltonian/qubit_hamiltonian.hpp"
#include "tableau/stabilizer_tableau.hpp"

namespace qsyn::hamiltonian {

/// JW Majorana Pauli \(\gamma_j\) with \(j \in [0, 2n)\): \(\gamma_{2p} = Z^{\otimes p} X\), \(\gamma_{2p+1} = Z^{\otimes p} Y\).
ComplexPauliTerm jw_majorana_term(std::size_t n_modes, std::size_t index);

std::string majorana_label(std::size_t index);

/// Map a Majorana monomial \(\mathrm{coeff}\,\gamma_{i_1}\cdots\gamma_{i_k}\) under JW.
ComplexPauliTerm jw_map_majorana_product(
    std::size_t n_modes,
    std::complex<double> coeff,
    std::vector<std::size_t> const& indices);

/// Conjugate a Pauli operator by Clifford \(C\): \(P \mapsto C P C^\dagger\) (Heisenberg / `PauliProduct::apply`).
ComplexPauliTerm conjugate_pauli_term(
    ComplexPauliTerm const& term,
    qsyn::tableau::StabilizerTableau const& clifford);

/**
 * @brief Majorana \(\gamma_j\) under an encoding: \(P = C P_{\mathrm{JW}} C^\dagger\).
 *
 * For Jordan–Wigner, \(C = I\). For ternary trees, \(C\) is ``mapping.to_clifford()``.
 */
ComplexPauliTerm mapped_majorana_term(
    FermionToQubitMapping const& mapping,
    std::size_t index);

/// Map a Majorana monomial under an encoding (JW product, then conjugate by \(C\)).
ComplexPauliTerm map_majorana_product(
    FermionToQubitMapping const& mapping,
    std::complex<double> coeff,
    std::vector<std::size_t> const& indices);

/// ``qubitize`` with Jordan–Wigner.
QubitHamiltonian qubitize_jw(MajoranaHamiltonian const& m_ham);
QubitHamiltonian qubitize_jw(FermionHamiltonian const& f_ham);

/// ``qubitize`` with a ternary-tree mapping.
QubitHamiltonian qubitize_tt(
    MajoranaHamiltonian const& m_ham,
    TernaryTreeMapping const& mapping);
QubitHamiltonian qubitize_tt(
    FermionHamiltonian const& f_ham,
    TernaryTreeMapping const& mapping);

void print_majorana_encoding_correspondence(FermionToQubitMapping const& mapping);

}  // namespace qsyn::hamiltonian
