/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Majorana Hamiltonian and fermion-to-Majorana conversion ]
*/

#pragma once

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "hamiltonian/fermionic_hamiltonian.hpp"

namespace qsyn::hamiltonian {

/**
 * @brief Hamiltonian as a sum of Majorana monomials.
 *
 * Each term is a coefficient times an ordered product of Majorana operators
 * \(\gamma_{i_1}\gamma_{i_2}\cdots\gamma_{i_k}\) with \(0 \le i_1 < i_2 < \cdots < i_k < 2n\).
 * The empty product denotes the identity.
 *
 * Convention (matching JW / Treespilation TT):
 * \(a_p = \frac12(\gamma_{2p} - i\gamma_{2p+1})\),
 * \(a_p^\dagger = \frac12(\gamma_{2p} + i\gamma_{2p+1})\).
 */
class MajoranaHamiltonian {
public:
    using Term = std::pair<std::complex<double>, std::vector<std::size_t>>;

    explicit MajoranaHamiltonian(std::size_t n_modes) : _n_modes(n_modes) {}

    /// Append a term after canonicalizing the Majorana product and merging like terms.
    void add_term(std::complex<double> coeff, std::vector<std::size_t> indices);

    std::size_t n_modes() const { return _n_modes; }
    std::size_t n_majoranas() const { return 2 * _n_modes; }

    std::vector<Term> const& get_terms() const { return _terms; }

private:
    std::size_t _n_modes;
    std::vector<Term> _terms;
};

/**
 * @brief Sort a Majorana product into ascending index order, apply anticommutation
 *        signs, and remove adjacent pairs (\(\gamma_i^2 = I\)).
 */
std::pair<std::complex<double>, std::vector<std::size_t>>
canonicalize_majorana_product(
    std::complex<double> coeff,
    std::vector<std::size_t> indices);

/**
 * @brief Convert a fermionic Hamiltonian into a Majorana Hamiltonian.
 *
 * Uses \(a_p = \frac12(\gamma_{2p}-i\gamma_{2p+1})\) and
 * \(a_p^\dagger = \frac12(\gamma_{2p}+i\gamma_{2p+1})\), expands each term,
 * canonicalizes Majorana products, and combines like terms.
 */
MajoranaHamiltonian to_majorana(FermionHamiltonian const& f_ham);

}  // namespace qsyn::hamiltonian
