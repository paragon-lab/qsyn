/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Fermion-to-qubit mappings (reusable F2Q logic) ]
  Author       [ April Wang (april864) ]
*/

#include "hamiltonian/f2q_mappings.hpp"

#include <complex>
#include <vector>

#include "ternarytree/tt_mappings.hpp"

namespace qsyn::hamiltonian {

using Pauli = qsyn::tableau::Pauli;

std::vector<ComplexPauliTerm>
JordanWignerMapping::map(std::size_t p, bool is_creation) const {
    // Make terms
    // a_p  = 0.5 Z_0 ... Z_{p-1} (X_p - iY_p)
    // a_p^ = 0.5 Z_0 ... Z_{p-1} (X_p + iY_p)
    std::vector<Pauli> z_string(this->n_modes(), Pauli::i);
    for (size_t i = 0; i < p; i++) {
        z_string[i] = Pauli::z;
    }

    std::vector<Pauli> x_part = z_string;
    x_part[p]                 = Pauli::x;
    std::vector<Pauli> y_part = z_string;
    y_part[p]                 = Pauli::y;

    return {
        ComplexPauliTerm(x_part, 0.5),
        ComplexPauliTerm(y_part, is_creation
                                     ? std::complex<double>(0, 0.5)
                                     : std::complex<double>(0, -0.5))};
}

std::unique_ptr<FermionToQubitMapping> JordanWignerMapping::clone() const {
    return std::make_unique<JordanWignerMapping>(n_modes());
}

tableau::StabilizerTableau JordanWignerMapping::to_clifford() const {
    return tableau::StabilizerTableau(n_modes());
}

tableau::StabilizerTableau TernaryTreeMapping::to_clifford() const {
    return ::qsyn::hamiltonian::to_clifford(_mapper);
}

QubitHamiltonian qubitize(FermionHamiltonian const& f_hamilt,
                          FermionToQubitMapping const& mapping) {
    return qubitize(to_majorana(f_hamilt), mapping);
}

}  // namespace qsyn::hamiltonian
