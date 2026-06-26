/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Fermion-to-qubit mappings (reusable F2Q logic) ]
  Author       [ April Wang (april864) ]
*/

#include "hamiltonian/f2q_mappings.hpp"

#include <fmt/core.h>

#include <complex>
#include <unordered_map>

#include "tableau/pauli_product_trait.hpp"
#include "ternarytree/tt_mappings.hpp"

namespace qsyn::hamiltonian {

using namespace qsyn::tableau;
using Pauli = qsyn::tableau::Pauli;

namespace {

std::vector<ComplexPauliTerm>
multiply_terms(std::vector<ComplexPauliTerm> const& lhs,
               std::vector<ComplexPauliTerm> const& rhs) {
    // Multiply two lists of ComplexPauliTerm
    // we don't attempt to combine like terms here. that's handled in the caller
    std::vector<ComplexPauliTerm> result;
    for (const auto& l : lhs) {
        for (const auto& r : rhs) {
            result.emplace_back(l * r);
        }
    }
    return result;
}

}  // namespace

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
    std::size_t n_qubits = f_hamilt.n_modes();

    // Accumulator for the final result; maps Pauli strings to their coefficients
    std::unordered_map<PauliProduct, std::complex<double>> term_map;

    for (auto const& [original_coeff, operators] : f_hamilt.get_terms()) {
        // Start with identity and original coefficient
        std::vector<ComplexPauliTerm> current_state;
        current_state.emplace_back(
            ComplexPauliTerm(
                PauliProduct(std::vector<Pauli>(n_qubits, Pauli::i), false),
                original_coeff));

        // Create JW term for each operator in the fermionic term
        for (const auto& [p, is_creation] : operators) {
            current_state = multiply_terms(
                current_state,
                mapping.map(p, is_creation));
        }

        // Add term to term_map and combining like terms
        std::ranges::for_each(current_state, [&term_map](const auto& t) {
            term_map[t.pauli_product()] += t.coeff();
        });
    }

    // Make final qubit hamiltonian from term_map
    QubitHamiltonian final_ham(n_qubits);
    std::ranges::for_each(term_map, [&final_ham](const auto& term) {
        const auto& [pauli_product, coeff] = term;
        if (coeff == std::complex<double>(0, 0)) return;
        final_ham.add_term(HermitianPauliTerm(pauli_product, coeff.real()));
    });

    return final_ham;
}

}  // namespace qsyn::hamiltonian
