/*
  PackageName  [ hamiltonian ]
  Synopsis     [ JW Majorana Paulis and Clifford conjugation (stabilizer formalism) ]
*/

#include "hamiltonian/f2q_majorana.hpp"

#include <fmt/core.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "tableau/pauli_product_trait.hpp"
#include "tableau/stabilizer_tableau.hpp"

namespace qsyn::hamiltonian {

using Pauli        = qsyn::tableau::Pauli;
using PauliProduct = qsyn::tableau::PauliProduct;

namespace {

constexpr double k_coeff_tol = 1e-12;

}  // namespace

ComplexPauliTerm jw_majorana_term(std::size_t n_modes, std::size_t index) {
    if (index >= 2 * n_modes) {
        throw std::invalid_argument(
            fmt::format("Majorana index {} out of range for {} modes", index, n_modes));
    }
    std::size_t const mode    = index / 2;
    Pauli const pauli_on_mode = (index % 2 == 0) ? Pauli::x : Pauli::y;
    std::vector<Pauli> paulis(n_modes, Pauli::i);
    for (std::size_t q = 0; q < mode; ++q) {
        paulis[q] = Pauli::z;
    }
    paulis[mode] = pauli_on_mode;
    return ComplexPauliTerm(paulis, std::complex<double>(1.0, 0.0));
}

std::string majorana_label(std::size_t index) {
    std::size_t const mode = index / 2;
    char const part        = (index % 2 == 0) ? 'X' : 'Y';
    return fmt::format("gamma_{} (mode {}, {})", index, mode, part);
}

ComplexPauliTerm jw_map_majorana_product(
    std::size_t n_modes,
    std::complex<double> coeff,
    std::vector<std::size_t> const& indices) {
    ComplexPauliTerm result(
        std::vector<Pauli>(n_modes, Pauli::i),
        coeff);
    for (auto const index : indices) {
        result *= jw_majorana_term(n_modes, index);
    }
    return result;
}

ComplexPauliTerm conjugate_pauli_term(
    ComplexPauliTerm const& term,
    qsyn::tableau::StabilizerTableau const& clifford) {
    auto pauli = term.pauli_product();
    auto ops   = qsyn::tableau::extract_clifford_operators(clifford);
    pauli.apply(ops);
    return ComplexPauliTerm(pauli, term.coeff());
}

ComplexPauliTerm mapped_majorana_term(
    FermionToQubitMapping const& mapping,
    std::size_t index) {
    auto const jw = jw_majorana_term(mapping.n_modes(), index);
    return conjugate_pauli_term(jw, mapping.to_clifford());
}

ComplexPauliTerm map_majorana_product(
    FermionToQubitMapping const& mapping,
    std::complex<double> coeff,
    std::vector<std::size_t> const& indices) {
    auto const jw = jw_map_majorana_product(mapping.n_modes(), coeff, indices);
    return conjugate_pauli_term(jw, mapping.to_clifford());
}

QubitHamiltonian qubitize(
    MajoranaHamiltonian const& m_ham,
    FermionToQubitMapping const& mapping) {
    std::size_t const n_modes = m_ham.n_modes();
    if (mapping.n_modes() != n_modes) {
        throw std::invalid_argument(fmt::format(
            "Mapping has {} modes but Majorana Hamiltonian has {}",
            mapping.n_modes(),
            n_modes));
    }

    auto const clifford = mapping.to_clifford();
    std::unordered_map<PauliProduct, std::complex<double>> term_map;

    for (auto const& [coeff, indices] : m_ham.get_terms()) {
        auto const jw     = jw_map_majorana_product(n_modes, coeff, indices);
        auto const mapped = conjugate_pauli_term(jw, clifford);
        term_map[mapped.pauli_product()] += mapped.coeff();
    }

    QubitHamiltonian final_ham(n_modes);
    for (auto const& [pauli_product, coeff] : term_map) {
        if (std::abs(coeff.real()) < k_coeff_tol && std::abs(coeff.imag()) < k_coeff_tol) {
            continue;
        }
        // Hermitian fermionic Hamiltonians yield real Pauli coefficients.
        final_ham.add_term(HermitianPauliTerm(pauli_product, coeff.real()));
    }
    return final_ham;
}

QubitHamiltonian qubitize_jw(MajoranaHamiltonian const& m_ham) {
    return qubitize(m_ham, JordanWignerMapping{m_ham.n_modes()});
}

QubitHamiltonian qubitize_jw(FermionHamiltonian const& f_ham) {
    return qubitize(f_ham, JordanWignerMapping{f_ham.n_modes()});
}

QubitHamiltonian qubitize_tt(
    MajoranaHamiltonian const& m_ham,
    TernaryTreeMapping const& mapping) {
    return qubitize(m_ham, mapping);
}

QubitHamiltonian qubitize_tt(
    FermionHamiltonian const& f_ham,
    TernaryTreeMapping const& mapping) {
    return qubitize(f_ham, mapping);
}

namespace {

void print_pauli_term_line(ComplexPauliTerm const& term) {
    auto const coeff = term.coeff();
    auto const pauli = term.pauli_product().to_string('+');
    if (std::abs(coeff.imag()) < 1e-12) {
        fmt::println("  {}", fmt::format("{} {}", coeff.real(), pauli));
    } else {
        fmt::println("  {}", fmt::format("({}, {}) {}", coeff.real(), coeff.imag(), pauli));
    }
}

}  // namespace

void print_majorana_encoding_correspondence(FermionToQubitMapping const& mapping) {
    std::size_t const n_modes = mapping.n_modes();
    auto const clifford       = mapping.to_clifford();

    fmt::println("");
    fmt::println(
        "Majorana Pauli correspondence (P_T = C P_JW C†, {} qubits, stabilizer tableau):",
        n_modes);
    fmt::println(
        "  {:<6}  {:<22}  {}",
        "index",
        "JW Pauli",
        "under encoding");

    for (std::size_t index = 0; index < 2 * n_modes; ++index) {
        auto const jw     = jw_majorana_term(n_modes, index);
        auto const mapped = conjugate_pauli_term(jw, clifford);

        fmt::println("");
        fmt::println("{}", majorana_label(index));
        fmt::println("  JW:");
        print_pauli_term_line(jw);
        fmt::println("  under encoding:");
        print_pauli_term_line(mapped);
    }
}

}  // namespace qsyn::hamiltonian
