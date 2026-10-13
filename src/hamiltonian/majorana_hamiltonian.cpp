/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Majorana Hamiltonian and fermion-to-Majorana conversion ]
*/

#include "hamiltonian/majorana_hamiltonian.hpp"

#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/core.h>

namespace qsyn::hamiltonian {

namespace {

constexpr double k_coeff_tol = 1e-12;

bool near_zero(std::complex<double> const& z) {
    return std::abs(z) < k_coeff_tol;
}

void multiply_by_majorana_sum(
    std::vector<MajoranaHamiltonian::Term>& state,
    std::complex<double> even_coeff,
    std::size_t even_index,
    std::complex<double> odd_coeff,
    std::size_t odd_index) {
    std::vector<MajoranaHamiltonian::Term> next;
    next.reserve(state.size() * 2);
    for (auto const& [coeff, indices] : state) {
        {
            auto product = indices;
            product.push_back(even_index);
            next.emplace_back(coeff * even_coeff, std::move(product));
        }
        {
            auto product = indices;
            product.push_back(odd_index);
            next.emplace_back(coeff * odd_coeff, std::move(product));
        }
    }
    state = std::move(next);
}

}  // namespace

std::pair<std::complex<double>, std::vector<std::size_t>>
canonicalize_majorana_product(
    std::complex<double> coeff,
    std::vector<std::size_t> indices) {
    // Insertion sort: each swap of distinct Majoranas contributes a minus sign.
    for (std::size_t i = 1; i < indices.size(); ++i) {
        auto const key = indices[i];
        std::size_t j  = i;
        while (j > 0 && indices[j - 1] > key) {
            indices[j] = indices[j - 1];
            coeff      = -coeff;
            --j;
        }
        indices[j] = key;
    }

    // Collapse adjacent pairs: γ_i² = I.
    std::vector<std::size_t> collapsed;
    collapsed.reserve(indices.size());
    for (auto const index : indices) {
        if (!collapsed.empty() && collapsed.back() == index) {
            collapsed.pop_back();
        } else {
            collapsed.push_back(index);
        }
    }
    return {coeff, std::move(collapsed)};
}

void MajoranaHamiltonian::add_term(
    std::complex<double> coeff,
    std::vector<std::size_t> indices) {
    auto [canon_coeff, canon_indices] =
        canonicalize_majorana_product(coeff, std::move(indices));
    if (near_zero(canon_coeff)) {
        return;
    }

    for (auto const index : canon_indices) {
        if (index >= n_majoranas()) {
            throw std::invalid_argument(fmt::format(
                "Majorana index {} out of range for {} modes ({} Majoranas)",
                index,
                _n_modes,
                n_majoranas()));
        }
    }

    for (std::size_t i = 0; i < _terms.size(); ++i) {
        if (_terms[i].second == canon_indices) {
            _terms[i].first += canon_coeff;
            if (near_zero(_terms[i].first)) {
                _terms[i] = std::move(_terms.back());
                _terms.pop_back();
            }
            return;
        }
    }
    _terms.emplace_back(canon_coeff, std::move(canon_indices));
}

MajoranaHamiltonian to_majorana(FermionHamiltonian const& f_ham) {
    std::size_t const n_modes = f_ham.n_modes();
    MajoranaHamiltonian result(n_modes);

    // Accumulate into a map so like terms from different fermionic terms combine.
    std::map<std::vector<std::size_t>, std::complex<double>> term_map;

    for (auto const& [original_coeff, operators] : f_ham.get_terms()) {
        std::vector<MajoranaHamiltonian::Term> current;
        current.emplace_back(original_coeff, std::vector<std::size_t>{});

        for (auto const& [mode, is_creation] : operators) {
            if (mode >= n_modes) {
                throw std::invalid_argument(fmt::format(
                    "Fermion mode {} out of range for {} modes", mode, n_modes));
            }
            std::size_t const even = 2 * mode;
            std::size_t const odd  = 2 * mode + 1;
            // a  = 1/2 (γ_even - i γ_odd)
            // a† = 1/2 (γ_even + i γ_odd)
            constexpr std::complex<double> half(0.5, 0.0);
            constexpr std::complex<double> plus_i_half(0.0, 0.5);
            constexpr std::complex<double> minus_i_half(0.0, -0.5);
            if (is_creation) {
                multiply_by_majorana_sum(current, half, even, plus_i_half, odd);
            } else {
                multiply_by_majorana_sum(current, half, even, minus_i_half, odd);
            }
        }

        for (auto& [coeff, indices] : current) {
            auto [canon_coeff, canon_indices] =
                canonicalize_majorana_product(coeff, std::move(indices));
            if (near_zero(canon_coeff)) {
                continue;
            }
            term_map[std::move(canon_indices)] += canon_coeff;
        }
    }

    for (auto& [indices, coeff] : term_map) {
        if (near_zero(coeff)) {
            continue;
        }
        result.add_term(coeff, indices);
    }

    return result;
}

}  // namespace qsyn::hamiltonian
