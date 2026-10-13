/*
  Unit tests for fermion-to-Majorana conversion.
*/

#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "hamiltonian/majorana_hamiltonian.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <vector>

using namespace qsyn::hamiltonian;

namespace {

MajoranaHamiltonian::Term const*
find_term(MajoranaHamiltonian const& ham, std::vector<std::size_t> const& indices) {
    for (auto const& term : ham.get_terms()) {
        if (term.second == indices) {
            return &term;
        }
    }
    return nullptr;
}

void require_coeff(
    MajoranaHamiltonian const& ham,
    std::vector<std::size_t> const& indices,
    std::complex<double> expected) {
    auto const* term = find_term(ham, indices);
    REQUIRE(term != nullptr);
    REQUIRE(term->first.real() == Catch::Approx(expected.real()).margin(1e-12));
    REQUIRE(term->first.imag() == Catch::Approx(expected.imag()).margin(1e-12));
}

}  // namespace

TEST_CASE("canonicalize_majorana_product sorts with anticommutation signs", "[majorana]") {
    auto const [coeff, indices] = canonicalize_majorana_product(
        1.0, {2, 0, 1});
    REQUIRE(indices == std::vector<std::size_t>{0, 1, 2});
    // Two inversions: (2,0) and (2,1) → (+1)
    REQUIRE(coeff.real() == Catch::Approx(1.0));
    REQUIRE(coeff.imag() == Catch::Approx(0.0));
}

TEST_CASE("canonicalize_majorana_product removes γ_i² = I pairs", "[majorana]") {
    auto const [coeff, indices] = canonicalize_majorana_product(
        1.0, {0, 1, 0});
    REQUIRE(indices == std::vector<std::size_t>{1});
    REQUIRE(coeff.real() == Catch::Approx(-1.0));
    REQUIRE(coeff.imag() == Catch::Approx(0.0));
}

TEST_CASE("to_majorana: single annihilation", "[majorana][to_majorana]") {
    FermionHamiltonian f_ham(1);
    f_ham.add_term(1.0, {{0, false}});

    auto const m_ham = to_majorana(f_ham);
    REQUIRE(m_ham.n_modes() == 1);
    REQUIRE(m_ham.get_terms().size() == 2);
    // a_0 = 1/2 (γ_0 - i γ_1)
    require_coeff(m_ham, {0}, {0.5, 0.0});
    require_coeff(m_ham, {1}, {0.0, -0.5});
}

TEST_CASE("to_majorana: single creation", "[majorana][to_majorana]") {
    FermionHamiltonian f_ham(1);
    f_ham.add_term(1.0, {{0, true}});

    auto const m_ham = to_majorana(f_ham);
    REQUIRE(m_ham.get_terms().size() == 2);
    // a_0† = 1/2 (γ_0 + i γ_1)
    require_coeff(m_ham, {0}, {0.5, 0.0});
    require_coeff(m_ham, {1}, {0.0, 0.5});
}

TEST_CASE("to_majorana: number operator n_0 = a†_0 a_0", "[majorana][to_majorana]") {
    FermionHamiltonian f_ham(1);
    f_ham.add_term(1.0, {{0, true}, {0, false}});

    auto const m_ham = to_majorana(f_ham);
    // n_0 = 1/2 - (i/2) γ_0 γ_1
    REQUIRE(m_ham.get_terms().size() == 2);
    require_coeff(m_ham, {}, {0.5, 0.0});
    require_coeff(m_ham, {0, 1}, {0.0, -0.5});
}

TEST_CASE("to_majorana: Hermitian hopping a†_0 a_1 + a†_1 a_0", "[majorana][to_majorana]") {
    FermionHamiltonian f_ham(2);
    f_ham.add_term(1.0, {{0, true}, {1, false}});
    f_ham.add_term(1.0, {{1, true}, {0, false}});

    auto const m_ham = to_majorana(f_ham);

    // a†_0 a_1 = 1/4 (γ0 + i γ1)(γ2 - i γ3)
    //          = 1/4 (γ0γ2 - i γ0γ3 + i γ1γ2 - i² γ1γ3)
    //          = 1/4 (γ0γ2 - i γ0γ3 + i γ1γ2 + γ1γ3)
    // a†_1 a_0 = 1/4 (γ2 + i γ3)(γ0 - i γ1)
    //          = 1/4 (γ2γ0 - i γ2γ1 + i γ3γ0 - i² γ3γ1)
    //          = 1/4 (-γ0γ2 + i γ1γ2 - i γ0γ3 + γ1γ3)   after canonicalizing
    // Sum: 1/2 (γ1γ3 + γ0γ2)  wait let's compute carefully in the test expectations:
    // From first:  0.25 γ0γ2 + 0.25 γ1γ3 - 0.25i γ0γ3 + 0.25i γ1γ2
    // From second after canon:
    //   γ2γ0 = -γ0γ2 → -0.25 γ0γ2
    //   -i γ2γ1 = -i (-γ1γ2) = i γ1γ2 → +0.25i γ1γ2
    //   i γ3γ0 = i (-γ0γ3) = -i γ0γ3 → -0.25i γ0γ3
    //   γ3γ1 = -γ1γ3 → -0.25? Wait: -i² γ3γ1 = +γ3γ1 = -γ1γ3 → -0.25 γ1γ3
    //
    // Sum:
    // γ0γ2: 0.25 - 0.25 = 0
    // γ1γ3: 0.25 - 0.25 = 0
    // γ0γ3: -0.25i - 0.25i = -0.5i
    // γ1γ2: 0.25i + 0.25i = 0.5i
    //
    // So H = -0.5i γ0γ3 + 0.5i γ1γ2

    REQUIRE(m_ham.get_terms().size() == 2);
    require_coeff(m_ham, {0, 3}, {0.0, -0.5});
    require_coeff(m_ham, {1, 2}, {0.0, 0.5});
}
