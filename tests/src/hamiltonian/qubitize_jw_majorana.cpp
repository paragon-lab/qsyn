/*
  Unit tests: Fermion → Majorana → JW Pauli (qubitize / qubitize_jw).
*/

#include "hamiltonian/f2q_majorana.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "hamiltonian/majorana_hamiltonian.hpp"
#include "hamiltonian/qubit_hamiltonian.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <map>
#include <string>

using namespace qsyn::hamiltonian;

namespace {

std::map<std::string, double> coeff_map(QubitHamiltonian const& ham) {
    std::map<std::string, double> out;
    for (auto const& term : ham) {
        // HermitianPauliTerm normalizes global minus into the coefficient.
        out[term.pauli_product().to_string('+')] += term.coeff();
    }
    // Drop numerical zeros.
    for (auto it = out.begin(); it != out.end();) {
        if (std::abs(it->second) < 1e-10) {
            it = out.erase(it);
        } else {
            ++it;
        }
    }
    return out;
}

void require_same_qubit_hamiltonian(
    QubitHamiltonian const& lhs,
    QubitHamiltonian const& rhs) {
    auto const left  = coeff_map(lhs);
    auto const right = coeff_map(rhs);
    INFO("legacy:\n"
         << lhs.to_string());
    INFO("via Majorana:\n"
         << rhs.to_string());
    REQUIRE(left.size() == right.size());
    for (auto const& [pauli, coeff] : left) {
        auto const it = right.find(pauli);
        REQUIRE(it != right.end());
        REQUIRE(it->second == Catch::Approx(coeff).margin(1e-10));
    }
}

void require_jw_fermion_and_majorana_entrypoints_match(FermionHamiltonian const& f_ham) {
    auto const from_fermion  = qubitize(f_ham, JordanWignerMapping{f_ham.n_modes()});
    auto const from_majorana = qubitize_jw(to_majorana(f_ham));
    require_same_qubit_hamiltonian(from_fermion, from_majorana);
}

}  // namespace

TEST_CASE("jw_map_majorana_product: identity and single Majoranas", "[qubitize][majorana][jw]") {
    auto const id = jw_map_majorana_product(2, 1.5, {});
    REQUIRE(id.pauli_product().to_string('+') == "+II");
    REQUIRE(id.coeff().real() == Catch::Approx(1.5));

    auto const g0 = jw_map_majorana_product(2, 1.0, {0});
    REQUIRE(g0.pauli_product().to_string('+') == "+XI");

    auto const g1 = jw_map_majorana_product(2, 1.0, {1});
    REQUIRE(g1.pauli_product().to_string('+') == "+YI");

    auto const g2 = jw_map_majorana_product(2, 1.0, {2});
    REQUIRE(g2.pauli_product().to_string('+') == "+ZX");
}

TEST_CASE("jw_map_majorana_product: γ0 γ1 = i Z", "[qubitize][majorana][jw]") {
    auto const product = jw_map_majorana_product(1, 1.0, {0, 1});
    // X * Y = i Z
    REQUIRE(product.pauli_product().to_string('+') == "+Z");
    REQUIRE(product.coeff().real() == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(product.coeff().imag() == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("qubitize_jw: number operator n0", "[qubitize][majorana][jw]") {
    FermionHamiltonian f_ham(1);
    f_ham.add_term(1.0, {{0, true}, {0, false}});

    auto const q_ham = qubitize_jw(f_ham);
    // n = 1/2 (I + Z)
    auto const coeffs = coeff_map(q_ham);
    REQUIRE(coeffs.size() == 2);
    REQUIRE(coeffs.at("+I") == Catch::Approx(0.5));
    REQUIRE(coeffs.at("+Z") == Catch::Approx(0.5));
}

TEST_CASE("qubitize JW: fermion and Majorana entry points agree on number operator", "[qubitize][majorana][jw]") {
    FermionHamiltonian f_ham(1);
    f_ham.add_term(1.0, {{0, true}, {0, false}});
    require_jw_fermion_and_majorana_entrypoints_match(f_ham);
}

TEST_CASE("qubitize JW: fermion and Majorana entry points agree on Hermitian hopping", "[qubitize][majorana][jw]") {
    FermionHamiltonian f_ham(2);
    f_ham.add_term(1.0, {{0, true}, {1, false}});
    f_ham.add_term(1.0, {{1, true}, {0, false}});
    require_jw_fermion_and_majorana_entrypoints_match(f_ham);
}

TEST_CASE("qubitize JW: fermion and Majorana entry points agree on density-density", "[qubitize][majorana][jw]") {
    // n0 n1 = a0† a0 a1† a1
    FermionHamiltonian f_ham(2);
    f_ham.add_term(1.0, {{0, true}, {0, false}, {1, true}, {1, false}});
    require_jw_fermion_and_majorana_entrypoints_match(f_ham);
}

TEST_CASE("qubitize JW: fermion and Majorana entry points agree on multi-term H", "[qubitize][majorana][jw]") {
    FermionHamiltonian f_ham(3);
    f_ham.add_term(0.5, {{0, true}, {0, false}});
    f_ham.add_term(-0.25, {{1, true}, {1, false}});
    f_ham.add_term(1.0, {{0, true}, {1, false}});
    f_ham.add_term(1.0, {{1, true}, {0, false}});
    f_ham.add_term(0.3, {{1, true}, {2, false}});
    f_ham.add_term(0.3, {{2, true}, {1, false}});
    require_jw_fermion_and_majorana_entrypoints_match(f_ham);
}
