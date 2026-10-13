/*
  Unit tests: Fermion → Majorana → TT Pauli (qubitize / qubitize_tt).
*/

#include "hamiltonian/f2q_majorana.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "hamiltonian/tt_mappings/fixtures.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <map>
#include <string>

using namespace qsyn::hamiltonian;
using namespace qsyn::hamiltonian::tt_mappings_test;

namespace {

std::map<std::string, double> coeff_map(QubitHamiltonian const& ham) {
    std::map<std::string, double> out;
    for (auto const& term : ham) {
        out[term.pauli_product().to_string('+')] += term.coeff();
    }
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

void require_tt_fermion_and_majorana_entrypoints_match(
    FermionHamiltonian const& f_ham,
    TernaryTreeMapping const& mapping) {
    auto const from_fermion  = qubitize(f_ham, mapping);
    auto const from_majorana = qubitize_tt(to_majorana(f_ham), mapping);
    require_same_qubit_hamiltonian(from_fermion, from_majorana);
}

FermionHamiltonian sample_hermitian_ham(std::size_t n_modes) {
    FermionHamiltonian f_ham(n_modes);
    f_ham.add_term(0.5, {{0, true}, {0, false}});
    if (n_modes >= 2) {
        f_ham.add_term(1.0, {{0, true}, {1, false}});
        f_ham.add_term(1.0, {{1, true}, {0, false}});
        f_ham.add_term(-0.25, {{1, true}, {1, false}});
    }
    if (n_modes >= 3) {
        f_ham.add_term(0.3, {{1, true}, {2, false}});
        f_ham.add_term(0.3, {{2, true}, {1, false}});
    }
    return f_ham;
}

}  // namespace

TEST_CASE("qubitize TT: fermion/Majorana entry points agree on JW-chain", "[qubitize][majorana][tt]") {
    constexpr size_t n = 3;
    auto const mapping = TernaryTreeMapping(make_jw_chain_tree(n));
    require_tt_fermion_and_majorana_entrypoints_match(sample_hermitian_ham(n), mapping);
}

TEST_CASE("qubitize TT: fermion/Majorana entry points agree on parity chain", "[qubitize][majorana][tt]") {
    constexpr size_t n = 4;
    auto const mapping = TernaryTreeMapping(make_parity_chain_tree(n));
    require_tt_fermion_and_majorana_entrypoints_match(sample_hermitian_ham(n), mapping);
}

TEST_CASE("qubitize TT: fermion/Majorana entry points agree on branching", "[qubitize][majorana][tt]") {
    auto const mapping = TernaryTreeMapping(make_branching_example_tree());
    require_tt_fermion_and_majorana_entrypoints_match(sample_hermitian_ham(mapping.n_modes()), mapping);
}

TEST_CASE("qubitize TT: fermion/Majorana entry points agree with braided JW-chain", "[qubitize][majorana][tt]") {
    constexpr size_t n = 5;
    TernaryTree tree   = make_jw_chain_tree(n);
    set_braided(tree, {0, 2, 4});
    auto const mapping = TernaryTreeMapping(std::move(tree));
    require_tt_fermion_and_majorana_entrypoints_match(sample_hermitian_ham(n), mapping);
}

TEST_CASE("qubitize TT: fermion/Majorana entry points agree with braided branching", "[qubitize][majorana][tt]") {
    TernaryTree tree = make_branching_example_tree();
    set_braided(tree, {4, 7});
    auto const mapping = TernaryTreeMapping(std::move(tree));
    require_tt_fermion_and_majorana_entrypoints_match(sample_hermitian_ham(mapping.n_modes()), mapping);
}

TEST_CASE("mapped_majorana_term: JW chain unbraided equals JW Majorana", "[qubitize][majorana][tt]") {
    constexpr size_t n = 3;
    auto const mapping = TernaryTreeMapping(make_jw_chain_tree(n));
    for (std::size_t index = 0; index < 2 * n; ++index) {
        auto const via_c = mapped_majorana_term(mapping, index);
        auto const jw    = jw_majorana_term(n, index);
        REQUIRE(via_c.pauli_product() == jw.pauli_product());
        REQUIRE(via_c.coeff().real() == Catch::Approx(jw.coeff().real()).margin(1e-12));
        REQUIRE(via_c.coeff().imag() == Catch::Approx(jw.coeff().imag()).margin(1e-12));
    }
}
