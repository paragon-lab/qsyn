/*
  Unit tests for Hamiltonian Trotterization.
*/

#include "hamiltonian/qubit_hamiltonian.hpp"
#include "hamiltonian/trotterize.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace qsyn::hamiltonian;

TEST_CASE("trotterize repeats one slice per step, skipping identity terms", "[trotterize]") {
    QubitHamiltonian hamiltonian{
        HermitianPauliTerm{"ZIIII", 1.0},
        HermitianPauliTerm{"XIIII", 0.5},
        HermitianPauliTerm{"IIIII", 4.0},
        HermitianPauliTerm{"IIIIZ", -1.5},
    };

    auto const prtabl = trotterize(hamiltonian, 0.1, 3);

    REQUIRE(prtabl.size() == 9);
    for (size_t step = 0; step < 3; ++step) {
        auto const base = 3 * step;
        REQUIRE(prtabl[base + 0].pauli_product().to_string() == "ZIIII");
        REQUIRE(prtabl[base + 1].pauli_product().to_string() == "XIIII");
        REQUIRE(prtabl[base + 2].pauli_product().to_string() == "IIIIZ");
    }
}
