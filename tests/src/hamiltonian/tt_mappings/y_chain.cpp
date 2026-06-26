/*
  to_clifford tests on Y-chain (mid-branch) ternary trees.
*/

#include <unistd.h>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <string>
#include <utility>

#include "hamiltonian/tt_mappings/fixtures.hpp"

using namespace qsyn::hamiltonian;
using namespace qsyn::hamiltonian::tt_mappings_test;

TEST_CASE("Y chain: tree topology", "[tt_mappings][y_chain]") {
    constexpr size_t num_qubits = 5;
    TernaryTree tree            = y_chain_tree(num_qubits);

    REQUIRE(tree.num_qubits() == num_qubits);

    TernaryNode* curr = tree.get_root();
    for (size_t i = 0; i < num_qubits; ++i) {
        auto* qnode = dynamic_cast<TernaryQubitNode*>(curr);
        REQUIRE(qnode != nullptr);
        REQUIRE(qnode->id == i);
        REQUIRE(qnode->qubit_label == i);
        REQUIRE(!qnode->is_braided);
        REQUIRE(curr->get_left_child()->is_leg());
        REQUIRE(curr->get_right_child()->is_leg());

        if (i + 1 < num_qubits) {
            REQUIRE(!curr->get_mid_child()->is_leg());
            curr = curr->get_mid_child();
        } else {
            REQUIRE(curr->get_mid_child()->is_leg());
        }
    }
}

TEST_CASE("Y chain: reference Clifford", "[tt_mappings][y_chain]") {
    constexpr size_t num_qubits = 5;
    TernaryTree tree            = y_chain_tree(num_qubits);
    TTMapper mapper(std::move(tree));
    auto clifford       = to_clifford(mapper);
    auto const expected = expected_y_chain_clifford(num_qubits);

    REQUIRE(clifford.n_qubits() == num_qubits);
    CAPTURE(format_synthesized_clifford_circuit(expected));
    CAPTURE(format_synthesized_clifford_circuit(clifford));
    REQUIRE(clifford == expected);
}
