/*
  to_clifford tests on JW (right-chain) ternary trees.
*/

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "hamiltonian/tt_mappings/fixtures.hpp"

using namespace qsyn::hamiltonian;
using namespace qsyn::hamiltonian::tt_mappings_test;

TEST_CASE("JW chain: identity Clifford", "[tt_mappings][jw]") {
    constexpr size_t num_qubits = 5;
    TernaryTree tree            = make_jw_chain_tree(num_qubits);

    REQUIRE(tree.num_qubits() == num_qubits);

    TernaryNode* curr = tree.get_root();
    for (size_t i = 0; i < num_qubits; ++i) {
        auto* qnode = dynamic_cast<TernaryQubitNode*>(curr);
        REQUIRE(qnode != nullptr);
        REQUIRE(qnode->id == i);
        REQUIRE(qnode->qubit_label == i);
        REQUIRE(!qnode->is_braided);
        REQUIRE(curr->get_left_child()->is_leg());
        REQUIRE(curr->get_mid_child()->is_leg());

        if (i + 1 < num_qubits) {
            REQUIRE(!curr->get_right_child()->is_leg());
            curr = curr->get_right_child();
        } else {
            REQUIRE(curr->get_right_child()->is_leg());
        }
    }

    TTMapper mapper(std::move(tree));
    auto clifford = to_clifford(mapper);

    REQUIRE(clifford.n_qubits() == num_qubits);
    REQUIRE(clifford.is_identity());
}

TEST_CASE("JW chain: single braided node", "[tt_mappings][jw]") {
    constexpr size_t num_qubits = 5;
    constexpr size_t braided    = 2;

    TernaryTree tree = make_jw_chain_tree(num_qubits);
    set_braided(tree, {braided});

    TTMapper mapper(std::move(tree));
    auto const clifford = to_clifford(mapper);
    auto const expected = expected_debraid_clifford(num_qubits, {braided});

    REQUIRE(clifford.n_qubits() == num_qubits);
    CAPTURE(format_synthesized_clifford_circuit(clifford));
    CAPTURE(format_synthesized_clifford_circuit(expected));
    REQUIRE(clifford == expected);
}

TEST_CASE("JW chain: multiple braided nodes", "[tt_mappings][jw]") {
    constexpr size_t num_qubits       = 5;
    std::vector<size_t> const braided = {0, 2, 4};

    TernaryTree tree = make_jw_chain_tree(num_qubits);
    set_braided(tree, braided);

    TTMapper mapper(std::move(tree));
    REQUIRE(to_clifford(mapper) == expected_debraid_clifford(num_qubits, braided));
}

TEST_CASE("JW chain: all nodes braided", "[tt_mappings][jw]") {
    constexpr size_t num_qubits = 5;

    TernaryTree tree = make_jw_chain_tree(num_qubits);
    set_braided(tree, {0, 1, 2, 3, 4});

    TTMapper mapper(std::move(tree));
    REQUIRE(to_clifford(mapper) == expected_debraid_clifford(num_qubits, {0, 1, 2, 3, 4}));
}
