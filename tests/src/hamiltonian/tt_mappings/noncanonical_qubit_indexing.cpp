/*
  to_clifford tests where inorder qubit indices differ from fermion labels.
*/

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hamiltonian/tt_mappings/fixtures.hpp"

using namespace qsyn::hamiltonian;
using namespace qsyn::hamiltonian::tt_mappings_test;

TEST_CASE("Noncanonical qubit indexing: tree topology", "[tt_mappings][noncanonical]") {
    TernaryTree tree = make_noncanonical_qubit_indexing_tree();
    REQUIRE(tree.num_qubits() == 5);

    // Fermion labels: 0 -> (1, 2, 3); 1 -> (4, _, _)
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_root())->id == 0);
    REQUIRE(tree.get_node_by_qubit(0) == tree.get_root());
    for (size_t k = 0; k < 5; ++k) {
        REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(k))->id == k);
        REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(k))->qubit_label == k);
    }
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_qubit(0))->get_left_child() ==
            tree.get_node_by_qubit(1));
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_qubit(0))->get_mid_child() ==
            tree.get_node_by_qubit(2));
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_qubit(0))->get_right_child() ==
            tree.get_node_by_qubit(3));
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_qubit(1))->get_left_child() ==
            tree.get_node_by_qubit(4));
    REQUIRE(tree.get_node_by_qubit(1)->get_mid_child()->is_leg());
    REQUIRE(tree.get_node_by_qubit(1)->get_right_child()->is_leg());
    for (size_t const leaf : {2, 3, 4}) {
        REQUIRE(tree.get_node_by_qubit(leaf)->get_left_child()->is_leg());
        REQUIRE(tree.get_node_by_qubit(leaf)->get_mid_child()->is_leg());
        REQUIRE(tree.get_node_by_qubit(leaf)->get_right_child()->is_leg());
    }
}

TEST_CASE("Noncanonical qubit indexing: reference Clifford", "[tt_mappings][noncanonical]") {
    TernaryTree tree = make_noncanonical_qubit_indexing_tree();
    TTMapper mapper(std::move(tree));
    auto const clifford = to_clifford(mapper);
    auto const expected = expected_noncanonical_qubit_indexing_clifford();

    REQUIRE(clifford.n_qubits() == 5);
    CAPTURE(format_synthesized_clifford_circuit(expected));
    CAPTURE(format_synthesized_clifford_circuit(clifford));
    REQUIRE(clifford == expected);
}
