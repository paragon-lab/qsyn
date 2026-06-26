/*
  to_clifford tests on the branching example ternary tree.
*/

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hamiltonian/tt_mappings/fixtures.hpp"

using namespace qsyn::hamiltonian;
using namespace qsyn::hamiltonian::tt_mappings_test;

TEST_CASE("Branching example: tree topology after remap_node_ids", "[tt_mappings][branching]") {
    TernaryTree tree = make_branching_example_tree();
    REQUIRE(tree.num_qubits() == 9);

    // Fermion labels: 4 -> (1, 7, 8); 1 -> (0, 2, 3); 7 -> (5, _, _); 5 -> (_, _, 6)
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_root())->id == 4);
    REQUIRE(tree.get_node_by_qubit(4) == tree.get_root());
    for (size_t k = 0; k < 9; ++k) {
        REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(k))->id == k);
        REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_index(k))->qubit_label == k);
    }
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_qubit(1))->get_left_child() ==
            tree.get_node_by_qubit(0));
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_qubit(1))->get_mid_child() ==
            tree.get_node_by_qubit(2));
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_qubit(1))->get_right_child() ==
            tree.get_node_by_qubit(3));
    REQUIRE(dynamic_cast<TernaryQubitNode*>(tree.get_node_by_qubit(7))->get_left_child() ==
            tree.get_node_by_qubit(5));
    REQUIRE(tree.get_node_by_qubit(7)->get_mid_child()->is_leg());
    REQUIRE(tree.get_node_by_qubit(7)->get_right_child()->is_leg());
    REQUIRE(tree.get_node_by_qubit(5)->get_left_child()->is_leg());
    REQUIRE(tree.get_node_by_qubit(5)->get_mid_child()->is_leg());
    REQUIRE(tree.get_node_by_qubit(5)->get_right_child() == tree.get_node_by_qubit(6));
    REQUIRE(tree.get_node_by_qubit(8)->get_left_child()->is_leg());
    REQUIRE(tree.get_node_by_qubit(8)->get_mid_child()->is_leg());
    REQUIRE(tree.get_node_by_qubit(8)->get_right_child()->is_leg());
}

TEST_CASE("Branching example: unbraided reference Clifford", "[tt_mappings][branching]") {
    TernaryTree tree = make_branching_example_tree();
    TTMapper mapper(std::move(tree));
    auto const clifford = to_clifford(mapper);
    auto const expected = expected_branching_example_clifford();

    REQUIRE(clifford.n_qubits() == 9);
    CAPTURE(format_synthesized_clifford_circuit(expected));
    CAPTURE(format_synthesized_clifford_circuit(clifford));
    REQUIRE(clifford == expected);
}

TEST_CASE("Branching example: braided modes 4 and 7", "[tt_mappings][branching]") {
    TernaryTree tree = make_branching_example_tree();
    set_braided(tree, {4, 7});
    TTMapper mapper(std::move(tree));

    auto [left_4, right_4] = mapper.leg_pairs().at(4);
    auto [left_7, right_7] = mapper.leg_pairs().at(7);
    auto const product_4   = mapper.leg_pauli_product(left_4, right_4);
    auto const product_7   = mapper.leg_pauli_product(left_7, right_7);
    require_iz_pauli_product(product_4, {1, 3, 4, 7});
    require_iz_pauli_product(product_7, {5, 6, 7});

    auto const clifford = to_clifford(mapper);
    auto const expected = expected_branching_braided_4_7_clifford();

    REQUIRE(clifford.n_qubits() == 9);
    CAPTURE(format_synthesized_clifford_circuit(expected));
    CAPTURE(format_synthesized_clifford_circuit(clifford));
    REQUIRE(clifford == expected);
    REQUIRE(clifford != expected_branching_example_clifford());
}
