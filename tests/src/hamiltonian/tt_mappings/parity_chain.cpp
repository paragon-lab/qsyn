/*
  to_clifford tests on parity (left-chain) ternary trees.
*/

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "hamiltonian/tt_mappings/fixtures.hpp"

using namespace qsyn::hamiltonian;
using namespace qsyn::hamiltonian::tt_mappings_test;

TEST_CASE("Parity chain: CX(i, j) for all i < j", "[tt_mappings][parity]") {
    constexpr size_t num_qubits = 3;
    TernaryTree tree            = make_parity_chain_tree(num_qubits);

    REQUIRE(tree.num_qubits() == num_qubits);

    auto* curr = dynamic_cast<TernaryQubitNode*>(tree.get_root());
    REQUIRE(curr != nullptr);
    REQUIRE(curr->id == num_qubits - 1);
    REQUIRE(!curr->is_braided);
    REQUIRE(curr->get_mid_child()->is_leg());
    REQUIRE(curr->get_right_child()->is_leg());

    for (size_t mode = num_qubits - 1; mode > 0; --mode) {
        REQUIRE(!curr->get_left_child()->is_leg());
        curr = dynamic_cast<TernaryQubitNode*>(curr->get_left_child());
        REQUIRE(curr != nullptr);
        REQUIRE(curr->id == mode - 1);
        REQUIRE(!curr->is_braided);
        REQUIRE(curr->get_mid_child()->is_leg());
        REQUIRE(curr->get_right_child()->is_leg());
    }
    REQUIRE(curr->get_left_child()->is_leg());

    TTMapper mapper(std::move(tree));
    auto const clifford = to_clifford(mapper);

    REQUIRE(clifford.n_qubits() == num_qubits);
    // n = 3: CX(1, 2), CX(0, 2), CX(0, 1) in that order
    REQUIRE(clifford == expected_parity_clifford(num_qubits));
}

TEST_CASE("Parity chain: braided mode 0", "[tt_mappings][parity]") {
    constexpr size_t num_qubits = 5;
    TernaryTree tree            = make_parity_chain_tree(num_qubits);
    set_braided(tree, {0});

    TTMapper mapper(std::move(tree));
    REQUIRE(to_clifford(mapper) == expected_parity_with_braiding(num_qubits, {0}));
}

TEST_CASE("Parity chain: braided j > 0 applies ZZ debraid", "[tt_mappings][parity]") {
    constexpr size_t num_qubits = 5;
    constexpr size_t braided    = 2;

    TernaryTree tree = make_parity_chain_tree(num_qubits);
    set_braided(tree, {braided});

    TTMapper mapper(std::move(tree));
    REQUIRE(to_clifford(mapper) == expected_parity_with_braiding(num_qubits, {braided}));
}

TEST_CASE("Parity chain: multiple braided modes", "[tt_mappings][parity]") {
    constexpr size_t num_qubits       = 5;
    std::vector<size_t> const braided = {0, 2, 3};

    TernaryTree tree = make_parity_chain_tree(num_qubits);
    set_braided(tree, braided);

    TTMapper mapper(std::move(tree));
    REQUIRE(to_clifford(mapper) == expected_parity_with_braiding(num_qubits, braided));
}

TEST_CASE("Parity chain: CX network for n = 5", "[tt_mappings][parity]") {
    constexpr size_t num_qubits = 5;
    TernaryTree tree            = make_parity_chain_tree(num_qubits);

    TTMapper mapper(std::move(tree));
    REQUIRE(to_clifford(mapper) == expected_parity_clifford(num_qubits));
}
