/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Ternary tree structure ]
  Author       [ April Wang (april864), Mu-Te (Joshua) Lau (joshmtlau) ]
*/

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "qsyn/qsyn_type.hpp"

namespace qsyn::hamiltonian {

enum class BranchType : uint8_t {
    left  = 0,
    mid   = 1,
    right = 2,
};

struct TernaryEdge;

struct TernaryNode {
    TernaryNode* parent;

    std::array<std::unique_ptr<TernaryEdge>, 3> edges;
    TernaryEdge* incoming_edge;

    TernaryNode(TernaryNode* p = nullptr, TernaryEdge* e = nullptr)
        : parent(p), incoming_edge(e) {}

    virtual ~TernaryNode() = default;
    virtual bool is_leg() const { return false; }

    TernaryEdge* get_edge(BranchType branch) const;

    TernaryNode* get_child(BranchType branch) const;
    TernaryNode* get_left_child() const { return get_child(BranchType::left); }
    TernaryNode* get_mid_child() const { return get_child(BranchType::mid); }
    TernaryNode* get_right_child() const { return get_child(BranchType::right); }

    bool has_child(BranchType branch) const { return get_child(branch) != nullptr; }

    void set_edge(BranchType branch, std::unique_ptr<TernaryEdge>&& edge);
};

struct TernaryQubitNode : public TernaryNode {
    std::size_t id;
    std::optional<QubitIdType> qubit_label{std::nullopt};
    bool is_braided{false};
    TernaryQubitNode(std::size_t id, TernaryNode* p = nullptr, TernaryEdge* e = nullptr)
        : TernaryNode(p, e), id(id) {}
};

struct TernaryLeg : public TernaryNode {
    TernaryLeg(TernaryNode* p = nullptr, TernaryEdge* e = nullptr)
        : TernaryNode(p, e) {}

    bool is_leg() const override { return true; }
};

struct TernaryEdge {
    TernaryNode* source;
    std::unique_ptr<TernaryNode> target;
    BranchType branch;

    TernaryEdge(TernaryNode* s, std::unique_ptr<TernaryNode>&& t, BranchType b)
        : source(s), target(std::move(t)), branch(b) {}
};

class TernaryTree {  // NOLINT(hicpp-special-member-functions, cppcoreguidelines-special-member-functions) : copy-swap idiom
public:
    TernaryTree();
    TernaryTree(size_t num_qubits);

    TernaryTree(TernaryTree const& other);
    TernaryTree(TernaryTree&& other) noexcept = default;

    TernaryTree& operator=(TernaryTree copy) {
        copy.swap(*this);
        return *this;
    }

    void swap(TernaryTree& other) noexcept;
    friend void swap(TernaryTree& a, TernaryTree& b) noexcept;

    void assign_qubit(size_t node_index, QubitIdType qubit_label);
    void swap_indices(std::size_t id1, std::size_t id2);

    /**
     * @brief Permute index slots so get_node_by_index(k) is the node with fermion id k.
     * @param slot_of_label slot_of_label[k] is the construction index of the node that
     *        should be labeled k (before calling assign_qubit).
     */
    void remap_node_ids(std::span<size_t const> slot_of_label);

    TernaryNode* get_root() const { return _root.get(); }
    std::unique_ptr<TernaryNode>& get_root_ptr() { return _root; }

    TernaryNode* get_node_by_qubit(QubitIdType qubit_label) const { return _qubit_to_node.at(qubit_label); }
    TernaryNode* get_node_by_index(size_t node_index) const { return _index_to_node.at(node_index); }

    const std::vector<TernaryLeg*>& get_legs() const { return _legs; }
    TernaryLeg* get_leg(size_t i) const { return _legs.at(i); }

    bool has_edge(QubitIdType u, QubitIdType v) const;

    size_t num_qubits() const { return _num_qubits; }

    size_t add_qubit_node(TernaryNode* parent, BranchType branch);
    std::optional<size_t> add_qubit_node_to_first_empty_branch(TernaryNode* parent);
    size_t add_leg_node(TernaryNode* parent, BranchType branch);

    void append_legs_to_tree();

    void swap_legs(TernaryLeg* leg1, TernaryLeg* leg2);

private:
    std::unique_ptr<TernaryNode> _root;
    std::unordered_map<size_t, TernaryNode*> _index_to_node;
    std::unordered_map<QubitIdType, TernaryNode*> _qubit_to_node;
    std::vector<TernaryLeg*> _legs;
    size_t _num_qubits;
};

std::string to_string(TernaryTree const& tree);

/// ``Logical → physical: [p0, p1, ...]`` with ``physical[i]`` from ``get_node_by_index(i)``.
std::string format_logical_to_physical_mapping(TernaryTree const& tree);

}  // namespace qsyn::hamiltonian
