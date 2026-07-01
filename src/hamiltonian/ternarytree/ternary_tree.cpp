/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Ternary tree structure ]
  Author       [ April Wang (april864), Mu-Te (Joshua) Lau (joshmtlau) ]
*/

#include "ternary_tree.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qsyn::hamiltonian {

TernaryEdge* TernaryNode::get_edge(BranchType branch) const {
    auto edge_id = static_cast<std::underlying_type_t<BranchType>>(branch);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return edges[edge_id].get();
}

TernaryNode* TernaryNode::get_child(BranchType branch) const {
    auto* edge = get_edge(branch);
    if (!edge) return nullptr;
    return edge->target.get();
}

void TernaryNode::set_edge(BranchType branch, std::unique_ptr<TernaryEdge>&& edge) {
    auto edge_id = static_cast<std::underlying_type_t<BranchType>>(branch);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    edges[edge_id] = std::move(edge);
}

TernaryTree::TernaryTree() : _num_qubits(1) {
    _root             = std::make_unique<TernaryQubitNode>(0);
    _index_to_node[0] = _root.get();
}

TernaryTree::TernaryTree(size_t num_qubits) {
    if (num_qubits == 0) {
        throw std::invalid_argument("A ternary tree must have at least one node (root)");
    }

    _root             = std::make_unique<TernaryQubitNode>(0);
    _index_to_node[0] = _root.get();

    _num_qubits = 1;

    size_t parent_index = 0;
    while (_num_qubits < num_qubits) {
        TernaryNode* parent = get_node_by_index(parent_index);

        for (auto branch : {BranchType::left, BranchType::mid, BranchType::right}) {
            if (_num_qubits >= num_qubits) break;
            add_qubit_node(parent, branch);
        }

        parent_index++;
    }

    assert(_num_qubits == num_qubits);

    append_legs_to_tree();
}

namespace {

std::unique_ptr<TernaryNode> clone_node(
    TernaryNode* old_node, TernaryNode* new_parent,
    std::unordered_map<TernaryNode const*, TernaryNode*>& old_to_new) {
    std::unique_ptr<TernaryNode> new_node;
    if (old_node->is_leg()) {
        new_node = std::make_unique<TernaryLeg>(new_parent, nullptr);
    } else {
        auto const old_qubit_node = dynamic_cast<TernaryQubitNode*>(old_node);
        assert(old_qubit_node);
        new_node                                                     = std::make_unique<TernaryQubitNode>(old_qubit_node->id, new_parent, nullptr);
        dynamic_cast<TernaryQubitNode*>(new_node.get())->qubit_label = old_qubit_node->qubit_label;
    }
    TernaryNode* new_node_ptr = new_node.get();
    old_to_new[old_node]      = new_node_ptr;

    for (int b = 0; b < 3; ++b) {
        auto const branch     = static_cast<BranchType>(b);
        TernaryEdge* old_edge = old_node->get_edge(branch);
        if (old_edge && old_edge->target) {
            auto new_target =
                clone_node(old_edge->target.get(), new_node_ptr, old_to_new);
            auto new_edge =
                std::make_unique<TernaryEdge>(new_node_ptr, std::move(new_target), branch);
            new_edge->target->incoming_edge = new_edge.get();
            new_node_ptr->set_edge(branch, std::move(new_edge));
        }
    }
    return new_node;
}

}  // namespace

TernaryTree::TernaryTree(TernaryTree const& other) : _num_qubits(other._num_qubits) {
    if (!other._root) {
        _root             = std::make_unique<TernaryQubitNode>(0);
        _index_to_node[0] = _root.get();
        _num_qubits       = 1;
        return;
    }
    std::unordered_map<TernaryNode const*, TernaryNode*> old_to_new;
    _root = clone_node(other._root.get(), nullptr, old_to_new);

    for (auto const& [index, old_node] : other._index_to_node) {
        _index_to_node[index] = old_to_new.at(old_node);
    }
    for (auto const& [qubit, old_node] : other._qubit_to_node) {
        _qubit_to_node[qubit] = old_to_new.at(old_node);
    }
    _legs.reserve(other._legs.size());
    for (TernaryLeg* old_leg : other._legs) {
        _legs.push_back(dynamic_cast<TernaryLeg*>(old_to_new.at(old_leg)));
    }
}

void TernaryTree::swap(TernaryTree& other) noexcept {
    _root.swap(other._root);

    std::swap(_num_qubits, other._num_qubits);

    std::swap(_index_to_node, other._index_to_node);
    std::swap(_qubit_to_node, other._qubit_to_node);
    std::swap(_legs, other._legs);
}

void swap(TernaryTree& a, TernaryTree& b) noexcept {
    a.swap(b);
}

void TernaryTree::assign_qubit(size_t node_index, QubitIdType qubit_label) {
    TernaryNode* node = _index_to_node.at(node_index);

    dynamic_cast<TernaryQubitNode*>(node)->qubit_label = qubit_label;
    _qubit_to_node[qubit_label]                        = node;
}

void TernaryTree::swap_indices(std::size_t id1, std::size_t id2) {
    if (id1 == id2) return;

    auto* node1 = static_cast<TernaryQubitNode*>(_index_to_node.at(id1));
    auto* node2 = static_cast<TernaryQubitNode*>(_index_to_node.at(id2));

    std::swap(node1->id, node2->id);

    _index_to_node[id1] = node2;
    _index_to_node[id2] = node1;
}

void TernaryTree::remap_node_ids(std::span<size_t const> slot_of_label) {
    if (slot_of_label.size() != _num_qubits) {
        throw std::invalid_argument("slot_of_label size must match num_qubits()");
    }

    std::vector<size_t> slot_contents(_num_qubits);
    std::iota(slot_contents.begin(), slot_contents.end(), 0);

    for (size_t label = 0; label < _num_qubits; ++label) {
        size_t const target_phys = slot_of_label[label];
        if (target_phys >= _num_qubits) {
            throw std::invalid_argument("slot_of_label contains invalid construction index");
        }
        while (slot_contents[label] != target_phys) {
            size_t swap_slot = label;
            for (size_t s = 0; s < _num_qubits; ++s) {
                if (slot_contents[s] == target_phys) {
                    swap_slot = s;
                    break;
                }
            }
            swap_indices(label, swap_slot);
            std::swap(slot_contents[label], slot_contents[swap_slot]);
        }
    }
}

bool TernaryTree::has_edge(QubitIdType u, QubitIdType v) const {
    auto it_u = _qubit_to_node.find(u);
    auto it_v = _qubit_to_node.find(v);
    
    if (it_u == _qubit_to_node.end() || it_v == _qubit_to_node.end()) {
        return false;
    }
    
    TernaryNode* node_u = it_u->second;
    TernaryNode* node_v = it_v->second;
    
    return (node_u->parent == node_v) || (node_v->parent == node_u);
}

size_t TernaryTree::add_qubit_node(TernaryNode* parent, BranchType branch) {
    if (parent->has_child(branch)) {
        throw std::runtime_error("Branch already has a node");
    }

    auto const node_index      = _num_qubits++;
    auto child                 = std::make_unique<TernaryQubitNode>(node_index, parent);
    TernaryNode* child_ptr     = child.get();
    _index_to_node[node_index] = child_ptr;

    auto edge                = std::make_unique<TernaryEdge>(parent, std::move(child), branch);
    child_ptr->incoming_edge = static_cast<TernaryEdge*>(edge.get());
    parent->set_edge(branch, std::move(edge));
    return node_index;
}

/**
 * @brief Add a qubit node to the first empty branch of the parent.
 * @param parent Parent node
 * @return Index of the added qubit node, or std::nullopt if no empty branch is found
 */
std::optional<size_t> TernaryTree::add_qubit_node_to_first_empty_branch(TernaryNode* parent) {
    for (auto branch : {BranchType::left, BranchType::mid, BranchType::right}) {
        if (!parent->has_child(branch)) {
            return std::make_optional(add_qubit_node(parent, branch));
        }
    }
    return std::nullopt;
}

size_t TernaryTree::add_leg_node(TernaryNode* parent, BranchType branch) {
    if (parent->has_child(branch)) {
        throw std::runtime_error("Branch already has a node");
    }

    auto child               = std::make_unique<TernaryLeg>(parent);
    TernaryLeg* child_ptr    = child.get();
    auto edge                = std::make_unique<TernaryEdge>(parent, std::move(child), branch);
    child_ptr->incoming_edge = static_cast<TernaryEdge*>(edge.get());
    parent->set_edge(branch, std::move(edge));
    _legs.push_back(child_ptr);
    return _legs.size() - 1;
}

/**
 * @brief Append legs all qubit nodes in the tree.
 *        Call this function after adding all qubit nodes to the tree.
 */
void TernaryTree::append_legs_to_tree() {
    for (size_t i = 0; i < _num_qubits; ++i) {
        TernaryNode* node = get_node_by_index(i);
        for (auto branch : {BranchType::left, BranchType::mid, BranchType::right}) {
            if (!node->has_child(branch)) {
                add_leg_node(node, branch);
            }
        }
    }
}

void TernaryTree::swap_legs(TernaryLeg* leg1, TernaryLeg* leg2) {
    if (leg1 == leg2) return;

    auto belongs_to_tree = [this](TernaryLeg* leg) {
        return std::find(_legs.begin(), _legs.end(), leg) != _legs.end();
    };
    if (!belongs_to_tree(leg1) || !belongs_to_tree(leg2)) {
        throw std::invalid_argument("Leg does not belong to this tree");
    }

    TernaryEdge* edge1 = leg1->incoming_edge;
    TernaryEdge* edge2 = leg2->incoming_edge;

    edge1->target.swap(edge2->target);

    leg1->parent        = edge2->source;
    leg1->incoming_edge = edge2;

    leg2->parent        = edge1->source;
    leg2->incoming_edge = edge1;
}

namespace {

void append_qubit_node_label(TernaryQubitNode const& qubit_node, std::string& out) {
    if (qubit_node.qubit_label.has_value()) {
        out += fmt::format("{} (q{})\n", qubit_node.id, qubit_node.qubit_label.value());
    } else {
        out += fmt::format("{} (unassigned)\n", qubit_node.id);
    }
}

void append_node_hierarchy(TernaryNode* node, std::string const& prefix,
                           bool is_last, std::string& out) {
    auto const branch_char = is_last ? "└── " : "├── ";
    out += prefix + branch_char;

    if (node->is_leg()) {
        out += "(leg)\n";
        return;
    }
    auto const qubit_node = dynamic_cast<TernaryQubitNode*>(node);
    assert(qubit_node);
    append_qubit_node_label(*qubit_node, out);

    std::array<TernaryNode*, 3> children = {nullptr, nullptr, nullptr};
    size_t n                             = 0;
    for (auto branch : {BranchType::left, BranchType::mid, BranchType::right}) {
        auto* child = node->get_child(branch);
        if (child) {
            children.at(n) = child;
            n++;
        }
    }

    std::string child_prefix = prefix + (is_last ? "    " : "│   ");
    for (size_t i = 0; i < n; ++i) {
        append_node_hierarchy(children.at(i), child_prefix, i == n - 1, out);
    }
}

}  // namespace

std::string format_logical_to_physical_mapping(TernaryTree const& tree) {
    std::vector<std::string> physical;
    physical.reserve(tree.num_qubits());
    for (size_t i = 0; i < tree.num_qubits(); ++i) {
        auto* const node       = tree.get_node_by_index(i);
        auto* const qubit_node = dynamic_cast<TernaryQubitNode*>(node);
        assert(qubit_node != nullptr);
        if (qubit_node->qubit_label.has_value()) {
            physical.push_back(fmt::format("{}", qubit_node->qubit_label.value()));
        } else {
            physical.push_back("?");
        }
    }
    return fmt::format("Logical → physical: [{}]\n", fmt::join(physical, ", "));
}

std::string to_string(TernaryTree const& tree) {
    std::string out;
    TernaryNode* root = tree.get_root();
    if (!root) {
        return "(empty tree)";
    }
    auto const root_qubit_node = dynamic_cast<TernaryQubitNode*>(root);
    assert(root_qubit_node);
    append_qubit_node_label(*root_qubit_node, out);
    std::array<TernaryNode*, 3> children = {nullptr, nullptr, nullptr};
    size_t n                             = 0;
    for (auto branch : {BranchType::left, BranchType::mid, BranchType::right}) {
        auto* child = root->get_child(branch);
        if (child) {
            children.at(n++) = child;
        }
    }
    std::string child_prefix = "";
    for (size_t i = 0; i < n; ++i) {
        append_node_hierarchy(children.at(i), child_prefix, i == n - 1, out);
    }
    return out;
}

}  // namespace qsyn::hamiltonian
