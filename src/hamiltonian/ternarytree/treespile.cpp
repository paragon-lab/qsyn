/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Treespilation mapping from fermionic Hamiltonian to mapped quantum circuits ]
  Author       [ Mu-Te (Joshua) Lau (joshmtlau) ]
*/

#include "./treespile.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <optional>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./bonsai.hpp"
#include "./ternary_tree.hpp"
#include "./tree_rotations.hpp"
#include "device/device_analysis.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/trotterize.hpp"
#include "qcir/basic_gate_type.hpp"
#include "tableau/pauli_rotation.hpp"
#include "tree_optimizations.hpp"
#include "util/graph/minimum_spanning_arborescence.hpp"
#include "util/simulated_annealing.hpp"

namespace qsyn::hamiltonian {

namespace {

TernaryQubitNode* as_qubit_node_or_throw(TernaryNode* node) {
    auto const qubit_node = dynamic_cast<TernaryQubitNode*>(node);
    if (!qubit_node) {
        throw std::runtime_error("Should not happen: tree node is not a qubit node");
    }
    return qubit_node;
}

size_t find_unique_connecting_ancilla(
    std::vector<std::vector<size_t>> const& connected_components,
    std::vector<size_t> const& physical_qubits,
    TernaryTree const& tree) {
    std::unordered_set<size_t> physical_qubits_set(
        physical_qubits.begin(),
        physical_qubits.end());
    std::vector<size_t> potential_ancilla_qubits;

    for (auto const& component : connected_components) {
        for (auto const& qubit : component) {
            auto const tree_node = as_qubit_node_or_throw(tree.get_node_by_qubit(qubit));
            auto const parent    = tree_node->parent;
            if (!parent) {
                // root node; skip
                continue;
            }
            auto const parent_node  = as_qubit_node_or_throw(parent);
            auto const parent_qubit = parent_node->qubit_label.value();
            if (!physical_qubits_set.contains(parent_qubit)) {
                potential_ancilla_qubits.push_back(parent_qubit);
            }
        }
    }

    if (potential_ancilla_qubits.empty()) {
        throw std::runtime_error("Should not happen: no potential ancilla qubits found");
    }

    auto majority_ancilla_qubit = potential_ancilla_qubits[0];
    for (auto const& qubit : potential_ancilla_qubits) {
        if (std::ranges::count(potential_ancilla_qubits, qubit) > potential_ancilla_qubits.size() / 2) {
            majority_ancilla_qubit = qubit;
        }
    }

    return majority_ancilla_qubit;
}

struct TermSynthesisInfo {
    std::vector<size_t> qubits;
    std::vector<bool> is_ancilla;

    void add_ancilla(size_t qubit) {
        qubits.push_back(qubit);
        is_ancilla.push_back(true);
    }
};

std::pair<size_t, size_t> find_shortest_terminals_between_two_connected_components(
    std::vector<std::vector<size_t>> const& connected_components,
    device::APSPResult const& apsp) {
    auto const& component0 = connected_components[0];
    auto const& component1 = connected_components[1];

    float min_dist                     = std::numeric_limits<float>::infinity();
    std::pair<size_t, size_t> min_pair = {0, 0};
    for (auto const& qubit : component0) {
        for (auto const& other_qubit : component1) {
            auto const dist = apsp.distance[qubit][other_qubit];
            if (dist < min_dist) {
                min_dist = dist;
                min_pair = {qubit, other_qubit};
            }
        }
    }

    return min_pair;
}

TermSynthesisInfo form_connected_components(
    tableau::PauliProduct const& pauli_product,
    TernaryTree const& tree,
    device::APSPResult const& apsp,
    device::Device const& device) {
    //
    std::vector<size_t> physical_qubits;

    for (size_t i = 0; i < pauli_product.n_qubits(); i++) {
        if (pauli_product.is_i(i)) {
            continue;
        }
        auto const tree_node = as_qubit_node_or_throw(tree.get_node_by_index(i));
        physical_qubits.push_back(tree_node->qubit_label.value());
    }

    auto const connected_components =
        get_connected_components(apsp, device, [&](size_t qubit) {
            return dvlab::contains(physical_qubits, qubit);
        });

    spdlog::debug("Connected components for term {}:", pauli_product.to_string());
    for (auto const& component : connected_components) {
        spdlog::debug("- Component: [{}]", fmt::join(component, ", "));
    }

    TermSynthesisInfo result{
        .qubits     = physical_qubits,
        .is_ancilla = std::vector<bool>(physical_qubits.size(), false),
    };

    if (physical_qubits.empty()) {
        return result;
    }

    // case 1: all qubits are in the same connected component
    if (connected_components.size() == 1) {
        return result;
    }

    // case 2: qubits are in two connected components
    // find the shortest path between the two connected components using the APSP result
    // add them to the synthesis info as ancilla qubits

    if (connected_components.size() == 2) {
        auto const [terminal0, terminal1] = find_shortest_terminals_between_two_connected_components(connected_components, apsp);
        auto const path                   = device::get_shortest_path(apsp, terminal0, terminal1);
        if (!path.has_value()) {
            throw std::runtime_error("Should not happen: no path found between two connected components");
        }

        // the two terminals are a part of the physical qubits. The rest must be
        // ancilla qubits because we are using the shortest path.
        auto const n_ancilla = path.value().size() - 2;
        result.qubits.reserve(result.qubits.size() + n_ancilla);
        result.is_ancilla.reserve(result.is_ancilla.size() + n_ancilla);
        for (auto const& qubit : path.value() | std::views::drop(1) | std::views::take(n_ancilla)) {
            result.add_ancilla(qubit);
        }

        return result;
    }

    // case 3: qubits are in multiple connected components
    // in this case, there must be a unique qubit that is not in physical_qubits
    // such that adding it to physical_qubits will form a single connected component
    // use the tree to check the parent of each tree nodes.

    auto const ancilla_qubit =
        find_unique_connecting_ancilla(connected_components, physical_qubits, tree);

    result.add_ancilla(ancilla_qubit);
    return result;
}

dvlab::Digraph<size_t, float> form_device_subgraph(
    TermSynthesisInfo const& synthesis_info,
    device::Device const& device,
    device::APSPResult const& apsp) {
    auto device_subgraph = dvlab::Digraph<size_t, float>{};
    // add all physical qubits to the subgraph (use qubit IDs as vertex IDs so
    // apsp.distance[src][dst] and ancilla lookups work correctly)
    for (auto const& qubit : synthesis_info.qubits) {
        device_subgraph.add_vertex_with_id(qubit);
    }
    for (auto const& qubit : synthesis_info.qubits) {
        for (auto const& other_qubit : synthesis_info.qubits) {
            if (qubit == other_qubit) continue;
            if (device.is_adjacent(qubit, other_qubit)) {
                device_subgraph.add_edge(qubit, other_qubit, apsp.distance[qubit][other_qubit]);
            }
        }
    }

    return device_subgraph;
}

void append_mst_node(
    dvlab::Digraph<size_t, float> const& mst,
    size_t v,
    std::unordered_set<size_t> const* ancilla_qubits,
    std::string const& prefix,
    bool is_last,
    std::string& out) {
    auto const branch_char = is_last ? "└── " : "├── ";
    out += prefix + branch_char;

    out += fmt::format("q{}", v);
    if (ancilla_qubits && ancilla_qubits->contains(v)) {
        out += " (ancilla)";
    }
    out += "\n";

    std::vector<size_t> children_vec(mst.out_neighbors(v).begin(),
                                     mst.out_neighbors(v).end());
    std::ranges::sort(children_vec);

    std::string child_prefix = prefix + (is_last ? "    " : "│   ");
    for (size_t i = 0; i < children_vec.size(); ++i) {
        append_mst_node(mst, children_vec[i], ancilla_qubits, child_prefix,
                        i == children_vec.size() - 1, out);
    }
}

std::string mst_to_string(
    dvlab::Digraph<size_t, float> const& mst,
    size_t root,
    std::unordered_set<size_t> const* ancilla_qubits = nullptr) {
    std::string out;
    out += fmt::format("q{}", root);
    if (ancilla_qubits && ancilla_qubits->contains(root)) {
        out += " (ancilla)";
    }
    out += "\n";

    std::vector<size_t> children_vec(mst.out_neighbors(root).begin(),
                                     mst.out_neighbors(root).end());
    std::ranges::sort(children_vec);

    for (size_t i = 0; i < children_vec.size(); ++i) {
        append_mst_node(mst, children_vec[i], ancilla_qubits, "",
                        i == children_vec.size() - 1, out);
    }
    return out;
}

/** Maps fermion tree indices (and ancilla physical ids) to dense logical QCir lines. */
struct LogicalQubitLayout {
    std::vector<size_t> physical_qubits;
    std::unordered_map<size_t, size_t> physical_to_logical;

    explicit LogicalQubitLayout(TernaryTree const& tree) {
        auto const n = tree.num_qubits();
        physical_qubits.resize(n);
        for (size_t i = 0; i < n; ++i) {
            auto const physical = as_qubit_node_or_throw(tree.get_node_by_index(i))->qubit_label.value();
            physical_qubits[i]  = physical;
            physical_to_logical.emplace(physical, i);
        }
    }

    size_t num_qubits() const { return physical_qubits.size(); }

    size_t to_logical(size_t physical) const { return physical_to_logical.at(physical); }

    void register_ancilla(size_t physical) {
        if (physical_to_logical.contains(physical)) {
            return;
        }
        auto const logical = physical_qubits.size();
        physical_qubits.push_back(physical);
        physical_to_logical.emplace(physical, logical);
    }
};

void collect_ancillas_for_hamiltonian(
    QubitHamiltonian const& qubit_hamiltonian,
    TernaryTree const& tree,
    device::APSPResult const& apsp,
    device::Device const& device,
    LogicalQubitLayout& layout) {
    for (auto const& term : qubit_hamiltonian) {
        if (term.pauli_product().is_identity()) {
            continue;
        }
        auto const synthesis_info = form_connected_components(term.pauli_product(), tree, apsp, device);
        for (size_t i = 0; i < synthesis_info.qubits.size(); ++i) {
            if (synthesis_info.is_ancilla[i]) {
                layout.register_ancilla(synthesis_info.qubits[i]);
            }
        }
    }
}

void synthesize_term(
    tableau::PauliRotation const& rotation,
    TernaryTree const& tree,
    device::APSPResult const& apsp,
    device::Device const& device,
    qcir::QCir& qcir,
    LogicalQubitLayout const* layout) {
    auto const& pauli_product = rotation.pauli_product();

    if (pauli_product.is_identity()) {
        return;
    }

    auto const synthesis_info = form_connected_components(pauli_product, tree, apsp, device);

    auto const device_subgraph = form_device_subgraph(synthesis_info, device, apsp);

    // Build set of ancilla qubit IDs (is_ancilla is parallel to qubits, not indexed by qubit ID)
    auto ancilla_qubits = std::unordered_set<size_t>{};
    for (size_t i = 0; i < synthesis_info.qubits.size(); ++i) {
        if (synthesis_info.is_ancilla[i]) {
            ancilla_qubits.insert(synthesis_info.qubits[i]);
        }
    }

    auto const mst_cost_fn = [&](auto const& e) {
        auto const [src, dst] = e;

        // a cnot (dst, src) is needed to connect the two qubits
        // if src is an ancilla qubit, we also need to synthesize a cnot (src, dst)
        if (ancilla_qubits.contains(src)) {
            return apsp.distance[dst][src] + apsp.distance[src][dst];
        }
        return apsp.distance[dst][src];
    };
    auto const [mst, root] =
        dvlab::minimum_spanning_arborescence_with_cost(device_subgraph, mst_cost_fn);

    spdlog::debug("MST for term {}:\n{}", rotation.to_string(), mst_to_string(mst, root, &ancilla_qubits));

    std::vector<size_t> post_order_traversal;
    std::stack<size_t> stack;

    stack.push(root);
    while (!stack.empty()) {
        auto const v = stack.top();
        stack.pop();
        post_order_traversal.push_back(v);
        for (auto const& n : mst.out_neighbors(v)) {
            stack.push(n);
        }
    }

    std::ranges::reverse(post_order_traversal);

    auto const num_qubits = layout ? layout->num_qubits() : device.get_num_qubits();
    auto const to_line    = [&](size_t physical) -> size_t {
        return layout ? layout->to_logical(physical) : physical;
    };

    qcir::QCir conjugation_qcir(num_qubits);

    // conjugate by V and H gates to put all Pauli letters to Z
    for (size_t i = 0; i < pauli_product.n_qubits(); ++i) {
        if (pauli_product.is_i(i)) {
            continue;
        }

        auto const qubit_line = layout ? i : as_qubit_node_or_throw(tree.get_node_by_index(i))->qubit_label.value();
        if (pauli_product.is_x(i)) {
            conjugation_qcir.append(qcir::HGate(), {qubit_line});
        }
        if (pauli_product.is_y(i)) {
            conjugation_qcir.append(qcir::SXGate(), {qubit_line});
        }
    }

    // build CX sequence according to the post-order traversal
    for (auto const& dst : post_order_traversal) {
        if (mst.in_degree(dst) == 0) {
            continue;
        }
        auto const src = *mst.in_neighbors(dst).begin();
        // The supplementary mat'l of Treespilation is incorrect here.
        // It states to add this gate after the main CX gate but it should be
        // before.
        if (ancilla_qubits.contains(src)) {
            conjugation_qcir.append(qcir::CXGate(), {to_line(src), to_line(dst)});
        }
        conjugation_qcir.append(qcir::CXGate(), {to_line(dst), to_line(src)});
    }
    qcir.compose(conjugation_qcir);

    qcir.append(qcir::PZGate(rotation.phase()), {to_line(root)});
    conjugation_qcir.adjoint_inplace();
    qcir.compose(conjugation_qcir);
}

}  // namespace

tl::expected<TreespileResult, TreespileFailReason>
treespile(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    double time,
    size_t n_trotterization_steps,
    device::APSPCostFnType const& cost_fn,
    bool optimize1,
    bool optimize2,
    bool exhaustive,
    bool use_logical_indices) {
    //
    using FailReason = TreespileFailReason;

    if (n_trotterization_steps == 0) {
        return tl::unexpected(FailReason::invalid_n_trotterization_steps);
    }

    auto const n_modes = hamiltonian.n_modes();
    if (n_modes == 0) {
        return tl::unexpected(FailReason::empty_hamiltonian);
    }

    if (n_modes > device.get_num_qubits()) {
        return tl::unexpected(FailReason::device_too_small);
    }

    // generate a ternary tree

    auto const apsp = floyd_warshall(device, cost_fn);

    auto tree = exhaustive
                    ? build_bonsai_ternary_tree_exhaustive(device, apsp, n_modes)
                    : build_bonsai_ternary_tree(device, apsp, n_modes);

    if (!tree.has_value()) {
        // NOTE: it's still possible for bonsai to fail because the device might
        // be disconnected.
        return tl::unexpected(FailReason::tt_build_failed_not_enough_qubits);
    }

    if (optimize1) {
        tree = pauli_weight_optimize_mapping(*tree, hamiltonian, &device);
    } else if (optimize2) {
        tree = cnot_proxy_optimize_mapping(*tree, hamiltonian, &device);
    }

    auto const mapping = TernaryTreeMapping(tree.value());

    auto const qubit_hamiltonian = qubitize(hamiltonian, mapping);

    std::optional<LogicalQubitLayout> layout;
    if (use_logical_indices) {
        layout.emplace(tree.value());
        collect_ancillas_for_hamiltonian(qubit_hamiltonian, tree.value(), apsp, device, *layout);
    }

    qcir::QCir qcir(use_logical_indices ? layout->num_qubits() : device.get_num_qubits());

    // if all terms are commutative, fix trotter steps to 1
    if (is_all_commutative(qubit_hamiltonian)) {
        n_trotterization_steps = 1;
    }
    auto const dt                        = time / static_cast<double>(n_trotterization_steps);
    LogicalQubitLayout const* layout_ptr = layout ? &*layout : nullptr;

    auto const prtabl = trotterize_single_step(qubit_hamiltonian, dt);

    for (auto const& rotation : prtabl) {
        synthesize_term(rotation, tree.value(), apsp, device, qcir, layout_ptr);
    }

    if (n_trotterization_steps > 1) {
        auto copy_qcir = qcir;
        for (size_t i = 1; i < n_trotterization_steps; ++i) {
            qcir.compose(copy_qcir);
        }
    }

    TreespileResult result;
    result.circuit         = std::move(qcir);
    result.physical_qubits = use_logical_indices ? layout->physical_qubits : std::vector<size_t>{};
    result.encoding        = std::make_unique<TernaryTreeMapping>(std::move(tree.value()));
    return result;
}

}  // namespace qsyn::hamiltonian
