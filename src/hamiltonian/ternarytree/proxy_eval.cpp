/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Evaluates proxy cost functions for ternary tree optimization ]
  Author       [ April Wang (april864) ]
*/

#include "./proxy_eval.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <stack>
#include <tl/expected.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./bonsai.hpp"
#include "./ternary_tree.hpp"
#include "./tree_rotations.hpp"
#include "./treespile.hpp"
#include "device/device_analysis.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "qcir/basic_gate_type.hpp"
#include "tableau/pauli_rotation.hpp"
#include "tree_optimizations.hpp"

namespace qsyn::hamiltonian {

namespace {

tableau::PauliRotation to_single_step_rotation(HermitianPauliTerm const& term, double dt = 1.0) {
    return tableau::PauliRotation(
        term.pauli_product(),
        dvlab::Phase::to_phase(2 * term.coeff() * dt, 1e-8));
}

void apply_random_tree_rotations(TernaryTree& tree, std::mt19937& rng, int mutations = 20) {
    TreeRotator rotator;
    for (int m = 0; m < mutations; ++m) {
        switch (rng() % 4) {
            case 0:
                rotator.root_change(&tree);
                break;
            case 1:
                rotator.pauli_shuffle(&tree);
                break;
            case 2:
                rotator.mode_association_swap(&tree);
                break;
            default:
                rotator.majorana_braiding_change(&tree);
                break;
        }
    }
}

struct ProxyCostSample {
    double proxy_cost = 0.0;
    qcir::QCir qcir;
};

enum class ProxyCostSampleError : std::uint8_t {
    build_failed,
    not_routable,
};

tl::expected<ProxyCostSample, ProxyCostSampleError> generate_proxy_cost_sample(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    device::APSPResult const& apsp,
    device::APSPResult const& path_cost_apsp,
    std::mt19937& rng) {
    auto base_tree = build_bonsai_ternary_tree(device, apsp, hamiltonian.n_modes());
    if (!base_tree) {
        return tl::unexpected(ProxyCostSampleError::build_failed);
    }

    TernaryTree tree = std::move(base_tree.value());
    apply_random_tree_rotations(tree, rng);

    double const proxy_cost = ApspPathCost(hamiltonian, path_cost_apsp).evaluate(tree);

    auto synthesized = synthesize_hamiltonian_on_tree(
        hamiltonian, tree, device, apsp, /*time=*/1.0, /*n_trotterization_steps=*/1);
    if (!synthesized) {
        return tl::unexpected(ProxyCostSampleError::not_routable);
    }

    return ProxyCostSample{
        .proxy_cost = proxy_cost,
        .qcir       = std::move(synthesized->circuit),
    };
}

}  // namespace

// Evaluates infidelity_cost proxy function to compare the proxy infidelity with the
// actual infidelity
void evaluate_proxy_cost(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    std::filesystem::path const& output_dir,
    std::string const& output_csv_name,
    size_t samples) {
    std::error_code ec;
    if (!std::filesystem::create_directories(output_dir, ec) && !std::filesystem::is_directory(output_dir)) {
        spdlog::error("Failed to create output directory \"{}\": {}", output_dir.string(), ec.message());
        return;
    }

    auto const output_csv    = output_dir / output_csv_name;
    auto const apsp          = floyd_warshall(device, device::default_floyd_warshall_cost);
    auto const path_cost_apsp = floyd_warshall(device, device::log_success_rate_floyd_warshall_cost);

    std::ofstream csv(output_csv, std::ios::app);
    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t i = 0; i < samples; ++i) {
        fmt::println("Starting tree {}", i);
        auto sample = generate_proxy_cost_sample(hamiltonian, device, apsp, path_cost_apsp, rng);
        if (!sample) {
            if (sample.error() == ProxyCostSampleError::build_failed) {
                spdlog::error("Failed to build base tree on sample {}.", i);
            } else {
                --i;  // retry until we get a routable sample
            }
            continue;
        }

        auto const qasm_file = output_dir / fmt::format("sample_{}.qasm", i);
        sample->qcir.write_qasm(qasm_file);

        csv << i << "," << sample->proxy_cost << "\n";
        csv.flush();
    }
}

// For use in evaluate_proxy_termwise_depth
// Keeps array of num_qubits x num_circuit_layers. For each gate in each Hamiltonian
// term, updates array[qubit][layer] to hold the index of the term.
struct DepthArray {
    size_t num_qubits;
    int IDLE = -1;
    std::vector<std::vector<int>> array;

    DepthArray(size_t q) : num_qubits(q), array(q) {}

    void next_gate(const std::vector<size_t>& qubits, int term_id) {
        if (qubits.size() == 1) {
            array[qubits[0]].push_back(term_id);
        } else if (qubits.size() == 2) {
            size_t q0 = qubits[0];
            size_t q1 = qubits[1];

            // Take max time slice between the two qubits
            size_t depth = std::max(array[q0].size(), array[q1].size());

            // Pad other entries
            while (array[q0].size() < depth) array[q0].push_back(IDLE);
            while (array[q1].size() < depth) array[q1].push_back(IDLE);

            array[q0].push_back(term_id);
            array[q1].push_back(term_id);
        }
    }

    // Counts each term's contribution to the overall circuit depth
    std::vector<size_t> get_term_contributions(size_t num_terms) {
        std::vector<size_t> contributions(num_terms, 0);

        // Find last qubit in critical path
        size_t max_depth = 0;
        size_t curr_q    = 0;
        for (size_t q = 0; q < num_qubits; ++q) {
            if (array[q].size() > max_depth) {
                max_depth = array[q].size();
                curr_q    = q;
            }
        }

        // Pad leftover empty spots
        for (size_t q = 0; q < num_qubits; ++q) {
            while (array[q].size() < max_depth) array[q].push_back(IDLE);
        }

        // Fill term contributions vector
        int curr_t = max_depth - 1;
        while (curr_t >= 0) {
            int current_term = array[curr_q][curr_t];

            if (current_term != IDLE) {
                contributions[current_term]++;  // current_term has contributed to overall circuit depth

                // Check if need to switch qubits because of 2-qubit gate
                if (curr_t > 0) {
                    // Find if another qubit shares this gate
                    int other_q = -1;
                    for (size_t q = 0; q < num_qubits; ++q) {
                        if (q != curr_q && array[q][curr_t] == current_term) {
                            other_q = q;
                            break;
                        }
                    }

                    // Switch qubits if this qubit is part of a 2q gate and was idle before this gate
                    if (other_q != -1 && array[curr_q][curr_t - 1] == IDLE) {
                        curr_q = other_q;
                    }
                }
            }
            curr_t--;
        }

        return contributions;
    }
};

// Evaluates infidelity_cost proxy function to determine the depth of each fermionic term's
// contribution to the circuit.
void evaluate_proxy_termwise_depth(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    std::string const& output_csv,
    size_t samples) {
    auto const apsp          = floyd_warshall(device, device::default_floyd_warshall_cost);
    auto const path_cost_apsp = floyd_warshall(device, device::log_success_rate_floyd_warshall_cost);

    std::ofstream csv(output_csv, std::ios::app);

    // Seed randomizer
    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t i = 0; i < samples; ++i) {
        fmt::println("Starting tree {}", i);
        auto base_tree = build_bonsai_ternary_tree(device, apsp, hamiltonian.n_modes());
        if (!base_tree) {
            spdlog::error("Failed to build base tree on sample {}.", i);
            continue;
        }
        TernaryTree tree = std::move(base_tree.value());
        apply_random_tree_rotations(tree, rng);

        // Tree map on hardware
        auto mapping = TernaryTreeMapping(tree);
        auto q_ham   = qubitize(hamiltonian, mapping);
        TreeOracle oracle(tree, path_cost_apsp);

        // Initialize circuit
        qcir::QCir full_qcir(device.get_num_qubits());
        DepthArray deptharray(device.get_num_qubits());
        bool routable = true;

        // To store values from each term (single_term_cnots and cnot_diff count 2-qubit gates)
        struct TermData {
            size_t term_index;
            double proxy;
        };
        std::vector<TermData> sample_terms(q_ham.n_terms());

        size_t term_index = 0;
        for (const auto& term : q_ham) {
            // Calculate proxy cost
            auto const active_nodes  = term.non_identity_qubits();
            double term_proxy        = oracle.get_subtree_weight(active_nodes);
            sample_terms[term_index] = {term_index, term_proxy};

            // Calculate contribution to full circuit
            size_t gates_before = full_qcir.get_gates().size();
            if (!try_synthesize_term(to_single_step_rotation(term), tree, apsp, device, full_qcir)) {
                routable = false;
                break;
            }
            size_t gates_after = full_qcir.get_gates().size();

            auto const& all_gates = full_qcir.get_gates();
            for (size_t g = gates_before; g < gates_after; ++g) {
                deptharray.next_gate(all_gates[g]->get_qubits(), term_index);
            }

            term_index++;
        }

        if (!routable) {
            i--;
            continue;
        }

        std::vector<size_t> term_contributions = deptharray.get_term_contributions(q_ham.n_terms());

        // Write to CSV
        for (size_t t = 0; t < sample_terms.size(); ++t) {
            csv << sample_terms[t].term_index << ","
                << sample_terms[t].proxy << ","
                << term_contributions[t] << "\n";
        }

        csv.flush();
    }
}

// Evaluates infidelity_cost proxy function to determine each fermionic term's
// contribution to the circuit fidelity.
void evaluate_proxy_termwise_fidelity(
    FermionHamiltonian const& hamiltonian,
    device::Device const& device,
    std::string const& output_csv,
    size_t samples) {
    auto const apsp          = floyd_warshall(device, device::default_floyd_warshall_cost);
    auto const path_cost_apsp = floyd_warshall(device, device::log_success_rate_floyd_warshall_cost);

    std::ofstream csv(output_csv, std::ios::app);

    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t i = 0; i < samples; ++i) {
        fmt::println("Starting tree {}", i);
        auto base_tree = build_bonsai_ternary_tree(device, path_cost_apsp, hamiltonian.n_modes());
        if (!base_tree) {
            spdlog::error("Failed to build base tree on sample {}.", i);
            continue;
        }
        TernaryTree tree = std::move(base_tree.value());
        apply_random_tree_rotations(tree, rng);

        TreeOracle oracle(tree, path_cost_apsp);

        // Tree map on hardware
        auto mapping = TernaryTreeMapping(tree);
        auto q_ham   = qubitize(hamiltonian, mapping);

        // Initialize circuit
        qcir::QCir full_qcir(device.get_num_qubits());
        bool routable = true;

        // To store values from each term (single_term_cnots and cnot_diff count 2-qubit gates)
        struct TermData {
            size_t term_index;
            double proxy_log_infidelity;
            double actual_log_infidelity;
        };
        std::vector<TermData> sample_terms(q_ham.n_terms());

        size_t term_index = 0;
        for (const auto& term : q_ham) {
            // Calculate proxy cost
            auto const active_nodes = term.non_identity_qubits();

            // Calculate proxy fidelity
            double term_infidelity = 0.0;
            if (!active_nodes.empty()) {
                term_infidelity = 2.0 * oracle.get_subtree_weight(active_nodes);
            }

            // Calculate actual fidelity
            size_t gates_before = full_qcir.get_gates().size();
            if (!try_synthesize_term(to_single_step_rotation(term), tree, apsp, device, full_qcir)) {
                routable = false;
                break;
            }
            size_t gates_after = full_qcir.get_gates().size();

            double actual_cost    = 0.0;
            auto const& all_gates = full_qcir.get_gates();

            for (size_t g = gates_before; g < gates_after; ++g) {
                auto qubits = all_gates[g]->get_qubits();

                if (qubits.size() == 2) {
                    size_t q0 = qubits[0];
                    size_t q1 = qubits[1];

                    qsyn::device::Device::QubitPair edge{q0, q1};
                    double edge_error = 1.0;

                    // Check the forward direction
                    if (device.is_adjacent(edge)) {
                        auto const& edge_gates = device.get_gate_info(edge);
                        auto const& gate_set   = device.get_gate_set();
                        for (auto const& info : edge_gates) {
                            std::string gate_name = dvlab::str::tolower_string(gate_set[info.gate_idx]);
                            if (gate_name == "cx" || gate_name == "cnot" || gate_name == "ecr" || gate_name == "cz") {
                                edge_error = info.error;
                                break;
                            }
                        }
                    } else {
                        // Check the reverse direction in case the digraph is strictly directional
                        qsyn::device::Device::QubitPair rev_edge{q1, q0};
                        if (device.is_adjacent(rev_edge)) {
                            auto const& edge_gates = device.get_gate_info(rev_edge);
                            auto const& gate_set   = device.get_gate_set();
                            for (auto const& info : edge_gates) {
                                std::string gate_name = dvlab::str::tolower_string(gate_set[info.gate_idx]);
                                if (gate_name == "cx" || gate_name == "cnot" || gate_name == "ecr" || gate_name == "cz") {
                                    edge_error = info.error;
                                    break;
                                }
                            }
                        }
                    }

                    double edge_fidelity = 1.0 - edge_error;
                    actual_cost += -std::log(edge_fidelity);
                }
            }

            sample_terms[term_index] = {term_index, term_infidelity, actual_cost};
            term_index++;
        }

        if (!routable) {
            i--;
            continue;
        }

        // Write to CSV
        for (const auto& td : sample_terms) {
            csv << td.term_index << ","
                << td.proxy_log_infidelity << ","
                << td.actual_log_infidelity << "\n";
        }

        csv.flush();
    }
}

}  // namespace qsyn::hamiltonian
