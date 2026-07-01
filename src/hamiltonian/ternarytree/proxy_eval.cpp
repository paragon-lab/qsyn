/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Evaluates proxy cost functions for ternary tree optimization ]
  Author       [ April Wang (april864) ]
*/

#include "./treespile.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <optional>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>

#include "device/device_analysis.hpp"
#include "./bonsai.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "./ternary_tree.hpp"
#include "./tree_rotations.hpp"
#include "qcir/basic_gate_type.hpp"
#include "util/graph/minimum_spanning_arborescence.hpp"
#include "util/simulated_annealing.hpp"
#include "tree_optimizations.hpp"
#include "./treespile.cpp"

namespace qsyn::hamiltonian {

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

    auto const output_csv = output_dir / output_csv_name;

    auto const apsp = floyd_warshall(device, device::default_floyd_warshall_cost);

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

        // Apply random tree rotation
        TreeRotator rotator;
        for (int m = 0; m < 20; ++m) {
            int mutation_type = rng() % 4; 
            
            if (mutation_type == 0) {
                rotator.root_change(&tree);
            } else if (mutation_type == 1) {
                rotator.pauli_shuffle(&tree);
            } else if (mutation_type == 2) {
                rotator.mode_association_swap(&tree);
            } else if (mutation_type == 3) {
                rotator.majorana_braiding_change(&tree);
            }
        }

        // BEGIN stuff for fidelity_apsp
        auto noise_aware_cost = [&tree, &device](qsyn::device::Device::QubitPair const& edge, qsyn::device::Device const& dev) -> float {
            
            // If edge not in tree, return infinite cost
            if (!tree.has_edge(edge.src, edge.dst) && !tree.has_edge(edge.dst, edge.src)) {
                return std::numeric_limits<float>::infinity();
            }

            auto const& edge_gates = dev.get_gate_info(edge);
            auto const& gate_set = dev.get_gate_set();
            float error_rate = 1.0f;
            
            // Find 2-qubit gate
            for (auto const& info : edge_gates) {
                std::string name = dvlab::str::tolower_string(gate_set[info.gate_idx]);
                if (name == "cx" || name == "cnot" || name == "ecr" || name == "cz") {
                    error_rate = info.error; 
                    break;
                }
            }
            
            // Return log infidelity
            if (error_rate >= 1.0f || error_rate < 0.0f) {
                return std::numeric_limits<float>::infinity();
            }
            return -std::log(1.0f - error_rate);
        };

        auto const fidelity_apsp = floyd_warshall(device, noise_aware_cost);
        // END stuff for fidelity apsp

        // double proxy_cost = fast_tree_cost(tree, hamiltonian, apsp);
        double proxy_cost = infidelity_cost(tree, hamiltonian, fidelity_apsp);

        qcir::QCir qcir(device.get_num_qubits());

        // Check if circuit can be synthesized on hardware
        auto mapping = TernaryTreeMapping(tree);
        auto q_ham = qubitize(hamiltonian, mapping);
        bool routable = true;
        try {
            for (const auto& term : q_ham) {
                synthesize_term(term, tree, apsp, device, 1.0, qcir, nullptr);
            }
        } catch (...) {
            routable = false;
        }
        if (!routable) {
            i--; 
            continue;
        }

        auto const qasm_file = output_dir / fmt::format("sample_{}.qasm", i);
        qcir.write_qasm(qasm_file);

        csv << i << "," << proxy_cost << "\n";
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
        } 
        else if (qubits.size() == 2) {
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
        size_t curr_q = 0;
        for (size_t q = 0; q < num_qubits; ++q) {
            if (array[q].size() > max_depth) {
                max_depth = array[q].size();
                curr_q = q;
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
                contributions[current_term]++; // current_term has contributed to overall circuit depth
                
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

    auto const apsp = floyd_warshall(device, device::default_floyd_warshall_cost);

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

        // Apply random tree rotations
        TreeRotator rotator;
        for (int m = 0; m < 20; ++m) {
            int mutation_type = rng() % 4; 
            
            if (mutation_type == 0) {
                rotator.root_change(&tree);
            } else if (mutation_type == 1) {
                rotator.pauli_shuffle(&tree);
            } else if (mutation_type == 2) {
                rotator.mode_association_swap(&tree);
            } else if (mutation_type == 3) {
                rotator.majorana_braiding_change(&tree);
            }
        }

        // BEGIN stuff for fidelity_apsp
        auto noise_aware_cost = [&tree, &device](qsyn::device::Device::QubitPair const& edge, qsyn::device::Device const& dev) -> float {
            
            // If edge not in tree, return infinite cost
            if (!tree.has_edge(edge.src, edge.dst) && !tree.has_edge(edge.dst, edge.src)) {
                return std::numeric_limits<float>::infinity();
            }

            auto const& edge_gates = dev.get_gate_info(edge);
            auto const& gate_set = dev.get_gate_set();
            float error_rate = 1.0f;
            
            // Find 2-qubit gate
            for (auto const& info : edge_gates) {
                std::string name = dvlab::str::tolower_string(gate_set[info.gate_idx]);
                if (name == "cx" || name == "cnot" || name == "ecr" || name == "cz") {
                    error_rate = info.error; 
                    break;
                }
            }
            
            // Return log infidelity
            if (error_rate >= 1.0f || error_rate < 0.0f) {
                return std::numeric_limits<float>::infinity();
            }
            return -std::log(1.0f - error_rate);
        };

        auto const fidelity_apsp = floyd_warshall(device, noise_aware_cost);

        // Tree map on hardware
        auto mapping = TernaryTreeMapping(tree);
        auto q_ham = qubitize(hamiltonian, mapping);
        TreeOracle oracle(tree, fidelity_apsp);
        
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

        try {
            size_t term_index = 0;
            for (const auto& term : q_ham) {
                // Calculate proxy cost
                std::vector<size_t> active_nodes;
                for (size_t i = 0; i < term.n_qubits(); ++i) {
                    if (!term.is_i(i)) active_nodes.push_back(i);
                }
                double term_proxy = oracle.get_subtree_weight(active_nodes);
                sample_terms[term_index] = {term_index, term_proxy}; 

                // Calculate contribution to full circuit
                size_t gates_before = full_qcir.get_gates().size();
                synthesize_term(term, tree, apsp, device, 1.0, full_qcir, nullptr);
                size_t gates_after = full_qcir.get_gates().size();

                auto const& all_gates = full_qcir.get_gates();
                for (size_t g = gates_before; g < gates_after; ++g) {
                    deptharray.next_gate(all_gates[g]->get_qubits(), term_index);
                }

                term_index++;
            }
        } catch (...) {
            routable = false;
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

    auto const apsp = floyd_warshall(device, device::default_floyd_warshall_cost);

    auto noise_aware_cost = [](qsyn::device::Device::QubitPair const& edge, qsyn::device::Device const& dev) -> float {
        // Fetch available gates for this edge
        auto const& edge_gates = dev.get_gate_info(edge);
        auto const& gate_set = dev.get_gate_set();
        
        float error_rate = 1.0f;

        // Loop through gates on this edge to find the CNOT
        for (auto const& info : edge_gates) {
            std::string gate_name = dvlab::str::tolower_string(gate_set[info.gate_idx]);
            if (gate_name == "cx" || gate_name == "cnot" || gate_name == "ecr" || gate_name == "cz") {
                error_rate = info.error;
                break; 
            }   
        }
        
        if (error_rate >= 1.0f || error_rate < 0.0f) {
            return std::numeric_limits<float>::infinity();
        }
        
        float fidelity = 1.0f - error_rate;
        return -std::log(fidelity); 
    };

    auto const fidelity_apsp = floyd_warshall(device, noise_aware_cost);

    std::ofstream csv(output_csv, std::ios::app);
    
    // Create random tree
    // Seed randomizer
    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t i = 0; i < samples; ++i) {
        fmt::println("Starting tree {}", i);
        auto base_tree = build_bonsai_ternary_tree(device, fidelity_apsp, hamiltonian.n_modes());
        if (!base_tree) {
            spdlog::error("Failed to build base tree on sample {}.", i);
            continue;
        }
        TernaryTree tree = std::move(base_tree.value());

        // Apply random tree rotations
        TreeRotator rotator;
        for (int m = 0; m < 20; ++m) {
            int mutation_type = rng() % 4; 
            
            if (mutation_type == 0) {
                rotator.root_change(&tree);
            } else if (mutation_type == 1) {
                rotator.pauli_shuffle(&tree);
            } else if (mutation_type == 2) {
                rotator.mode_association_swap(&tree);
            } else if (mutation_type == 3) {
                rotator.majorana_braiding_change(&tree);
            }
        }

        // Build tree-constrained apsp
        auto tree_constrained_cost = [&tree, &device](qsyn::device::Device::QubitPair const& edge, qsyn::device::Device const& dev) -> float {
            
            if (!tree.has_edge(edge.src, edge.dst) && !tree.has_edge(edge.dst, edge.src)) {
                return std::numeric_limits<float>::infinity();
            }

            auto const& edge_gates = dev.get_gate_info(edge);
            auto const& gate_set = dev.get_gate_set();
            float error_rate = 1.0f;
            
            for (auto const& info : edge_gates) {
                std::string name = dvlab::str::tolower_string(gate_set[info.gate_idx]);
                if (name == "cx" || name == "cnot" || name == "ecr" || name == "cz") {
                    error_rate = info.error; 
                    break;
                }
            }
            
            if (error_rate >= 1.0f || error_rate < 0.0f) {
                return std::numeric_limits<float>::infinity();
            }
            return -std::log(1.0f - error_rate);
        };

        auto const tree_apsp = floyd_warshall(device, tree_constrained_cost);
        TreeOracle oracle(tree, tree_apsp);

        // Tree map on hardware
        auto mapping = TernaryTreeMapping(tree);
        auto q_ham = qubitize(hamiltonian, mapping);
        
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

        try {
            size_t term_index = 0;
            for (const auto& term : q_ham) {
                // Calculate proxy cost
                std::vector<size_t> active_nodes;
                for (size_t i = 0; i < term.n_qubits(); ++i) {
                    if (!term.is_i(i)) active_nodes.push_back(i);
                }
                
                // Calculate proxy fidelity
                double term_infidelity = 0.0;
                if (!active_nodes.empty()) {
                    term_infidelity = 2.0 * oracle.get_subtree_weight(active_nodes);
                }

                // Calculate actual fidelity
                size_t gates_before = full_qcir.get_gates().size();
                synthesize_term(term, tree, apsp, device, 1.0, full_qcir, nullptr);
                size_t gates_after = full_qcir.get_gates().size();

                double actual_cost = 0.0;
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
                            auto const& gate_set = device.get_gate_set();
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
                                auto const& gate_set = device.get_gate_set();
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
        } catch (...) {
            routable = false;
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

}