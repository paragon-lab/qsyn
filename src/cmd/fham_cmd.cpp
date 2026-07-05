/****************************************************************************
  PackageName  [ hamiltonian ]
  Synopsis     [ Define fermionic Hamiltonian commands ]
  Author       [ Design Verification Lab ]
****************************************************************************/

#include "./fham_cmd.hpp"

#include <filesystem>
#include <string>

#include "argparse/arg_parser.hpp"
#include "cli/cli.hpp"
#include "cmd/device_mgr.hpp"
#include "cmd/fham_mgr.hpp"
#include "device/device_analysis.hpp"
#include "device/ibmq_devices.hpp"
#include "hamiltonian/f2q_majorana.hpp"
#include "hamiltonian/f2q_mappings.hpp"
#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "hamiltonian/fham_workspace.hpp"
#include "hamiltonian/qubit_hamiltonian.hpp"
#include "hamiltonian/ternarytree/bonsai.hpp"
#include "hamiltonian/ternarytree/proxy_eval.hpp"
#include "hamiltonian/ternarytree/ternary_tree.hpp"
#include "hamiltonian/ternarytree/tree_optimizations.hpp"
#include "hamiltonian/ternarytree/treespile.hpp"
#include "qcir/qcir.hpp"
#include "util/data_structure_manager_common_cmd.hpp"

using namespace dvlab::argparse;

namespace qsyn::hamiltonian {

namespace {

struct FhamQubitizeOutcome {
    QubitHamiltonian q_ham;
    bool used_workspace_encoding{false};
};

TernaryTree optimize_ternary_tree_mapping(
    TernaryTree tree,
    FermionHamiltonian const& f_ham,
    device::DeviceMgr const& device_mgr,
    bool optimize1,
    bool optimize2) {
    device::Device const* dev_ptr = nullptr;
    if (!device_mgr.empty() && device_mgr.get()->get_num_qubits() >= f_ham.n_modes()) {
        dev_ptr = device_mgr.get();
    }
    if (optimize1 && optimize2) {
        fmt::println("Warning: Both -o1 and -o2 specified. Defaulting to -o2 (fidelity proxy).");
    }
    if (optimize2) {
        fmt::println("Optimizing ternary tree mapping to minimize proxy fidelity...");
        return infidelity_proxy_optimize_mapping(tree, f_ham, dev_ptr);
    }
    if (optimize1) {
        fmt::println("Optimizing ternary tree mapping to minimize Pauli weight...");
        return pauli_weight_optimize_mapping(tree, f_ham, dev_ptr);
    }
    return tree;
}

FhamQubitizeOutcome qubitize_fham_workspace(
    FhamWorkspace& workspace,
    device::DeviceMgr const& device_mgr,
    bool strategy_explicit,
    std::string const& strategy_str,
    bool optimize1,
    bool optimize2) {
    auto const& f_ham = workspace.hamiltonian;

    if (!strategy_explicit && workspace.encoding != nullptr) {
        if (auto* tt_enc = dynamic_cast<TernaryTreeMapping*>(workspace.encoding.get())) {
            if (optimize1 || optimize2) {
                auto tree          = optimize_ternary_tree_mapping(tt_enc->tree(), f_ham, device_mgr, optimize1, optimize2);
                workspace.encoding = std::make_unique<TernaryTreeMapping>(std::move(tree));
            }
        } else if (optimize1 || optimize2) {
            fmt::println("Warning: -o1/-o2 only apply to ternary tree encodings; ignoring.");
        }

        return {qubitize(f_ham, *workspace.encoding), true};
    }

    auto const strategy = dvlab::str::tolower_string(strategy_explicit ? strategy_str : "jw");
    if (dvlab::str::is_prefix_of(strategy, "jw")) {
        workspace.encoding = std::make_unique<JordanWignerMapping>(f_ham.n_modes());
        return {qubitize(f_ham, *workspace.encoding), false};
    }

    std::optional<TernaryTree> initial_tree = std::nullopt;
    device::Device const* dev_ptr           = nullptr;

    if (!device_mgr.empty()) {
        dev_ptr = device_mgr.get();
        if (dev_ptr->get_num_qubits() < f_ham.n_modes()) {
            fmt::println("Warning: Device ({} qubits) is too small for Hamiltonian ({} modes).", dev_ptr->get_num_qubits(), f_ham.n_modes());
            fmt::println("Using standard ternary tree...");
            dev_ptr = nullptr;
        } else {
            fmt::println("Using device topology to build ternary tree.");

            auto apsp        = device::floyd_warshall(*dev_ptr, device::default_floyd_warshall_cost);
            auto tree_result = build_bonsai_ternary_tree(*dev_ptr, apsp, f_ham.n_modes());

            if (tree_result.has_value()) {
                initial_tree = std::move(tree_result.value());
            } else {
                spdlog::warn("Failed to build Bonsai tree (device too small/disconnected?). Using standard ternary tree.");
            }
        }
    }

    if (!initial_tree.has_value()) {
        if (device_mgr.empty()) {
            fmt::println("No device. Using standard ternary tree.");
        }
        initial_tree = TernaryTree(f_ham.n_modes());
    }

    auto tree          = optimize_ternary_tree_mapping(std::move(initial_tree.value()), f_ham, device_mgr, optimize1, optimize2);
    workspace.encoding = std::make_unique<TernaryTreeMapping>(std::move(tree));
    return {qubitize(f_ham, *workspace.encoding), false};
}

dvlab::Command fham_read_cmd(FermionHamiltonianMgr& fham_mgr) {
    return dvlab::Command(
        "read",
        [](ArgumentParser& parser) {
            parser.description("Read a fermionic Hamiltonian from a text file");

            parser.add_argument<std::string>("filepath")
                .help(
                    "The path to the input file. "
                    "File format: '(re, im) ops' per line, e.g. '(0.5, 0.5) 0^ 1'");
        },
        [&](ArgumentParser const& parser) {
            auto const filepath = std::filesystem::path(parser.get<std::string>("filepath"));

            auto ferm_opt = read_fermionic_hamiltonian(filepath);
            if (!ferm_opt.has_value()) {
                return dvlab::CmdExecResult::error;
            }

            size_t new_id = fham_mgr.get_next_id();
            fham_mgr.add(new_id, std::make_unique<FhamWorkspace>(std::move(*ferm_opt)));
            fham_mgr.set_filename(std::filesystem::path{filepath}.stem().string());

            return dvlab::CmdExecResult::done;
        });
}

dvlab::Command fham_qubitize_cmd(FermionHamiltonianMgr& fham_mgr, QubitHamiltonianMgr& qbham_mgr, device::DeviceMgr& device_mgr) {
    return dvlab::Command(
        "qubitize",
        [](ArgumentParser& parser) {
            parser.description(
                "Transform the focused fermionic Hamiltonian to a qubit Hamiltonian");

            parser.add_argument<std::string>("-s", "--strategy")
                .constraint(choices_allow_prefix({"jw", "ternary_tree"}))
                .help(
                    "Fermion-to-qubit mapping strategy: 'jw' or 'ternary_tree'. "
                    "If omitted, uses the workspace encoding when present, otherwise Jordan-Wigner");

            parser.add_argument<bool>("-o1", "--optimize1")
                .action(store_true)
                .help("Run simulated annealing to minimize Pauli weight (only applies to ternary_tree strategy)");

            parser.add_argument<bool>("-o2", "--optimize2")
                .action(store_true)
                .help("Run simulated annealing to minimize proxy fidelity (only applies to ternary_tree strategy)");
        },
        [&](ArgumentParser const& parser) {
            if (!dvlab::utils::mgr_has_data(fham_mgr)) {
                return dvlab::CmdExecResult::error;
            }

            auto* workspace = fham_mgr.get();

            bool const strategy_explicit = parser.parsed("--strategy");
            bool optimize1               = parser.get<bool>("--optimize1");
            bool optimize2               = parser.get<bool>("--optimize2");

            auto const outcome = qubitize_fham_workspace(
                *workspace,
                device_mgr,
                strategy_explicit,
                strategy_explicit ? parser.get<std::string>("--strategy") : std::string{},
                optimize1,
                optimize2);

            size_t id = qbham_mgr.get_next_id();
            qbham_mgr.add(id, std::make_unique<QubitHamiltonian>(std::move(outcome.q_ham)));
            qbham_mgr.set_filename(fham_mgr.get_filename());
            qbham_mgr.add_procedures(fham_mgr.get_procedures());

            std::string proc_name;
            if (outcome.used_workspace_encoding) {
                proc_name = dynamic_cast<TernaryTreeMapping const*>(workspace->encoding.get()) != nullptr
                                ? "fham_qubitize_ternary_tree"
                                : "fham_qubitize_jw";
            } else {
                auto const strategy_str = strategy_explicit ? parser.get<std::string>("--strategy") : "jw";
                auto const strategy     = dvlab::str::tolower_string(strategy_str);
                proc_name               = dvlab::str::is_prefix_of(strategy, "ternary_tree")
                                              ? "fham_qubitize_ternary_tree"
                                              : "fham_qubitize_jw";
            }
            if ((optimize1 || optimize2) &&
                dynamic_cast<TernaryTreeMapping const*>(workspace->encoding.get()) != nullptr) {
                proc_name += "_optimized";
            }
            qbham_mgr.add_procedure(proc_name);

            fmt::println("Transformed focused fermionic Hamiltonian to QubitHamiltonian with ID: {}", id);

            return dvlab::CmdExecResult::done;
        });
}

}  // namespace

namespace {

void print_encoding_summary(FhamWorkspace const& workspace) {
    if (workspace.encoding == nullptr) {
        fmt::println("Encoding: none");
        return;
    }
    if (dynamic_cast<JordanWignerMapping const*>(workspace.encoding.get()) != nullptr) {
        fmt::println("Encoding: Jordan-Wigner ({} modes)", workspace.encoding->n_modes());
    } else if (dynamic_cast<TernaryTreeMapping const*>(workspace.encoding.get()) != nullptr) {
        fmt::println("Encoding: ternary tree ({} modes)", workspace.encoding->n_modes());
    } else {
        fmt::println("Encoding: present ({} modes)", workspace.encoding->n_modes());
    }
}

void print_encoding_detail(FhamWorkspace const& workspace) {
    if (workspace.encoding == nullptr) {
        fmt::println("No encoding attached to this workspace.");
        return;
    }
    if (auto const* jw = dynamic_cast<JordanWignerMapping const*>(workspace.encoding.get())) {
        fmt::println("Jordan-Wigner mapping ({} modes)", jw->n_modes());
        fmt::println("Clifford: identity on {} qubits", jw->n_modes());
    } else if (auto const* tt = dynamic_cast<TernaryTreeMapping const*>(workspace.encoding.get())) {
        fmt::println("Ternary tree mapping ({} modes)", tt->n_modes());
        fmt::print("{}", format_logical_to_physical_mapping(tt->tree()));
        fmt::println("{}", to_string(tt->tree()));
    } else {
        fmt::println("Unknown encoding type ({} modes)", workspace.encoding->n_modes());
    }
}

dvlab::Command fham_bonsai_cmd(device::DeviceMgr& device_mgr, FermionHamiltonianMgr& fham_mgr) {
    return dvlab::Command(
        "bonsai",
        [](ArgumentParser& parser) {
            parser.description(
                "Build a Bonsai ternary tree on the current device for the focused "
                "fermionic Hamiltonian and attach it as the workspace encoding");
            parser.add_argument<size_t>("-r", "--root-qubit-id")
                .help("ID of the qubit to root the tree at. If not specified, a center of the device coupling graph will be the root.");
            parser.add_argument<bool>("-e", "--exhaustive")
                .action(store_true)
                .help("Exhaustively search for the best ternary tree, stemming from all qubits. This flag is ignored if --root-qubit-id is specified.");
            parser.add_argument<std::string>("--cost-fn")
                .constraint(choices_allow_prefix({"log_success_rate", "default"}))
                .default_value("default")
                .help("cost function for Floyd-Warshall (used to pick tree center and order)");
        },
        [&](ArgumentParser const& parser) {
            if (device_mgr.empty()) {
                spdlog::error("No device loaded. Read or fetch a device first.");
                return dvlab::CmdExecResult::error;
            }

            if (!dvlab::utils::mgr_has_data(fham_mgr)) {
                spdlog::error("No fermionic Hamiltonian loaded. Read or create one first.");
                return dvlab::CmdExecResult::error;
            }

            auto* workspace        = fham_mgr.get();
            auto const& f_ham      = workspace->hamiltonian;
            auto const n_qubits    = f_ham.n_modes();
            auto const& device     = *device_mgr.get();
            auto const cost_fn_str = parser.get<std::string>("--cost-fn");
            auto cost_fn           = device::default_floyd_warshall_cost;
            if (dvlab::str::is_prefix_of(dvlab::str::tolower_string(cost_fn_str), "log_success_rate")) {
                cost_fn = device::log_success_rate_floyd_warshall_cost;
            }
            bool exhaustive = parser.get<bool>("--exhaustive");

            if (device.get_num_qubits() < n_qubits) {
                spdlog::error(
                    "Device ({} qubits) is too small for Hamiltonian ({} modes).",
                    device.get_num_qubits(),
                    n_qubits);
                return dvlab::CmdExecResult::error;
            }

            auto const apsp = device::floyd_warshall(device, cost_fn);
            auto tree       = [&]() {
                if (parser.parsed("--root-qubit-id")) {
                    auto const root_qubit_id = parser.get<size_t>("--root-qubit-id");
                    return build_bonsai_ternary_tree(root_qubit_id, device, apsp, n_qubits);
                }
                if (exhaustive) {
                    return build_bonsai_ternary_tree_exhaustive(device, apsp, n_qubits);
                }
                return build_bonsai_ternary_tree(device, apsp, n_qubits);
            }();

            if (!tree.has_value()) {
                switch (tree.error()) {
                    case BonsaiFailReason::not_enough_qubits:
                        spdlog::error("Failed to build Bonsai ternary tree: not enough qubits in the device");
                        spdlog::error("(Potentially disconnected device?)");
                        break;
                    case BonsaiFailReason::invalid_root_qubit:
                        spdlog::error("Failed to build Bonsai ternary tree: invalid root qubit");
                        break;
                }
                return dvlab::CmdExecResult::error;
            }

            workspace->encoding = std::make_unique<TernaryTreeMapping>(std::move(tree.value()));

            return dvlab::CmdExecResult::done;
        });
}

dvlab::Command fham_print_cmd(FermionHamiltonianMgr const& fham_mgr) {
    return dvlab::Command{
        "print",
        [](ArgumentParser& parser) {
            parser.description("Print the focused fermionic Hamiltonian");
            parser.add_argument<bool>("-v", "--verbose")
                .action(store_true)
                .help(
                    "with --encoding: list JW Majorana Paulis and C P C† images; "
                    "otherwise list each fermionic term");
            parser.add_argument<bool>("--encoding")
                .action(store_true)
                .help("display the workspace F2Q encoding in detail");
        },
        [&](ArgumentParser const& parser) {
            if (!dvlab::utils::mgr_has_data(fham_mgr)) {
                return dvlab::CmdExecResult::error;
            }

            auto const* workspace = fham_workspace(fham_mgr);
            auto const* f_ham     = &workspace->hamiltonian;
            fmt::println("Fermionic Hamiltonian ({} modes, {} terms)",
                         f_ham->n_modes(),
                         f_ham->get_terms().size());
            print_encoding_summary(*workspace);

            if (parser.parsed("--encoding")) {
                print_encoding_detail(*workspace);
                if (parser.parsed("--verbose")) {
                    if (workspace->encoding == nullptr) {
                        spdlog::error(
                            "No encoding on focused FHam. Run `fham qubitize` or `fham treespile` first.");
                        return dvlab::CmdExecResult::error;
                    }
                    print_majorana_encoding_correspondence(*workspace->encoding);
                }
                return dvlab::CmdExecResult::done;
            }

            if (!parser.parsed("--verbose")) {
                return dvlab::CmdExecResult::done;
            }
            auto const& terms = f_ham->get_terms();
            for (std::size_t i = 0; i < terms.size(); ++i) {
                auto const& [coeff, ops] = terms[i];
                fmt::print("  Term {}: ({}, {})  ", i, coeff.real(), coeff.imag());
                for (auto const& [mode, is_creation] : ops) {
                    fmt::print("{}{}", mode, is_creation ? "^ " : " ");
                }
                fmt::println("");
            }

            return dvlab::CmdExecResult::done;
        }};
}

dvlab::Command fham_treespile_cmd(
    device::DeviceMgr& device_mgr,
    FermionHamiltonianMgr& fham_mgr,
    qcir::QCirMgr& qcir_mgr) {
    return dvlab::Command(
        "treespile",
        [](ArgumentParser& parser) {
            parser.description(
                "Apply treespile mapping from the focused fermionic Hamiltonian on the currently loaded device");
            parser.add_argument<double>("time")
                .help("Time for the Hamiltonian to evolve for");
            parser.add_argument<size_t>("n-steps")
                .help("Number of trotterization steps to apply");
            parser.add_argument<std::string>("--cost-fn")
                .constraint(choices_allow_prefix({"log_success_rate", "default"}))
                .default_value("default")
                .help("cost function for Floyd-Warshall used inside treespile");
            parser.add_argument<bool>("-o1", "--optimize1")
                .action(store_true)
                .help("Run simulated annealing to optimize tree for Pauli weight");
            parser.add_argument<bool>("-o2", "--optimize2")
                .action(store_true)
                .help("Run simulated annealing to optimize tree for proxy fidelity (requires device)");
            parser.add_argument<bool>("-e", "--exhaustive")
                .action(store_true)
                .help("Exhaustively search for the best ternary tree, stemming from all qubits");
            parser.add_argument<bool>("--logical-index")
                .action(store_true)
                .help(
                    "Label QCir qubits by fermion tree index (0..n-1) instead of physical device "
                    "qubit id; checkout an induced subdevice on the device manager");
        },
        [&](ArgumentParser const& parser) {
            if (device_mgr.empty()) {
                spdlog::error("No device loaded. Read or fetch a device first.");
                return dvlab::CmdExecResult::error;
            }

            if (!dvlab::utils::mgr_has_data(fham_mgr)) {
                spdlog::error("No fermionic Hamiltonian loaded. Read or create one first.");
                return dvlab::CmdExecResult::error;
            }

            auto const& device = *device_mgr.get();
            auto const& f_ham  = fham_mgr.get()->hamiltonian;

            auto const n_steps = parser.get<size_t>("n-steps");
            if (n_steps == 0) {
                spdlog::error("Number of trotterization steps must be greater than 0");
                return dvlab::CmdExecResult::error;
            }

            auto const time        = parser.get<double>("time");
            auto const cost_fn_str = parser.get<std::string>("--cost-fn");
            bool optimize1         = parser.get<bool>("--optimize1");
            bool optimize2         = parser.get<bool>("--optimize2");
            bool exhaustive        = parser.get<bool>("--exhaustive");
            bool use_logical_index = parser.get<bool>("--logical-index");
            auto cost_fn           = device::default_floyd_warshall_cost;
            if (dvlab::str::is_prefix_of(dvlab::str::tolower_string(cost_fn_str), "log_success_rate")) {
                cost_fn = device::log_success_rate_floyd_warshall_cost;
            }

            auto result = treespile(
                f_ham, device, time, n_steps, cost_fn, optimize1, optimize2, exhaustive, use_logical_index);

            if (!result.has_value()) {
                auto const reason = result.error();
                switch (reason) {
                    case TreespileFailReason::empty_hamiltonian:
                        spdlog::error("treespile failed: empty Hamiltonian");
                        break;
                    case TreespileFailReason::device_too_small:
                        spdlog::error("treespile failed: device too small");
                        break;
                    case TreespileFailReason::tt_build_failed_not_enough_qubits:
                        spdlog::error("treespile failed: could not build ternary tree (not enough qubits / disconnected)");
                        break;
                    case TreespileFailReason::invalid_n_trotterization_steps:
                        spdlog::error("treespile failed: invalid number of trotterization steps");
                        break;
                }
                return dvlab::CmdExecResult::error;
            }

            auto& treespile_out      = result.value();
            fham_mgr.get()->encoding = std::move(treespile_out.encoding);
            auto circuit             = std::move(treespile_out.circuit);
            auto const new_id        = qcir_mgr.get_next_id();

            qcir_mgr.add(new_id, std::make_unique<qcir::QCir>(std::move(circuit)));
            qcir_mgr.set_filename(fham_mgr.get_filename());
            qcir_mgr.add_procedures(fham_mgr.get_procedures());
            qcir_mgr.add_procedure(use_logical_index ? "fham_treespile_logical" : "fham_treespile");

            device::Device const* error_device = &device;
            if (use_logical_index) {
                auto const* parent_ibmq = dynamic_cast<device::IBMQDevice const*>(&device);
                if (parent_ibmq == nullptr || !parent_ibmq->jsons.has_value()) {
                    spdlog::error(
                        "treespile --logical-index requires an IBM device with JSON "
                        "(run `device fetch` on the backend first)");
                    return dvlab::CmdExecResult::error;
                }

                auto const& physical_qubits = treespile_out.physical_qubits;
                auto sub                    = device.induced_subdevice(physical_qubits);
                if (!sub.has_value()) {
                    switch (sub.error()) {
                        case device::InducedSubdeviceError::duplicate_qubit_id:
                            spdlog::error("treespile: duplicate physical qubit in induced subdevice");
                            break;
                        case device::InducedSubdeviceError::unknown_qubit_id:
                            spdlog::error("treespile: unknown physical qubit in induced subdevice");
                            break;
                    }
                    return dvlab::CmdExecResult::error;
                }

                auto sliced_json = device::slice_ibmq_device_jsons(*parent_ibmq->jsons, physical_qubits);
                if (!sliced_json.has_value()) {
                    switch (sliced_json.error()) {
                        case device::SliceIBMQDeviceError::duplicate_qubit_id:
                            spdlog::error("treespile: duplicate physical qubit when slicing IBM JSON");
                            break;
                        case device::SliceIBMQDeviceError::unknown_qubit_id:
                            spdlog::error("treespile: unknown physical qubit when slicing IBM JSON");
                            break;
                    }
                    return dvlab::CmdExecResult::error;
                }

                auto const sub_id = device_mgr.get_next_id();
                device_mgr.add(
                    sub_id,
                    std::make_unique<device::IBMQDevice>(
                        device::make_ibmq_subdevice(*parent_ibmq, std::move(sub->device), std::move(*sliced_json))));
                device_mgr.checkout(sub_id);
                error_device = device_mgr.get();
                fmt::println(
                    "Checked out induced subdevice ({} qubits, physical [{}]) as device {}",
                    error_device->get_num_qubits(),
                    fmt::join(physical_qubits, ", "),
                    sub_id);
            }

            // go through the gates in the circuits and collect the errors of the 2-qubit gates
            std::vector<float> two_qubit_gate_errors;

            for (auto const& gate : qcir_mgr.get()->get_gates()) {
                if (gate->get_num_qubits() == 2) {
                    auto const& qubits    = gate->get_qubits();
                    auto const& gate_info = error_device->get_gate_info(device::Device::QubitPair{qubits[0], qubits[1]});
                    assert(!gate_info.empty());
                    two_qubit_gate_errors.push_back(gate_info[0].error);
                }
            }

            // report geomean error rates of the 2-qubit gates
            float geomean_success_rate = 1.0f;
            for (auto const& error : two_qubit_gate_errors) {
                geomean_success_rate *= (1.0f - error);
            }
            geomean_success_rate = std::pow(geomean_success_rate, 1.0f / static_cast<float>(two_qubit_gate_errors.size()));
            fmt::println("Geomean error rate of the 2-qubit gates: {}", 1.0f - geomean_success_rate);

            return dvlab::CmdExecResult::done;
        });
}

// NEW
dvlab::Command fham_eval_cmd(
    device::DeviceMgr& device_mgr,
    FermionHamiltonianMgr& fham_mgr) {
    return dvlab::Command(
        "eval-proxy",
        [](ArgumentParser& parser) {
            parser.description("Evaluate the proxy cost function by generating random tree mappings.");
            parser.add_argument<std::string>("-d", "--output-dir")
                .required(true)
                .help("Directory for proxy evaluation outputs (CSV and sample QASM files)");
            parser.add_argument<size_t>("-s", "--samples")
                .default_value(10)
                .help("Number of random trees to sample");
            parser.add_argument<std::string>("-o", "--output")
                .default_value("proxy_evaluation.csv")
                .help("Output CSV file name within --output-dir");
        },
        [&](ArgumentParser const& parser) {
            if (device_mgr.empty() || !dvlab::utils::mgr_has_data(fham_mgr)) {
                spdlog::error("Please load a device and a fermionic Hamiltonian first.");
                return dvlab::CmdExecResult::error;
            }
            auto const output_dir = std::filesystem::path(parser.get<std::string>("--output-dir"));
            // evaluate_proxy_termwise_depth(
            //     *fham_mgr.get(),
            //     *device_mgr.get(),
            //     output_dir / parser.get<std::string>("--output"),
            //     parser.get<size_t>("--samples")
            // );
            // evaluate_proxy_termwise_fidelity(
            //     *fham_mgr.get(),
            //     *device_mgr.get(),
            //     output_dir / parser.get<std::string>("--output"),
            //     parser.get<size_t>("--samples")
            // );
            evaluate_proxy_cost(
                *fham_hamiltonian(fham_mgr),
                *device_mgr.get(),
                output_dir,
                parser.get<std::string>("--output"),
                parser.get<size_t>("--samples"));
            return dvlab::CmdExecResult::done;
        });
}

}  // namespace

dvlab::Command fham_cmd(
    FermionHamiltonianMgr& fham_mgr,
    QubitHamiltonianMgr& qbham_mgr,
    qcir::QCirMgr& qcir_mgr,
    device::DeviceMgr& device_mgr) {
    auto cmd = dvlab::utils::mgr_root_cmd(fham_mgr);

    cmd.add_subcommand("fham-cmd-group", dvlab::utils::mgr_list_cmd(fham_mgr));
    cmd.add_subcommand("fham-cmd-group", dvlab::utils::mgr_delete_cmd(fham_mgr));
    cmd.add_subcommand("fham-cmd-group", dvlab::utils::mgr_checkout_cmd(fham_mgr));
    cmd.add_subcommand("fham-cmd-group", dvlab::utils::mgr_copy_cmd(fham_mgr));
    cmd.add_subcommand("fham-cmd-group", fham_read_cmd(fham_mgr));
    cmd.add_subcommand("fham-cmd-group", fham_qubitize_cmd(fham_mgr, qbham_mgr, device_mgr));
    cmd.add_subcommand("fham-cmd-group", fham_bonsai_cmd(device_mgr, fham_mgr));
    cmd.add_subcommand("fham-cmd-group", fham_treespile_cmd(device_mgr, fham_mgr, qcir_mgr));
    cmd.add_subcommand("fham-cmd-group", fham_print_cmd(fham_mgr));
    cmd.add_subcommand("fham-cmd-group", fham_eval_cmd(device_mgr, fham_mgr));

    return cmd;
}

bool add_fham_cmds(
    dvlab::CommandLineInterface& cli,
    FermionHamiltonianMgr& fham_mgr,
    QubitHamiltonianMgr& qbham_mgr,
    qcir::QCirMgr& qcir_mgr,
    device::DeviceMgr& device_mgr) {
    if (!cli.add_command(fham_cmd(fham_mgr, qbham_mgr, qcir_mgr, device_mgr))) {
        spdlog::error("Registering \"fham\" commands fails... exiting");
        return false;
    }

    return true;
}

}  // namespace qsyn::hamiltonian
