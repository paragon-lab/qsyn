/****************************************************************************
  PackageName  [ device ]
  Synopsis     [ Define device package commands ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "./device_cmd.hpp"

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "./device_mgr.hpp"
#include "device/device.hpp"
#include "device/device_analysis.hpp"
#include "device/ibmq_devices.hpp"
#include "hamiltonian/ternarytree/bonsai.hpp"
#include "hamiltonian/ternarytree/ternary_tree.hpp"
#include "qsyn/qsyn_type.hpp"
#include "util/data_structure_manager_common_cmd.hpp"
#include "util/spinner.hpp"
#include "util/sysdep.hpp"
#include "util/tmp_files.hpp"

using namespace dvlab::argparse;
using dvlab::CmdExecResult;

namespace qsyn::device {

std::function<bool(size_t const&)> valid_device_id(qsyn::device::DeviceMgr const& device_mgr) {
    return [&device_mgr](size_t const& id) {
        if (device_mgr.is_id(id)) return true;
        spdlog::error("Device {} does not exist!!", id);
        return false;
    };
};

dvlab::Command device_print_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {
        "print",
        [](ArgumentParser& parser) {
            parser.description("print Device information");
            parser.add_argument<size_t>("ids")
                .nargs(NArgsOption::zero_or_more)
                .help(
                    "if not specified, print basic information about the device;\n"
                    "if one ID is specified, print information about the qubit with the ID; \n"
                    "if two IDs are specified, print information about the adjacency between the two qubits."
                    "An error will be reported if the two qubits are not adjacent.");
            parser.add_argument<bool>("--centers")
                .action(store_true)
                .help("print the centers of the device");
            parser.add_argument<bool>("--connected-components")
                .action(store_true)
                .help("print the connected components of the device");
            parser.add_argument<std::string>("--cost-fn")
                .constraint(choices_allow_prefix({"log_success_rate", "default"}))
                .default_value("default")
                .help("the cost function to use for the Floyd-Warshall algorithm");
        },
        [&device_mgr](ArgumentParser const& parser) {
            auto const cost_fn_str = parser.get<std::string>("--cost-fn");
            auto cost_fn           = default_floyd_warshall_cost;
            if (dvlab::str::is_prefix_of(dvlab::str::tolower_string(cost_fn_str), "log_success_rate")) {
                cost_fn = log_success_rate_floyd_warshall_cost;
            }

            auto apsp = floyd_warshall(*device_mgr.get(), cost_fn);
            if (parser.parsed("--centers")) {
                auto const& device              = *device_mgr.get();
                auto const connected_components = get_connected_components(apsp, device);
                auto const eccentricities       = get_eccentricities(apsp, device);
                for (size_t c = 0; c < connected_components.size(); ++c) {
                    auto const& component = connected_components[c];
                    auto radius           = std::numeric_limits<float>::infinity();
                    for (auto q : component) radius = std::min(radius, eccentricities[q]);
                    std::vector<QubitIdType> centers;
                    for (auto q : component) {
                        if (eccentricities[q] == radius) centers.push_back(q);
                    }
                    fmt::println("Component {} (size {}): centers = {}", c, component.size(), centers);
                }
                return CmdExecResult::done;
            }

            if (parser.parsed("--connected-components")) {
                auto const ids       = parser.get<std::vector<size_t>>("ids");
                auto const filter_fn = [&](QubitIdType const& qubit_id) {
                    if (ids.size() == 0) return true;
                    return dvlab::contains(ids, qubit_id);
                };

                auto const connected_components =
                    get_connected_components(apsp, *device_mgr.get(), filter_fn);
                for (auto const& component : connected_components) {
                    fmt::println("Component (size {}): [{}]",
                                 component.size(), fmt::join(component, ", "));
                }
                return CmdExecResult::done;
            }

            auto const ids = parser.get<std::vector<size_t>>("ids");

            if (ids.size() > 2) {
                spdlog::error("Too many qubit IDs specified. Please specify at most two qubit IDs.");
                return CmdExecResult::error;
            }

            if (ids.size() == 0) {
                fmt::println("{}", device_mgr.get()->info_string());

            } else if (ids.size() == 1) {
                auto gate_info = device_mgr.get()->gate_info_string(ids[0]);
                if (!gate_info.has_value()) {
                    spdlog::error("Qubit {} does not exist", ids[0]);
                    return CmdExecResult::error;
                }
                fmt::println("{}", gate_info.value());
            } else {
                auto gate_info = device_mgr.get()->gate_info_string(Device::QubitPair{ids[0], ids[1]});
                if (!gate_info.has_value()) {
                    switch (gate_info.error()) {
                        case TwoQubitGateInfoAccessError::invalid_first_qubit_id:
                            spdlog::error("Qubit {} does not exist", ids[0]);
                            return CmdExecResult::error;
                        case TwoQubitGateInfoAccessError::invalid_second_qubit_id:
                            spdlog::error("Qubit {} does not exist", ids[1]);
                            return CmdExecResult::error;
                        case TwoQubitGateInfoAccessError::invalid_qubit_pair:
                            fmt::println("({}, {}) is not adjacent", ids[0], ids[1]);
                            auto const path = get_shortest_path(apsp, ids[0], ids[1]);
                            if (path.has_value()) {
                                fmt::println("Shortest path: [{}]", fmt::join(path.value(), ", "));
                            } else {
                                spdlog::error("No path found between {} and {}", ids[0], ids[1]);
                            }
                            return CmdExecResult::done;
                    }
                }
                fmt::println("{}", gate_info.value());
            }
            return CmdExecResult::done;
        }};
}

dvlab::Command device_read_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {"read",
            [](ArgumentParser& parser) {
                parser.description("read a device topology");

                parser.add_argument<std::string>("filepath")
                    .help("the filepath to device file");

                parser.add_argument<bool>("-r", "--replace")
                    .action(store_true)
                    .help("if specified, replace the current device; otherwise store to a new one");
            },
            [&device_mgr](ArgumentParser const& parser) {
                auto filepath = parser.get<std::string>("filepath");
                auto replace  = parser.get<bool>("--replace");

                auto device = read_qsyn_device_file(filepath);

                if (!device.has_value()) {
                    spdlog::error("the format in \"{}\" has something wrong!!", filepath);
                    return CmdExecResult::error;
                }

                if (device_mgr.empty() || !replace) {
                    device_mgr.add(device_mgr.get_next_id(), std::make_unique<qsyn::device::Device>(std::move(device.value())));
                } else {
                    device_mgr.set(std::make_unique<qsyn::device::Device>(std::move(device.value())));
                }

                return CmdExecResult::done;
            }};
}

dvlab::Command device_bonsai_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {
        "bonsai",
        [](ArgumentParser& parser) {
            parser.description("build and print a Bonsai ternary tree for the current device");
            parser.add_argument<size_t>("-n", "--n-qubits")
                .default_value(std::numeric_limits<size_t>::max())
                .help("number of qubits in the Bonsai tree. If not specified, all qubits in the device will be used.");
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
        [&device_mgr](ArgumentParser const& parser) {
            if (device_mgr.empty()) {
                spdlog::error("No device loaded. Read or fetch a device first.");
                return CmdExecResult::error;
            }
            auto const n_qubits    = parser.get<size_t>("--n-qubits");
            auto const cost_fn_str = parser.get<std::string>("--cost-fn");
            auto cost_fn           = default_floyd_warshall_cost;
            if (dvlab::str::is_prefix_of(dvlab::str::tolower_string(cost_fn_str), "log_success_rate")) {
                cost_fn = log_success_rate_floyd_warshall_cost;
            }
            bool exhaustive = parser.get<bool>("--exhaustive");
            auto const apsp = floyd_warshall(*device_mgr.get(), cost_fn);
            auto tree       = [&]() {
                if (parser.parsed("--root-qubit-id")) {
                    auto const root_qubit_id = parser.get<size_t>("--root-qubit-id");
                    return qsyn::hamiltonian::build_bonsai_ternary_tree(
                        root_qubit_id, *device_mgr.get(), apsp, n_qubits);
                } else if (exhaustive) {
                    return qsyn::hamiltonian::build_bonsai_ternary_tree_exhaustive(
                        *device_mgr.get(), apsp, n_qubits);
                } else {
                    return qsyn::hamiltonian::build_bonsai_ternary_tree(
                        *device_mgr.get(), apsp, n_qubits);
                }
            }();
            if (!tree.has_value()) {
                switch (tree.error()) {
                    case qsyn::hamiltonian::BonsaiFailReason::not_enough_qubits:
                        spdlog::error("Failed to build Bonsai ternary tree: not enough qubits in the device");
                        spdlog::error("(Potentially disconnected device?)");
                        return CmdExecResult::error;
                    case qsyn::hamiltonian::BonsaiFailReason::invalid_root_qubit:
                        spdlog::error("Failed to build Bonsai ternary tree: invalid root qubit");
                        return CmdExecResult::error;
                }
                return CmdExecResult::error;
            }
            fmt::println("{}", qsyn::hamiltonian::to_string(tree.value()));
            return CmdExecResult::done;
        }};
}

dvlab::Command device_fetch_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {
        "fetch", [](ArgumentParser& parser) {
            parser.description(
                "fetch and create a device. Currently only supports "
                "IBM backends. This command tries to retrieve the device "
                "by connecting to the IBM Quantum Platform. An "
                "`IBMQ_API_KEY` needs to be set in the `.env` file. If the "
                "backend is not found, this command will try to retrieve "
                "cached attributes from "
                "`~/.config/qsyn/cached_backend_attrs/`, saved by previous "
                "calls to this command. If that fails, the command will try "
                "to use a fake backend. If that still fails, the command "
                "gives up and returns an error.");

            parser.add_argument<std::string>("backend")
                .help(
                    "the name of the IBM backend to fetch. This function "
                    "tries to normalize the backend to the format expected by "
                    "the IBM Quantum Platform/Qiskit. For example, `fez` will "
                    "be normalized to `ibm_fez` or `fake_fez`.");

            parser.add_argument<bool>("-f", "--fake")
                .action(store_true)
                .help(
                    "Only use fake backend. Note that fake backends will "
                    "not be cached.");

            parser.add_argument<bool>("-c", "--cached")
                .action(store_true)
                .help(
                    "Use cached backend if available; only fetch if "
                    "a cached backend is not available. This flag is "
                    "ignored if `--fake` is specified."); },
        [&device_mgr](ArgumentParser const& parser) {
            (void)device_mgr;  // reserved for device loading when hooked up
            auto const backend_name = parser.get<std::string>("backend");
            auto const fake         = parser.get<bool>("--fake");
            auto const cached       = parser.get<bool>("--cached");

            auto const home_dir = dvlab::utils::get_home_directory();
            if (!home_dir) {
                spdlog::error("Cannot find home directory");
                return CmdExecResult::error;
            }
            // set the cached directory. using make_optional to avoid copying
            auto const cached_dir = std::make_optional((
                std::filesystem::path(home_dir.value()) /
                ".config/qsyn/cached_backend_attrs/"));

            auto const result = dvlab::utils::with_spinner(
                [&]() {
                    return fetch_ibmq_device_attrs_with_fallback(
                        backend_name, fake, cached, cached_dir);
                },
                "Fetching IBM backend attributes...");

            if (!result) {
                spdlog::error("Failed to fetch IBM backend attributes for {}", backend_name);

                if (fake) {
                    fmt::println("Please provide a valid fake backend name (e.g., fake_manila, fake_oslo)");
                } else {
                    dvlab::utils::uv_run_script("scripts/get_ibm_backend_attrs.py", {backend_name, "--print-available-backends"});
                    fmt::println("Alternatively, fetch fake backends (e.g., fake_manila, fake_oslo) by specifying the -f/--fake flag.");
                }
                fmt::println("Available fake backends can be found at:");
                fmt::println("https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider");
                return CmdExecResult::error;
            }

            if (result->source == IBMQDeviceJsonsSource::cached && !cached) {
                spdlog::warn(
                    "Failed to fetch real backend attributes for {}. "
                    "Using cached backend attributes instead...",
                    backend_name);
            }

            if (result->source == IBMQDeviceJsonsSource::fake && !fake) {
                spdlog::warn(
                    "Failed to fetch real backend attributes for {}. "
                    "Using fake backend attributes instead...",
                    backend_name);
            }

            auto device = read_ibmq_device(result.value());
            if (!device.has_value()) {
                spdlog::error("Failed to parse IBM backend attributes for {}", backend_name);
                return CmdExecResult::error;
            }
            device_mgr.add(device_mgr.get_next_id(), std::make_unique<IBMQDevice>(std::move(device.value())));

            return CmdExecResult::done;
        }};
}

dvlab::Command device_write_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {
        "write",
        [](ArgumentParser& parser) {
            parser.description("Write the focused device to disk");

            parser.add_argument<std::string>("output")
                .help(
                    "Output path: a .json file (any name), or a directory "
                    "(writes <device_name>.json there).");

            parser.add_argument<bool>("--ibmq")
                .action(store_true)
                .help("Write IBM calibration bundle for Qiskit/Aer (single JSON file)");
        },
        [&device_mgr](ArgumentParser const& parser) {
            if (device_mgr.empty()) {
                spdlog::error("No device loaded");
                return CmdExecResult::error;
            }

            if (!parser.get<bool>("--ibmq")) {
                spdlog::error("Only `--ibmq` export is supported currently");
                return CmdExecResult::error;
            }

            auto const* ibmq = dynamic_cast<IBMQDevice const*>(device_mgr.get());
            if (ibmq == nullptr || !ibmq->jsons.has_value()) {
                spdlog::error("Focused device has no IBM JSON (use `device fetch` first)");
                return CmdExecResult::error;
            }

            auto const output_arg = parser.get<std::string>("output");
            if (output_arg.empty()) {
                spdlog::error("Output path is empty");
                return CmdExecResult::error;
            }

            std::filesystem::path output_path = output_arg;
            if (!output_arg.empty() && output_arg.back() == '/') {
                output_path = output_path / fmt::format("{}.json", ibmq->get_name());
            } else if (std::filesystem::is_directory(output_path)) {
                output_path = output_path / fmt::format("{}.json", ibmq->get_name());
            } else if (output_path.extension().empty()) {
                output_path.replace_extension(".json");
            }

            if (!write_ibmq_calibration(*ibmq, output_path)) {
                return CmdExecResult::error;
            }
            return CmdExecResult::done;
        }};
}

dvlab::Command device_cmd(qsyn::device::DeviceMgr& device_mgr) {
    auto cmd = dvlab::utils::mgr_root_cmd(device_mgr);
    cmd.add_subcommand("device-cmd-group", dvlab::utils::mgr_list_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", device_print_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", device_bonsai_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", dvlab::utils::mgr_checkout_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", device_read_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", device_fetch_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", device_write_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", dvlab::utils::mgr_delete_cmd(device_mgr));
    return cmd;
}

bool add_device_cmds(dvlab::CommandLineInterface& cli, qsyn::device::DeviceMgr& device_mgr) {
    if (!cli.add_command(device_cmd(device_mgr))) {
        spdlog::critical("Registering \"device\" commands fails... exiting");
        return false;
    }
    return true;
}

}  // namespace qsyn::device
