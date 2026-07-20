/****************************************************************************
  PackageName  [ device ]
  Synopsis     [ Slice IBM backend JSON to a subdevice calibration export ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2026 PARAG@N Lab, CS, Northwestern U, IL, USA ]
****************************************************************************/

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cctype>
#include <fstream>
#include <tl/enumerate.hpp>
#include <unordered_map>
#include <unordered_set>

#include "device/ibmq_devices.hpp"

namespace qsyn::device {

namespace {

nlohmann::json remap_qubit_list(
    nlohmann::json const& qubits,
    std::unordered_map<QubitIdType, QubitIdType> const& old_to_new) {
    nlohmann::json out = nlohmann::json::array();
    for (auto const& q : qubits) {
        out.push_back(old_to_new.at(q.get<QubitIdType>()));
    }
    return out;
}

bool qubits_in_active_set(
    nlohmann::json const& qubits,
    std::unordered_set<QubitIdType> const& active) {
    if (!qubits.is_array() || qubits.empty()) {
        return false;
    }
    for (auto const& q : qubits) {
        if (!active.contains(q.get<QubitIdType>())) {
            return false;
        }
    }
    return true;
}

}  // namespace

auto slice_ibmq_device_jsons(
    IBMQDeviceJsons const& full,
    std::span<QubitIdType const> physical_qubits)
    -> tl::expected<SlicedIBMQDeviceExport, SliceIBMQDeviceError> {
    std::vector<QubitIdType> ordered_physical;
    ordered_physical.reserve(physical_qubits.size());
    std::unordered_set<QubitIdType> active;
    std::unordered_map<QubitIdType, QubitIdType> old_to_new;

    auto const parent_n_qubits = full.properties_json.contains("qubits")
                                     ? full.properties_json["qubits"].size()
                                     : full.device_json.value("n_qubits", 0);

    for (auto const [logical, physical] : tl::views::enumerate(physical_qubits)) {
        if (!active.insert(physical).second) {
            return tl::unexpected(SliceIBMQDeviceError::duplicate_qubit_id);
        }
        if (physical >= parent_n_qubits) {
            return tl::unexpected(SliceIBMQDeviceError::unknown_qubit_id);
        }
        ordered_physical.push_back(physical);
        old_to_new.emplace(physical, static_cast<QubitIdType>(logical));
    }

    SlicedIBMQDeviceExport result{
        .jsons           = full,
        .physical_qubits = std::move(ordered_physical),
    };

    auto& device_json     = result.jsons.device_json;
    auto& properties_json = result.jsons.properties_json;

    if (properties_json.contains("qubits")) {
        nlohmann::json sliced_qubits = nlohmann::json::array();
        for (auto const physical : result.physical_qubits) {
            sliced_qubits.push_back(properties_json["qubits"].at(physical));
        }
        properties_json["qubits"] = std::move(sliced_qubits);
    }

    if (properties_json.contains("gates")) {
        nlohmann::json sliced_gates = nlohmann::json::array();
        for (auto const& gate : properties_json["gates"]) {
            if (!qubits_in_active_set(gate["qubits"], active)) {
                continue;
            }
            auto sliced_gate      = gate;
            sliced_gate["qubits"] = remap_qubit_list(gate["qubits"], old_to_new);
            sliced_gates.push_back(std::move(sliced_gate));
        }
        properties_json["gates"] = std::move(sliced_gates);
    }

    if (properties_json.contains("general")) {
        nlohmann::json sliced_general = nlohmann::json::array();
        for (auto const& item : properties_json["general"]) {
            if (!item.contains("name")) {
                continue;
            }
            auto const name = item["name"].get<std::string>();
            auto const pos  = name.rfind('_');
            if (pos == std::string::npos || pos + 3 > name.size()) {
                continue;
            }
            auto const suffix = name.substr(pos + 1);
            if (suffix.size() != 2 || !std::isdigit(suffix[0]) || !std::isdigit(suffix[1])) {
                continue;
            }
            auto const q0 = static_cast<QubitIdType>(suffix[0] - '0');
            auto const q1 = static_cast<QubitIdType>(suffix[1] - '0');
            if (!active.contains(q0) || !active.contains(q1)) {
                continue;
            }
            auto sliced_item    = item;
            sliced_item["name"] = fmt::format(
                "{}_{}{}",
                name.substr(0, pos),
                old_to_new.at(q0),
                old_to_new.at(q1));
            sliced_general.push_back(std::move(sliced_item));
        }
        properties_json["general"] = std::move(sliced_general);
    }

    device_json["n_qubits"] = result.physical_qubits.size();
    if (device_json.contains("backend_name")) {
        device_json["backend_name"] = fmt::format(
            "{}_sub_{}",
            device_json["backend_name"].get<std::string>(),
            fmt::join(result.physical_qubits, "_"));
    }

    if (device_json.contains("coupling_map")) {
        nlohmann::json sliced_coupling = nlohmann::json::array();
        for (auto const& pair : device_json["coupling_map"]) {
            if (!pair.is_array() || pair.size() != 2) {
                continue;
            }
            auto const q0 = pair[0].get<QubitIdType>();
            auto const q1 = pair[1].get<QubitIdType>();
            if (!active.contains(q0) || !active.contains(q1)) {
                continue;
            }
            sliced_coupling.push_back(nlohmann::json::array({old_to_new.at(q0), old_to_new.at(q1)}));
        }
        device_json["coupling_map"] = std::move(sliced_coupling);
    }

    if (device_json.contains("gates")) {
        for (auto& gate : device_json["gates"]) {
            if (!gate.contains("coupling_map")) {
                continue;
            }
            nlohmann::json sliced_coupling = nlohmann::json::array();
            for (auto const& pair : gate["coupling_map"]) {
                if (!pair.is_array() || pair.empty()) {
                    continue;
                }
                if (pair.size() == 1) {
                    auto const q = pair[0].get<QubitIdType>();
                    if (!active.contains(q)) {
                        continue;
                    }
                    sliced_coupling.push_back(nlohmann::json::array({old_to_new.at(q)}));
                } else if (pair.size() == 2) {
                    auto const q0 = pair[0].get<QubitIdType>();
                    auto const q1 = pair[1].get<QubitIdType>();
                    if (!active.contains(q0) || !active.contains(q1)) {
                        continue;
                    }
                    sliced_coupling.push_back(nlohmann::json::array({old_to_new.at(q0), old_to_new.at(q1)}));
                }
            }
            gate["coupling_map"] = std::move(sliced_coupling);
        }
    }

    return result;
}

IBMQDevice make_ibmq_subdevice(
    IBMQDevice const& parent,
    Device induced_device,
    SlicedIBMQDeviceExport sliced) {
    IBMQDevice out;
    static_cast<Device&>(out)  = std::move(induced_device);
    out.backend_version        = parent.backend_version;
    out.last_update_time       = parent.last_update_time;
    out.json_source            = parent.json_source;
    out.jsons                  = std::move(sliced.jsons);
    out.parent_physical_qubits = std::move(sliced.physical_qubits);
    return out;
}

bool write_ibmq_calibration(
    IBMQDevice const& device,
    std::filesystem::path const& output_path) {
    if (!device.jsons.has_value()) {
        spdlog::error("Device has no IBM JSON to write (use `device fetch` first)");
        return false;
    }

    auto const& jsons = *device.jsons;
    std::vector<QubitIdType> physical_qubits;
    if (device.parent_physical_qubits.has_value()) {
        physical_qubits = *device.parent_physical_qubits;
    } else {
        auto const n = jsons.properties_json.contains("qubits")
                           ? jsons.properties_json["qubits"].size()
                           : jsons.device_json.value("n_qubits", 0);
        physical_qubits.resize(n);
        for (size_t i = 0; i < n; ++i) {
            physical_qubits[i] = i;
        }
    }

    nlohmann::json bundle = {
        {"qsyn_ibmq_calibration", 1},
        {"physical_qubits", physical_qubits},
        {"configuration", jsons.device_json},
        {"properties", jsons.properties_json},
    };

    std::error_code ec;
    std::filesystem::create_directories(output_path.parent_path(), ec);
    if (ec) {
        spdlog::error(
            "Failed to create output directory {}: {}",
            output_path.parent_path().string(),
            ec.message());
        return false;
    }

    std::ofstream out(output_path);
    if (!out) {
        spdlog::error("Failed to open {} for writing", output_path.string());
        return false;
    }
    out << bundle.dump(2);

    fmt::println("Wrote IBM calibration to {}", output_path.string());
    fmt::println("  physical qubits: [{}]", fmt::join(physical_qubits, ", "));
    return true;
}

auto read_ibmq_calibration_bundle(std::filesystem::path const& path)
    -> std::optional<IBMQDevice> {
    std::ifstream in(path);
    if (!in) {
        return std::nullopt;
    }

    nlohmann::json bundle;
    try {
        in >> bundle;
    } catch (nlohmann::json::exception const&) {
        return std::nullopt;
    }

    if (!bundle.contains("qsyn_ibmq_calibration")) {
        return std::nullopt;
    }
    if (!bundle.contains("configuration") || !bundle.contains("properties")) {
        spdlog::error(
            "IBM calibration bundle {} is missing configuration or properties",
            path.string());
        return std::nullopt;
    }

    IBMQDeviceJsons jsons{
        .source          = IBMQDeviceJsonsSource::unknown,
        .device_json     = bundle["configuration"],
        .properties_json = bundle["properties"],
    };

    auto device = read_ibmq_device(jsons);
    if (!device.has_value()) {
        spdlog::error("Failed to parse IBM calibration bundle {}", path.string());
        return std::nullopt;
    }

    if (bundle.contains("physical_qubits")) {
        std::vector<QubitIdType> physical_qubits;
        physical_qubits.reserve(bundle["physical_qubits"].size());
        for (auto const& q : bundle["physical_qubits"]) {
            physical_qubits.push_back(q.get<QubitIdType>());
        }
        device->parent_physical_qubits = std::move(physical_qubits);
    }

    return device;
}

}  // namespace qsyn::device
