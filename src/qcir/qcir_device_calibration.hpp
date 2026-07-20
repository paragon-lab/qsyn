/****************************************************************************
  PackageName  [ qcir ]
  Synopsis     [ Shared helpers for looking up device gate calibrations ]
  Author       [ Mu-Te Lau ]
****************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "device/device.hpp"
#include "qsyn/qsyn_type.hpp"

namespace qsyn::qcir::detail {

inline bool device_has_gate_type(device::Device const& device, std::string_view gate_type) {
    for (auto const& name : device.get_gate_set()) {
        if (name == gate_type) {
            return true;
        }
    }
    return false;
}

inline std::optional<std::string> device_native_gate_name(
    std::string_view base_name,
    device::Device const& device) {
    if (device_has_gate_type(device, base_name)) {
        return std::string(base_name);
    }
    return std::nullopt;
}

inline std::optional<size_t> gate_type_index(device::Device const& device, std::string_view gate_type) {
    auto const& gate_set = device.get_gate_set();
    for (size_t i = 0; i < gate_set.size(); ++i) {
        if (gate_set[i] == gate_type) {
            return i;
        }
    }
    return std::nullopt;
}

inline std::optional<device::GateInfo> find_1q_gate_info(
    device::Device const& device,
    QubitIdType qubit,
    size_t gate_idx) {
    for (auto const& info : device.get_gate_info(qubit)) {
        if (info.gate_idx == gate_idx) {
            return info;
        }
    }
    return std::nullopt;
}

inline std::optional<device::GateInfo> find_2q_gate_info(
    device::Device const& device,
    QubitIdType q0,
    QubitIdType q1,
    size_t gate_idx) {
    auto const try_edge = [&](device::Device::QubitPair pair) -> std::optional<device::GateInfo> {
        if (!device.is_adjacent(pair)) {
            return std::nullopt;
        }
        for (auto const& info : device.get_gate_info(pair)) {
            if (info.gate_idx == gate_idx) {
                return info;
            }
        }
        return std::nullopt;
    };

    if (auto info = try_edge({q0, q1})) {
        return info;
    }
    return try_edge({q1, q0});
}

inline std::optional<float> find_1q_gate_error(
    device::Device const& device,
    QubitIdType qubit,
    size_t gate_idx) {
    if (auto info = find_1q_gate_info(device, qubit, gate_idx)) {
        return info->error;
    }
    return std::nullopt;
}

inline std::optional<float> find_2q_gate_error(
    device::Device const& device,
    QubitIdType q0,
    QubitIdType q1,
    size_t gate_idx) {
    if (auto info = find_2q_gate_info(device, q0, q1, gate_idx)) {
        return info->error;
    }
    return std::nullopt;
}

}  // namespace qsyn::qcir::detail
