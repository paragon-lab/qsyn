/****************************************************************************
  PackageName  [ qcir ]
  Synopsis     [ Estimated success probability (ESP) for a circuit on a device ]
  Author       [ Design Verification Lab ]
****************************************************************************/

#include "qcir/qcir_esp.hpp"

#include <fmt/core.h>

#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "device/device.hpp"
#include "qcir/gate_name.hpp"
#include "qcir/operation.hpp"
#include "qcir/qcir.hpp"

namespace qsyn::qcir {

namespace {

bool device_has_gate_type(device::Device const& device, std::string_view gate_type) {
    for (auto const& name : device.get_gate_set()) {
        if (name == gate_type) {
            return true;
        }
    }
    return false;
}

/**
 * Map a circuit gate's base name to a name in the device gate set.
 *
 * OpenQASM ``p`` is calibrated as ``rz`` on IBM backends; ``cx`` often appears as ``cz``
 * in properties JSON.
 */
std::optional<std::string> resolve_device_gate_name(
    std::string_view base_name,
    device::Device const& device) {
    if (device_has_gate_type(device, base_name)) {
        return std::string(base_name);
    }

    static std::pair<std::string_view, std::string_view> const aliases[] = {
        {"p", "rz"},
        {"px", "rx"},
        {"py", "ry"},
        {"cp", "cz"},
        {"cx", "cz"},
        {"cy", "cz"},
    };
    for (auto const& [from, to] : aliases) {
        if (base_name == from && device_has_gate_type(device, to)) {
            return std::string(to);
        }
    }
    return std::nullopt;
}

std::optional<size_t> gate_type_index(device::Device const& device, std::string_view gate_type) {
    auto const& gate_set = device.get_gate_set();
    for (size_t i = 0; i < gate_set.size(); ++i) {
        if (gate_set[i] == gate_type) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<float> find_1q_gate_error(
    device::Device const& device,
    QubitIdType qubit,
    size_t gate_idx) {
    for (auto const& info : device.get_gate_info(qubit)) {
        if (info.gate_idx == gate_idx) {
            return info.error;
        }
    }
    return std::nullopt;
}

std::optional<float> find_2q_gate_error(
    device::Device const& device,
    QubitIdType q0,
    QubitIdType q1,
    size_t gate_idx) {
    auto const try_edge = [&](device::Device::QubitPair pair) -> std::optional<float> {
        if (!device.is_adjacent(pair)) {
            return std::nullopt;
        }
        for (auto const& info : device.get_gate_info(pair)) {
            if (info.gate_idx == gate_idx) {
                return info.error;
            }
        }
        return std::nullopt;
    };

    if (auto err = try_edge({q0, q1})) {
        return err;
    }
    return try_edge({q1, q0});
}

}  // namespace

tl::expected<EspResult, EspError> calculate_esp(
    QCir const& qcir,
    device::Device const& device,
    std::vector<std::string>& unsupported_gate_reprs,
    std::string& missing_calibration_detail) {
    unsupported_gate_reprs.clear();
    missing_calibration_detail.clear();

    auto const num_qubits = qcir.get_num_qubits();
    if (num_qubits > device.get_num_qubits()) {
        return tl::unexpected(EspError::circuit_qubit_out_of_range);
    }

    std::unordered_set<std::string> unsupported_set;
    for (auto const* gate : qcir.get_gates()) {
        auto const& op       = gate->get_operation();
        auto const base_name = op.get_type();
        auto const full_repr = op.get_repr();
        if (!resolve_device_gate_name(base_name, device).has_value()) {
            unsupported_set.insert(full_repr);
        }
    }
    if (!unsupported_set.empty()) {
        unsupported_gate_reprs.assign(unsupported_set.begin(), unsupported_set.end());
        return tl::unexpected(EspError::unsupported_gates);
    }

    double gate_success_product = 1.0;
    size_t num_gates            = 0;

    for (auto const* gate : qcir.get_gates()) {
        auto const& op         = gate->get_operation();
        auto const base_name   = op.get_type();
        auto const full_repr   = op.get_repr();
        auto const device_gate = resolve_device_gate_name(base_name, device);
        assert(device_gate.has_value());
        auto const gate_idx = gate_type_index(device, *device_gate);
        assert(gate_idx.has_value());

        auto const& qubits = gate->get_qubits();
        std::optional<float> gate_error;
        if (qubits.size() == 1) {
            if (qubits[0] >= device.get_num_qubits()) {
                return tl::unexpected(EspError::circuit_qubit_out_of_range);
            }
            gate_error = find_1q_gate_error(device, qubits[0], *gate_idx);
            if (!gate_error.has_value()) {
                missing_calibration_detail = fmt::format(
                    "1-qubit gate '{}' (device gate '{}') on qubit {}",
                    full_repr,
                    *device_gate,
                    qubits[0]);
                return tl::unexpected(EspError::missing_gate_calibration);
            }
        } else if (qubits.size() == 2) {
            if (qubits[0] >= device.get_num_qubits() || qubits[1] >= device.get_num_qubits()) {
                return tl::unexpected(EspError::circuit_qubit_out_of_range);
            }
            gate_error = find_2q_gate_error(device, qubits[0], qubits[1], *gate_idx);
            if (!gate_error.has_value()) {
                missing_calibration_detail = fmt::format(
                    "2-qubit gate '{}' (device gate '{}') on qubits ({}, {})",
                    full_repr,
                    *device_gate,
                    qubits[0],
                    qubits[1]);
                return tl::unexpected(EspError::missing_gate_calibration);
            }
        } else {
            missing_calibration_detail = fmt::format(
                "gate '{}' with {} qubits (only 1- and 2-qubit gates supported)",
                full_repr,
                qubits.size());
            return tl::unexpected(EspError::missing_gate_calibration);
        }

        gate_success_product *= (1.0 - static_cast<double>(*gate_error));
        ++num_gates;
    }

    double measurement_success_product = 1.0;
    size_t num_measurements            = 0;

    for (QubitIdType q = 0; q < num_qubits; ++q) {
        auto const& props = device.get_qubit_properties(q);
        if (!props.readout_error.has_value()) {
            missing_calibration_detail = fmt::format("readout error on qubit {}", q);
            return tl::unexpected(EspError::missing_readout_calibration);
        }
        measurement_success_product *= (1.0 - static_cast<double>(*props.readout_error));
        ++num_measurements;
    }

    return EspResult{
        .esp              = gate_success_product * measurement_success_product,
        .num_gates        = num_gates,
        .num_measurements = num_measurements,
    };
}

}  // namespace qsyn::qcir
