/****************************************************************************
  PackageName  [ qcir ]
  Synopsis     [ Estimated success probability (ESP) for a circuit on a device ]
  Author       [ Mu-Te Lau ]
****************************************************************************/

#include "qcir/qcir_esp.hpp"

#include <fmt/core.h>

#include <cassert>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "device/device.hpp"
#include "qcir/operation.hpp"
#include "qcir/qcir_device_calibration.hpp"
#include "qcir/qcir.hpp"

namespace qsyn::qcir {

tl::expected<EspResult, EspError> calculate_esp(
    QCir const& qcir,
    device::Device const& device,
    std::vector<std::string>& unsupported_gate_reprs,
    std::string& missing_calibration_detail,
    bool exclude_idle_qubits,
    bool gates_only) {
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
        if (!detail::device_native_gate_name(base_name, device).has_value()) {
            unsupported_set.insert(full_repr);
        }
    }
    if (!unsupported_set.empty()) {
        unsupported_gate_reprs.assign(unsupported_set.begin(), unsupported_set.end());
        return tl::unexpected(EspError::unsupported_gates);
    }

    std::vector<bool> qubit_has_gates(num_qubits, false);
    double esp       = 1.0;
    size_t num_gates = 0;

    for (auto const* gate : qcir.get_gates()) {
        auto const& op         = gate->get_operation();
        auto const base_name   = op.get_type();
        auto const full_repr   = op.get_repr();
        auto const device_gate = detail::device_native_gate_name(base_name, device);
        assert(device_gate.has_value());
        auto const gate_idx = detail::gate_type_index(device, *device_gate);
        assert(gate_idx.has_value());

        auto const& qubits = gate->get_qubits();
        std::optional<float> gate_error;
        if (qubits.size() == 1) {
            if (qubits[0] >= device.get_num_qubits()) {
                return tl::unexpected(EspError::circuit_qubit_out_of_range);
            }
            gate_error = detail::find_1q_gate_error(device, qubits[0], *gate_idx);
            if (!gate_error.has_value()) {
                missing_calibration_detail = fmt::format(
                    "1-qubit gate '{}' on qubit {}",
                    full_repr,
                    qubits[0]);
                return tl::unexpected(EspError::missing_gate_calibration);
            }
        } else if (qubits.size() == 2) {
            if (qubits[0] >= device.get_num_qubits() || qubits[1] >= device.get_num_qubits()) {
                return tl::unexpected(EspError::circuit_qubit_out_of_range);
            }
            gate_error = detail::find_2q_gate_error(device, qubits[0], qubits[1], *gate_idx);
            if (!gate_error.has_value()) {
                missing_calibration_detail = fmt::format(
                    "2-qubit gate '{}' on qubits ({}, {})",
                    full_repr,
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

        esp *= (1.0 - static_cast<double>(*gate_error));
        for (auto const qubit : qubits) {
            qubit_has_gates[qubit] = true;
        }
        ++num_gates;
    }

    size_t num_measurements = 0;

    if (!gates_only) {
        for (QubitIdType q = 0; q < num_qubits; ++q) {
            if (exclude_idle_qubits && !qubit_has_gates[q]) {
                continue;
            }
            auto const& props = device.get_qubit_properties(q);
            if (!props.readout_error.has_value()) {
                missing_calibration_detail = fmt::format("readout error on qubit {}", q);
                return tl::unexpected(EspError::missing_readout_calibration);
            }
            esp *= (1.0 - static_cast<double>(*props.readout_error));
            ++num_measurements;
        }
    }

    return EspResult{
        .esp              = esp,
        .num_gates        = num_gates,
        .num_measurements = num_measurements,
    };
}

}  // namespace qsyn::qcir
