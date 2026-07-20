/****************************************************************************
  PackageName  [ qcir ]
  Synopsis     [ Proxy fidelity (Wu et al. 2026) for a circuit on a device ]
  Author       [ Mu-Te Lau ]
****************************************************************************/

#include "qcir/qcir_proxy_fidelity.hpp"

#include <fmt/core.h>

#include <cassert>
#include <cmath>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "device/device.hpp"
#include "qcir/operation.hpp"
#include "qcir/qcir_device_calibration.hpp"
#include "qcir/qcir.hpp"

namespace qsyn::qcir {

namespace {

/// Convert average gate error rate ``r`` to depolarizing probability ``p``.
/// Uses ``p = d/(d-1) * r`` with Hilbert-space dimension ``d = 2^{n}`` (Magesan et al.).
double depolarizing_probability(double gate_error, size_t num_gate_qubits) {
    auto const dim = static_cast<double>(1ULL << num_gate_qubits);
    return gate_error * dim / (dim - 1.0);
}

/// Apply Eq. (8): ``f <- 1/2 + (f - 1/2)(1 - p)``.
void apply_depolarizing(double& fidelity, double p) {
    fidelity = 0.5 + (fidelity - 0.5) * (1.0 - p);
}

/// Apply Eq. (9): ``f <- 1/2 + (f - 1/2)(2/3 e^{-t/T2} + 1/3 e^{-t/T1})``.
/// ``t``, ``t1``, and ``t2`` must use the same time unit.
void apply_thermal_relaxation(double& fidelity, double t, double t1, double t2) {
    auto const factor = (2.0 / 3.0) * std::exp(-t / t2) + (1.0 / 3.0) * std::exp(-t / t1);
    fidelity          = 0.5 + (fidelity - 0.5) * factor;
}

/// Gate duration in microseconds (device T1/T2 are stored in µs; gate time in ns).
double gate_duration_us(device::GateInfo const& info) {
    return static_cast<double>(info.time.count()) * 1e-3;
}

}  // namespace

tl::expected<ProxyFidelityResult, ProxyFidelityError> calculate_proxy_fidelity(
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
        return tl::unexpected(ProxyFidelityError::circuit_qubit_out_of_range);
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
        return tl::unexpected(ProxyFidelityError::unsupported_gates);
    }

    std::vector<double> qubit_fidelity(num_qubits, 1.0);
    std::vector<bool> qubit_has_gates(num_qubits, false);
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
        std::optional<device::GateInfo> gate_info;
        if (qubits.size() == 1) {
            if (qubits[0] >= device.get_num_qubits()) {
                return tl::unexpected(ProxyFidelityError::circuit_qubit_out_of_range);
            }
            gate_info = detail::find_1q_gate_info(device, qubits[0], *gate_idx);
            if (!gate_info.has_value()) {
                missing_calibration_detail = fmt::format(
                    "1-qubit gate '{}' on qubit {}",
                    full_repr,
                    qubits[0]);
                return tl::unexpected(ProxyFidelityError::missing_gate_calibration);
            }
        } else if (qubits.size() == 2) {
            if (qubits[0] >= device.get_num_qubits() || qubits[1] >= device.get_num_qubits()) {
                return tl::unexpected(ProxyFidelityError::circuit_qubit_out_of_range);
            }
            gate_info = detail::find_2q_gate_info(device, qubits[0], qubits[1], *gate_idx);
            if (!gate_info.has_value()) {
                missing_calibration_detail = fmt::format(
                    "2-qubit gate '{}' on qubits ({}, {})",
                    full_repr,
                    qubits[0],
                    qubits[1]);
                return tl::unexpected(ProxyFidelityError::missing_gate_calibration);
            }
        } else {
            missing_calibration_detail = fmt::format(
                "gate '{}' with {} qubits (only 1- and 2-qubit gates supported)",
                full_repr,
                qubits.size());
            return tl::unexpected(ProxyFidelityError::missing_gate_calibration);
        }

        auto const p    = depolarizing_probability(static_cast<double>(gate_info->error), qubits.size());
        auto const t_us = gate_duration_us(*gate_info);

        for (auto const qubit : qubits) {
            auto const& props = device.get_qubit_properties(qubit);
            if (!props.t1.has_value() || !props.t2.has_value() ||
                *props.t1 <= 0.f || *props.t2 <= 0.f) {
                missing_calibration_detail = fmt::format(
                    "T1/T2 on qubit {} (needed for thermal relaxation)",
                    qubit);
                return tl::unexpected(ProxyFidelityError::missing_relaxation_calibration);
            }

            // Two-qubit depolarizing channel decomposes into parallel single-qubit
            // channels with the same p (Wu et al. Sec. 3.4.2).
            apply_depolarizing(qubit_fidelity[qubit], p);
            apply_thermal_relaxation(
                qubit_fidelity[qubit],
                t_us,
                static_cast<double>(*props.t1),
                static_cast<double>(*props.t2));
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
                return tl::unexpected(ProxyFidelityError::missing_readout_calibration);
            }
            qubit_fidelity[q] *= (1.0 - static_cast<double>(*props.readout_error));
            ++num_measurements;
        }
    }

    double proxy_fidelity = 1.0;
    for (QubitIdType q = 0; q < num_qubits; ++q) {
        if (exclude_idle_qubits && !qubit_has_gates[q]) {
            continue;
        }
        proxy_fidelity *= qubit_fidelity[q];
    }

    return ProxyFidelityResult{
        .proxy_fidelity     = proxy_fidelity,
        .per_qubit_fidelity = std::move(qubit_fidelity),
        .num_gates          = num_gates,
        .num_measurements   = num_measurements,
    };
}

}  // namespace qsyn::qcir
