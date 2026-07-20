/****************************************************************************
  PackageName  [ qcir ]
  Synopsis     [ Proxy fidelity (Wu et al. 2026) for a circuit on a device ]
  Author       [ Mu-Te Lau ]
****************************************************************************/

#pragma once

#include <cstdint>
#include <string>
#include <tl/expected.hpp>
#include <vector>

#include "device/device.hpp"
#include "qcir/qcir.hpp"

namespace qsyn::qcir {

enum class ProxyFidelityError : std::uint8_t {
    circuit_qubit_out_of_range,
    unsupported_gates,
    missing_gate_calibration,
    missing_relaxation_calibration,
    missing_readout_calibration,
};

struct ProxyFidelityResult {
    double proxy_fidelity;
    std::vector<double> per_qubit_fidelity;
    size_t num_gates;
    size_t num_measurements;
};

/**
 * @brief Compute circuit proxy fidelity from Wu et al. (2026), Eq. (7) per qubit
 *        and Eq. (6) product aggregation.
 *
 * For each gate on a qubit, applies a depolarizing update then a thermal-relaxation
 * update. Optionally multiplies by (1 - readout error) as the final SPAM channel.
 * Requires device-native gates only.
 *
 * @param exclude_idle_qubits If true, omit qubits that experience no gates from the product.
 * @param gates_only          If true, skip the final SPAM/readout factor.
 */
[[nodiscard]] tl::expected<ProxyFidelityResult, ProxyFidelityError> calculate_proxy_fidelity(
    QCir const& qcir,
    device::Device const& device,
    std::vector<std::string>& unsupported_gate_reprs,
    std::string& missing_calibration_detail,
    bool exclude_idle_qubits = false,
    bool gates_only          = false);

}  // namespace qsyn::qcir
