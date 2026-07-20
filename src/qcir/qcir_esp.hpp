/****************************************************************************
  PackageName  [ qcir ]
  Synopsis     [ Estimated success probability (ESP) for a circuit on a device ]
  Author       [ Mu-Te Lau ]
****************************************************************************/

#pragma once

#include <string>
#include <vector>

#include "device/device.hpp"
#include "qcir/qcir.hpp"

namespace qsyn::qcir {

enum class EspError : std::uint8_t {
    circuit_qubit_out_of_range,
    unsupported_gates,
    missing_gate_calibration,
    missing_readout_calibration,
};

struct EspResult {
    double esp;
    size_t num_gates;
    size_t num_measurements;
};

[[nodiscard]] tl::expected<EspResult, EspError> calculate_esp(
    QCir const& qcir,
    device::Device const& device,
    std::vector<std::string>& unsupported_gate_reprs,
    std::string& missing_calibration_detail,
    bool exclude_idle_qubits = false,
    bool gates_only          = false);

}  // namespace qsyn::qcir
