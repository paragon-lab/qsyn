/****************************************************************************
  PackageName  [ device ]
  Synopsis     [ Floyd-Warshall and device analysis utilities ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#pragma once

#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include "device/device.hpp"
#include "qsyn/qsyn_type.hpp"
#include "util/graph/floyd_warshall.hpp"

namespace qsyn::device {

using APSPResult = dvlab::APSPResult<QubitIdType>;

using APSPCostFnType = std::function<float(Device::QubitPair const&, Device const&)>;

float default_floyd_warshall_cost(Device::QubitPair const& /*adj*/, Device const& /*device*/);
float log_success_rate_floyd_warshall_cost(Device::QubitPair const& adj, Device const& device);
/// Cost ``-log2(PF)`` with
/// ``PF = (1 - ε) * sqrt(∏_q (2/3 e^{-t/T2} + 1/3 e^{-t/T1}))``.
float log_proxy_fidelity_floyd_warshall_cost(Device::QubitPair const& adj, Device const& device);

APSPResult floyd_warshall(
    Device const& device,
    APSPCostFnType const& cost_fn = default_floyd_warshall_cost);

std::vector<float> get_eccentricities(APSPResult const& apsp, Device const& device);

using QubitFilterFn = std::function<bool(QubitIdType const&)>;

extern QubitFilterFn accept_all_qubit_ids;

std::vector<QubitIdType>
get_centers(
    std::vector<float> const& eccentricities,
    QubitFilterFn const& filter_fn = accept_all_qubit_ids);

std::vector<QubitIdType>
get_centers(
    APSPResult const& apsp,
    Device const& device,
    QubitFilterFn const& filter_fn = accept_all_qubit_ids);

std::optional<std::vector<QubitIdType>> get_shortest_path(APSPResult const& apsp, QubitIdType src, QubitIdType dest);

std::vector<std::vector<QubitIdType>>
get_connected_components(
    APSPResult const& apsp,
    Device const& device,
    QubitFilterFn const& filter_fn = accept_all_qubit_ids);

}  // namespace qsyn::device
