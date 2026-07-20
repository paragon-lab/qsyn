/****************************************************************************
  PackageName  [ device ]
  Synopsis     [ Floyd-Warshall and device analysis implementations ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "device/device_analysis.hpp"

#include <algorithm>
#include <cmath>

#include "util/graph/floyd_warshall.hpp"

namespace qsyn::device {

float default_floyd_warshall_cost(Device::QubitPair const& /*adj*/, Device const& /*device*/) {
    return 1.f;
}

/**
 * @brief Additive cost function for Floyd-Warshall algorithm.
 *        Cost is -log2(1 - error) for the first gate info of the adjacency.
 *        Returns infinity when error >= 1 (broken/unusable coupling).
 * @param adj Adjacency pair
 * @param device Device
 * @return Cost
 */
float log_success_rate_floyd_warshall_cost(Device::QubitPair const& adj, Device const& device) {
    // assumes the first gate info is the only one for this adjacency
    auto const& gate_info = device.get_gate_info(adj)[0];

    if (1.f - gate_info.error <= 0.f) {
        return std::numeric_limits<float>::infinity();
    }

    return -std::log2(1.f - gate_info.error);
}

/**
 * @brief Additive cost ``-log2(PF)`` for Floyd-Warshall, where
 *        ``PF = (1 - ε) √(∏_{q∈{q1,q2}} (2/3 e^{-t/T2} + 1/3 e^{-t/T1}))``
 *        approximates the gate proxy fidelity (depolarizing + thermal).
 *        Gate time is converted from ns to µs to match device T1/T2 units.
 *        Returns infinity when the coupling is broken or T1/T2 is missing.
 */
float log_proxy_fidelity_floyd_warshall_cost(Device::QubitPair const& adj, Device const& device) {
    // assumes the first gate info is the only one for this adjacency
    auto const& gate_info = device.get_gate_info(adj)[0];
    auto const success    = 1.f - gate_info.error;
    if (success <= 0.f) {
        return std::numeric_limits<float>::infinity();
    }

    auto const& props_src = device.get_qubit_properties(adj.src);
    auto const& props_dst = device.get_qubit_properties(adj.dst);
    if (!props_src.t1.has_value() || !props_src.t2.has_value() ||
        !props_dst.t1.has_value() || !props_dst.t2.has_value() ||
        *props_src.t1 <= 0.f || *props_src.t2 <= 0.f ||
        *props_dst.t1 <= 0.f || *props_dst.t2 <= 0.f) {
        return std::numeric_limits<float>::infinity();
    }

    // GateDelayNanoSec::count() is nanoseconds; T1/T2 are microseconds.
    auto const t_us = gate_info.time.count() * 1e-3f;

    auto const thermal = [t_us](float t1, float t2) {
        return (2.f / 3.f) * std::exp(-t_us / t2) + (1.f / 3.f) * std::exp(-t_us / t1);
    };

    auto const pf =
        success *
        std::sqrt(
            thermal(*props_src.t1, *props_src.t2) *
            thermal(*props_dst.t1, *props_dst.t2));
    if (pf <= 0.f) {
        return std::numeric_limits<float>::infinity();
    }

    return -std::log2(pf);
}

/**
 * @brief Floyd-Warshall APSP for the device coupling graph.
 *        Wraps cost_fn so broken couplings (error == 1) always get cost inf, then delegates to generic APSP.
 */
APSPResult floyd_warshall(
    Device const& device,
    std::function<float(Device::QubitPair const&, Device const&)> const& cost_fn) {
    auto const edge_cost = [&](Device::QubitPair const& e) {
        return (device.get_gate_info(e)[0].error == 1.f)
                   ? std::numeric_limits<float>::infinity()
                   : cost_fn(e, device);
    };
    return dvlab::floyd_warshall(device.get_coupling_graph(), edge_cost);
}

std::vector<float> get_eccentricities(APSPResult const& apsp, Device const& device) {
    std::vector<float> eccentricities(device.get_num_qubits(), std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < device.get_num_qubits(); i++) {
        for (size_t j = 0; j < device.get_num_qubits(); j++) {
            if (std::isinf(apsp.distance[i][j])) continue;
            eccentricities[i] = std::max(eccentricities[i], apsp.distance[i][j]);
        }
    }
    return eccentricities;
}

/**
 * @brief Get the centers of the device, i.e., arg min(max_i d(i, j)) where j is all other qubits
 * @param eccentricities Eccentricities of the qubits
 * @param filter_fn Function to filter the qubits. A qubit is only considered if filter_fn(qubit_id) returns true.
 * @return Centers
 */
std::vector<QubitIdType> get_centers(
    std::vector<float> const& eccentricities,
    std::function<bool(QubitIdType const&)> const& filter_fn) {
    auto radius = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < eccentricities.size(); i++) {
        if (!filter_fn(i)) continue;
        radius = std::min(radius, eccentricities[i]);
    }
    std::vector<QubitIdType> centers;
    for (size_t i = 0; i < eccentricities.size(); i++) {
        if (!filter_fn(i)) continue;
        if (eccentricities[i] == radius) {
            centers.push_back(i);
        }
    }
    return centers;
}

/**
 * @brief Get the centers of the device, i.e., arg min(max_i d(i, j)) where j is all other qubits
 * @param apsp APSPResult
 * @param device Device
 * @param filter_fn Function to filter the qubits. A qubit is only considered if filter_fn(qubit_id) returns true.
 * @return Centers
 */
std::vector<QubitIdType>
get_centers(
    APSPResult const& apsp, Device const& device,
    std::function<bool(QubitIdType const&)> const& filter_fn) {
    auto const eccentricities = get_eccentricities(apsp, device);
    return get_centers(eccentricities, filter_fn);
}

std::optional<std::vector<QubitIdType>> get_shortest_path(APSPResult const& apsp, QubitIdType src, QubitIdType dest) {
    std::vector<QubitIdType> path;
    path.push_back(src);
    while (src != dest) {
        if (!apsp.predecessor[dest][src].has_value()) return std::nullopt;
        src = apsp.predecessor[dest][src].value();
        path.push_back(src);
    }
    return path;
}

QubitFilterFn accept_all_qubit_ids = [](QubitIdType const& /*qubit_id*/) { return true; };

/**
 * @brief Get the connected components of the device.
 * @param apsp APSPResult. This is used to check if two qubits are effectively
 *        connected. For example, if two qubits are connected but the APSP
 *        distance is infinite (e.g., due to broken couplings),
 *        this function will treat them as disconnected.
 * @param device Device
 * @return A vector of vector of qubit ID, where each vector contains qubits
           that can reach each other.
 */
std::vector<std::vector<QubitIdType>>
get_connected_components(
    APSPResult const& apsp,
    Device const& device,
    QubitFilterFn const& filter_fn) {
    std::vector<std::vector<QubitIdType>> connected_components;

    std::vector<bool> visited(device.get_num_qubits(), false);

    for (size_t i = 0; i < device.get_num_qubits(); i++) {
        if (!filter_fn(i) || visited[i]) continue;
        // do DFS to find all qubits that can reach i
        connected_components.push_back(std::vector<QubitIdType>());
        std::vector<QubitIdType> stack;
        stack.push_back(i);
        while (!stack.empty()) {
            auto const current = stack.back();
            stack.pop_back();
            if (!filter_fn(current) || visited[current]) continue;
            visited[current] = true;
            connected_components.back().push_back(current);
            for (auto const& neighbor : device.get_adjacencies(current)) {
                if (std::isinf(apsp.distance[current][neighbor])) continue;
                if (!visited[neighbor]) stack.push_back(neighbor);
            }
        }
        std::ranges::sort(connected_components.back());
    }
    return connected_components;
}

}  // namespace qsyn::device
