/****************************************************************************
  PackageName  [ device ]
  Synopsis     [ Define class IBMQDevices and fetching functions ]
  Author       [ Mu-Te (Joshua) Lau]
  Copyright    [ Copyright(c) 2026 PARAG@N Lab, CS, Northwestern U, IL, USA ]
****************************************************************************/

#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string_view>
#include <tl/expected.hpp>
#include <vector>

#include "device/device.hpp"

namespace qsyn::device {

enum struct IBMQDeviceJsonsSource : std::uint8_t {
    real,
    fake,
    cached,
    unknown,
};

struct IBMQDeviceJsons {
    IBMQDeviceJsonsSource source;
    nlohmann::json device_json;
    nlohmann::json properties_json;
};

struct IBMQDevice : public Device {
    std::string backend_version;
    std::chrono::sys_seconds last_update_time;
    IBMQDeviceJsonsSource json_source;
    /// IBM backend JSON (full or already sliced/remapped to logical 0..n-1).
    std::optional<IBMQDeviceJsons> jsons;
    /// When set, ``parent_physical_qubits[logical]`` on the parent backend (for remap export).
    std::optional<std::vector<QubitIdType>> parent_physical_qubits;

    std::string info_string() const override;
};

enum class SliceIBMQDeviceError : uint8_t {
    duplicate_qubit_id,
    unknown_qubit_id,
};

struct SlicedIBMQDeviceExport {
    IBMQDeviceJsons jsons;
    /// ``physical_qubits[logical]`` on the parent backend.
    std::vector<QubitIdType> physical_qubits;
};

[[nodiscard]] tl::expected<SlicedIBMQDeviceExport, SliceIBMQDeviceError> slice_ibmq_device_jsons(
    IBMQDeviceJsons const& full,
    std::span<QubitIdType const> physical_qubits);

IBMQDevice make_ibmq_subdevice(
    IBMQDevice const& parent,
    Device induced_device,
    SlicedIBMQDeviceExport sliced);

bool write_ibmq_calibration(
    IBMQDevice const& device,
    std::filesystem::path const& output_path);

/// Load a device from ``device write --ibmq`` JSON (``qsyn_ibmq_calibration`` bundle).
auto read_ibmq_calibration_bundle(std::filesystem::path const& path)
    -> std::optional<IBMQDevice>;

auto read_ibmq_device(IBMQDeviceJsons const& jsons) -> std::optional<IBMQDevice>;

auto load_ibmq_devices_jsons(std::filesystem::path const& device_path,
                             std::filesystem::path const& properties_path,
                             IBMQDeviceJsonsSource source = IBMQDeviceJsonsSource::unknown)
    -> std::optional<IBMQDeviceJsons>;

auto fetch_ibmq_device_attrs(std::string_view backend_name, bool fake = false)
    -> std::optional<IBMQDeviceJsons>;

auto fetch_ibmq_device_attrs_with_fallback(
    std::string_view backend_name,
    bool fake                                             = false,
    bool cached                                           = false,
    std::optional<std::filesystem::path> const& cache_dir = std::nullopt)
    -> std::optional<IBMQDeviceJsons>;

}  // namespace qsyn::device
