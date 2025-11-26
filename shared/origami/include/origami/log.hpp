// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <type_traits>
#include <unordered_map>

namespace origami {

/**
 * @brief Logger for collecting and exporting analytical metrics in JSON format.
 *
 * Provides a single templated logging function that stores key-value pairs
 * and can export them as JSON. The caller is responsible for checking if
 * debug is enabled before calling log().
 */
class logger_t {
 public:
  /**
   * @brief Log a key-value pair.
   *
   * @tparam T Type of the value (must be convertible to JSON-compatible string)
   * @param key The metric key
   * @param value The metric value
   */
  template<typename T>
  void log(const std::string& key, const T& value) {
    metrics_[key] = to_json_string(value);
  }

  /**
   * @brief Clear all logged metrics.
   */
  void clear() { metrics_.clear(); }

  /**
   * @brief Print all metrics as JSON to stdout.
   */
  void print() const;

  /**
   * @brief Export metrics to a JSON file.
   *
   * @param filename Output filename
   */
  void export_json(const std::string& filename) const;

  /**
   * @brief Get all metrics as a map.
   *
   * @return Map of metric key-value pairs
   */
  std::unordered_map<std::string, std::string> get_metrics() const { return metrics_; }

  /**
   * @brief Check if logger has any metrics.
   *
   * @return true if metrics map is not empty
   */
  bool empty() const { return metrics_.empty(); }

 private:
  std::unordered_map<std::string, std::string> metrics_;

  // Convert value to JSON-compatible string
  template<typename T>
  std::string to_json_string(const T& value) {
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::string>) {
      return "\"" + value + "\"";
    } else if constexpr (std::is_convertible_v<T, const char*> || std::is_array_v<std::remove_reference_t<T>>) {
      return "\"" + std::string(value) + "\"";
    } else if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, bool>) {
      return value ? "true" : "false";
    } else if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>) {
      return std::to_string(value);
    } else {
      // For other types, try to_string (this may fail for some types)
      static_assert(std::is_arithmetic_v<T>, "Type must be convertible to string or arithmetic");
      return std::to_string(value);
    }
  }
};

}  // namespace origami

