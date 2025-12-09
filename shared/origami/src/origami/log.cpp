/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "origami/log.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace origami {

void logger_t::print() const {
  if (!metrics_ || metrics_->empty()) {
    std::cout << "{}\n";
    return;
  }
  std::cout << "{\n";
  bool first = true;
  for (const auto& [key, val] : *metrics_) {
    if (!first) std::cout << ",\n";
    std::cout << "  \"" << key << "\": " << val;
    first = false;
  }
  std::cout << "\n}\n";
}

void logger_t::export_json(const std::string& filename) const {
  if (!metrics_ || metrics_->empty()) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << " for writing\n";
      return;
    }
    file << "{}\n";
    file.close();
    std::cout << "Analytical metrics exported to JSON: " << filename << "\n";
    return;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing\n";
    return;
  }

  file << std::fixed << std::setprecision(6);
  file << "{\n";
  bool first = true;
  for (const auto& [key, val] : *metrics_) {
    if (!first) file << ",\n";
    file << "  \"" << key << "\": " << val;
    first = false;
  }
  file << "\n}\n";

  file.close();
  std::cout << "Analytical metrics exported to JSON: " << filename << "\n";
}

}  // namespace origami
