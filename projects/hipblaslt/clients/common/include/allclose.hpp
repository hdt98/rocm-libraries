/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#pragma once

#include "allclose.hpp"
#include "cblas.h"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_vector.hpp"
#include "utility.hpp"
#include <cstdio>
#include <hipblaslt/hipblaslt.h>
#include <limits>
#include <set>
#include <memory>

/* =====================================================================
        allclose check: abs(A-B) <= atol + abs(rtol * B)
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides allclose check
 */

/* ============== Allclose Check for General Matrix ============= */
/*! \brief compare the allclose error of two matrices hCPU & hGPU */

template <typename T>
bool allclose(size_t* N, T* a, T* b, double atol, double rtol, bool equal_nan = false, bool verbose = false)
{
    size_t error_count = 0;
    double max_error = 0.0;
    size_t first_error_idx = 0;
    bool found_first_error = false;
    
    for(size_t i = 0; i < *N; i++)
    {
        //Returning ture immediately if 2 elements are identical.
        //This also can handle inf == inf case
        if(a[i] == b[i])
            continue;
        //NaNs compare equal when equal_nan is true
        if(equal_nan && (std::isnan(a[i]) && std::isnan(b[i])))
            continue;
        if(equal_nan && (std::isnan(a[i]) ^ std::isnan(b[i])))
        {
            error_count++;
            if(!found_first_error)
            {
                first_error_idx = i;
                found_first_error = true;
            }
            if(!verbose)
                return false;
            continue;
        }

        double error     = std::abs(a[i] - b[i]);
        double tolerance = atol + std::abs(rtol * b[i]);
        if(!(error <= tolerance))
        {
            error_count++;
            if(error > max_error)
                max_error = error;
            if(!found_first_error)
            {
                first_error_idx = i;
                found_first_error = true;
            }
            if(!verbose)
                return false;
        }
    }
    
    if(verbose && error_count > 0)
    {
        hipblaslt_cout << "=== allclose check failed ===" << std::endl;
        hipblaslt_cout << "Total elements: " << *N << std::endl;
        hipblaslt_cout << "Error elements: " << error_count << std::endl;
        hipblaslt_cout << "Error ratio: " << (100.0 * error_count / *N) << "%" << std::endl;
        hipblaslt_cout << "Max absolute error: " << max_error << std::endl;
        hipblaslt_cout << "First error at index " << first_error_idx << ": "
                       << "CPU=" << a[first_error_idx] << ", GPU=" << b[first_error_idx]
                       << ", diff=" << std::abs(a[first_error_idx] - b[first_error_idx]) << std::endl;
        hipblaslt_cout << "Tolerance: atol=" << atol << ", rtol=" << rtol << std::endl;
    }
    
    return error_count == 0;
}

template <
    typename T,
    std::enable_if_t<!(std::is_same<T, hipblaslt_f8_fnuz>{} || std::is_same<T, hipblaslt_bf8_fnuz>{}
                       || std::is_same<T, hipblaslt_f8>{} || std::is_same<T, hipblaslt_bf8>{}
                       ),
                     int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            T*      hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    std::vector<double> atols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    std::vector<double> rtols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    for(auto& atol : atols)
    {
        for(auto& rtol : rtols)
        {
            if(allclose(&size, hCPU_double.data(), hGPU_double.data(), atol, rtol, false, false))
            {
                hipblaslt_atol = atol;
                hipblaslt_rtol = rtol;
                //early termination for accending rtols
                break;
            }
        }
        //early termination for accending atols
        if(hipblaslt_atol != 1)
            break;
    }

    if(hipblaslt_atol == 1)
    {
        // Failed all tolerance checks, print detailed error info with strictest tolerance
        hipblaslt_cout << "Matrix dimensions: M=" << M << ", N=" << N << ", lda=" << lda << std::endl;
        
        // Find and print first error with matrix coordinates, and collect all error columns
        size_t error_count = 0;
        double max_error = 0.0;
        size_t first_error_idx = 0;
        bool found_first_error = false;
        std::set<int64_t> error_cols;
        std::set<int64_t> non_zero_cols;  // Columns with non-zero elements
        
        for(size_t i = 0; i < size; i++)
        {
            // Check if column has non-zero elements
            int64_t col = i / lda;
            if(std::abs(hCPU_double[i]) > 1e-10 || std::abs(hGPU_double[i]) > 1e-10)
            {
                non_zero_cols.insert(col);
            }
            
            if(hCPU_double[i] == hGPU_double[i])
                continue;
            
            double error = std::abs(hCPU_double[i] - hGPU_double[i]);
            double tolerance = 1e-6 + std::abs(1e-6 * hGPU_double[i]);
            if(!(error <= tolerance))
            {
                error_count++;
                if(error > max_error)
                    max_error = error;
                if(!found_first_error)
                {
                    first_error_idx = i;
                    found_first_error = true;
                }
                // Collect column number (column-major: index = row + col * lda)
                error_cols.insert(col);
            }
        }
        
        if(found_first_error)
        {
            // Convert linear index to matrix coordinates (column-major: index = row + col * lda)
            int64_t col = first_error_idx / lda;
            int64_t row = first_error_idx % lda;
            
            hipblaslt_cout << "=== allclose check failed ===" << std::endl;
            hipblaslt_cout << "Total elements: " << size << std::endl;
            hipblaslt_cout << "Error elements: " << error_count << std::endl;
            hipblaslt_cout << "Error ratio: " << (100.0 * error_count / size) << "%" << std::endl;
            hipblaslt_cout << "Max absolute error: " << max_error << std::endl;
            hipblaslt_cout << "First error at index " << first_error_idx 
                          << " (row=" << row << ", col=" << col << "): "
                          << "CPU=" << hCPU_double[first_error_idx] 
                          << ", GPU=" << hGPU_double[first_error_idx]
                          << ", diff=" << std::abs(hCPU_double[first_error_idx] - hGPU_double[first_error_idx]) 
                          << std::endl;
            
            // Print column statistics
            size_t total_cols = N;
            size_t error_col_count = error_cols.size();
            size_t correct_col_count = total_cols - error_col_count;
            hipblaslt_cout << "Column statistics: Total=" << total_cols 
                          << ", Error=" << error_col_count 
                          << ", Correct=" << correct_col_count << std::endl;
            
            // Print error columns (first 32 and last 32 if more than 64, otherwise all)
            if(!error_cols.empty())
            {
                std::vector<int64_t> error_cols_vec(error_cols.begin(), error_cols.end());
                hipblaslt_cout << "Error columns: ";
                bool first = true;
                
                if(error_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t err_col : error_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << err_col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << error_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (error_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = error_cols_vec.size() - 32; i < error_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << error_cols_vec[i];
                        if(i < error_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
            
            // Print columns with non-zero elements
            if(!non_zero_cols.empty())
            {
                std::vector<int64_t> non_zero_cols_vec(non_zero_cols.begin(), non_zero_cols.end());
                hipblaslt_cout << "Non-zero columns: ";
                bool first = true;
                
                if(non_zero_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t col : non_zero_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << non_zero_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (non_zero_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = non_zero_cols_vec.size() - 32; i < non_zero_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << non_zero_cols_vec[i];
                        if(i < non_zero_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
            
            // Print correct columns (first 32 and last 32 if more than 64, otherwise all)
            if(correct_col_count > 0)
            {
                std::vector<int64_t> correct_cols_vec;
                for(int64_t col = 0; col < static_cast<int64_t>(total_cols); col++)
                {
                    if(error_cols.find(col) == error_cols.end())
                    {
                        correct_cols_vec.push_back(col);
                    }
                }
                
                hipblaslt_cout << "Correct columns: ";
                bool first = true;
                
                if(correct_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t corr_col : correct_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << corr_col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << correct_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (correct_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = correct_cols_vec.size() - 32; i < correct_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << correct_cols_vec[i];
                        if(i < correct_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
        }
        
        return false;
    }

    return true;
}

template <typename T,
          std::enable_if_t<(std::is_same<T, hipblaslt_f8_fnuz>{}
                            || std::is_same<T, hipblaslt_bf8_fnuz>{}),
                           int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            T*      hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(float(hCPU[idx]));
            hGPU_double[idx] = double(float(hGPU[idx]));
        }
    }

    std::vector<double> atols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    std::vector<double> rtols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    for(auto& atol : atols)
    {
        for(auto& rtol : rtols)
        {
            if(allclose(&size, hCPU_double.data(), hGPU_double.data(), atol, rtol, false, false))
            {
                hipblaslt_atol = atol;
                hipblaslt_rtol = rtol;
                //early termination for accending rtols
                break;
            }
        }
        //early termination for accending atols
        if(hipblaslt_atol != 1)
            break;
    }

    if(hipblaslt_atol == 1)
    {
        // Failed all tolerance checks, print detailed error info with strictest tolerance
        hipblaslt_cout << "Matrix dimensions: M=" << M << ", N=" << N << ", lda=" << lda << std::endl;
        
        // Find and print first error with matrix coordinates, and collect all error columns
        size_t error_count = 0;
        double max_error = 0.0;
        size_t first_error_idx = 0;
        bool found_first_error = false;
        std::set<int64_t> error_cols;
        std::set<int64_t> non_zero_cols;  // Columns with non-zero elements
        
        for(size_t i = 0; i < size; i++)
        {
            // Check if column has non-zero elements
            int64_t col = i / lda;
            if(std::abs(hCPU_double[i]) > 1e-10 || std::abs(hGPU_double[i]) > 1e-10)
            {
                non_zero_cols.insert(col);
            }
            
            if(hCPU_double[i] == hGPU_double[i])
                continue;
            
            double error = std::abs(hCPU_double[i] - hGPU_double[i]);
            double tolerance = 1e-6 + std::abs(1e-6 * hGPU_double[i]);
            if(!(error <= tolerance))
            {
                error_count++;
                if(error > max_error)
                    max_error = error;
                if(!found_first_error)
                {
                    first_error_idx = i;
                    found_first_error = true;
                }
                // Collect column number (column-major: index = row + col * lda)
                error_cols.insert(col);
            }
        }
        
        if(found_first_error)
        {
            // Convert linear index to matrix coordinates (column-major: index = row + col * lda)
            int64_t col = first_error_idx / lda;
            int64_t row = first_error_idx % lda;
            
            hipblaslt_cout << "=== allclose check failed ===" << std::endl;
            hipblaslt_cout << "Total elements: " << size << std::endl;
            hipblaslt_cout << "Error elements: " << error_count << std::endl;
            hipblaslt_cout << "Error ratio: " << (100.0 * error_count / size) << "%" << std::endl;
            hipblaslt_cout << "Max absolute error: " << max_error << std::endl;
            hipblaslt_cout << "First error at index " << first_error_idx 
                          << " (row=" << row << ", col=" << col << "): "
                          << "CPU=" << hCPU_double[first_error_idx] 
                          << ", GPU=" << hGPU_double[first_error_idx]
                          << ", diff=" << std::abs(hCPU_double[first_error_idx] - hGPU_double[first_error_idx]) 
                          << std::endl;
            
            // Print column statistics
            size_t total_cols = N;
            size_t error_col_count = error_cols.size();
            size_t correct_col_count = total_cols - error_col_count;
            hipblaslt_cout << "Column statistics: Total=" << total_cols 
                          << ", Error=" << error_col_count 
                          << ", Correct=" << correct_col_count << std::endl;
            
            // Print error columns (first 32 and last 32 if more than 64, otherwise all)
            if(!error_cols.empty())
            {
                std::vector<int64_t> error_cols_vec(error_cols.begin(), error_cols.end());
                hipblaslt_cout << "Error columns: ";
                bool first = true;
                
                if(error_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t err_col : error_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << err_col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << error_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (error_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = error_cols_vec.size() - 32; i < error_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << error_cols_vec[i];
                        if(i < error_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
            
            // Print columns with non-zero elements
            if(!non_zero_cols.empty())
            {
                std::vector<int64_t> non_zero_cols_vec(non_zero_cols.begin(), non_zero_cols.end());
                hipblaslt_cout << "Non-zero columns: ";
                bool first = true;
                
                if(non_zero_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t col : non_zero_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << non_zero_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (non_zero_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = non_zero_cols_vec.size() - 32; i < non_zero_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << non_zero_cols_vec[i];
                        if(i < non_zero_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
            
            // Print correct columns (first 32 and last 32 if more than 64, otherwise all)
            if(correct_col_count > 0)
            {
                std::vector<int64_t> correct_cols_vec;
                for(int64_t col = 0; col < static_cast<int64_t>(total_cols); col++)
                {
                    if(error_cols.find(col) == error_cols.end())
                    {
                        correct_cols_vec.push_back(col);
                    }
                }
                
                hipblaslt_cout << "Correct columns: ";
                bool first = true;
                
                if(correct_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t corr_col : correct_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << corr_col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << correct_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (correct_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = correct_cols_vec.size() - 32; i < correct_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << correct_cols_vec[i];
                        if(i < correct_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
        }
        
        return false;
    }

    return true;
}

template <typename T,
          std::enable_if_t<(std::is_same<T, hipblaslt_f8>{} || std::is_same<T, hipblaslt_bf8>{}),
                           int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            T*      hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(float(hCPU[idx]));
            hGPU_double[idx] = double(float(hGPU[idx]));
        }
    }

    std::vector<double> atols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    std::vector<double> rtols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    for(auto& atol : atols)
    {
        for(auto& rtol : rtols)
        {
            if(allclose(&size, hCPU_double.data(), hGPU_double.data(), atol, rtol, false, false))
            {
                hipblaslt_atol = atol;
                hipblaslt_rtol = rtol;
                //early termination for accending rtols
                break;
            }
        }
        //early termination for accending atols
        if(hipblaslt_atol != 1)
            break;
    }

    if(hipblaslt_atol == 1)
    {
        // Failed all tolerance checks, print detailed error info with strictest tolerance
        hipblaslt_cout << "Matrix dimensions: M=" << M << ", N=" << N << ", lda=" << lda << std::endl;
        
        // Find and print first error with matrix coordinates, and collect all error columns
        size_t error_count = 0;
        double max_error = 0.0;
        size_t first_error_idx = 0;
        bool found_first_error = false;
        std::set<int64_t> error_cols;
        std::set<int64_t> non_zero_cols;  // Columns with non-zero elements
        
        for(size_t i = 0; i < size; i++)
        {
            // Check if column has non-zero elements
            int64_t col = i / lda;
            if(std::abs(hCPU_double[i]) > 1e-10 || std::abs(hGPU_double[i]) > 1e-10)
            {
                non_zero_cols.insert(col);
            }
            
            if(hCPU_double[i] == hGPU_double[i])
                continue;
            
            double error = std::abs(hCPU_double[i] - hGPU_double[i]);
            double tolerance = 1e-6 + std::abs(1e-6 * hGPU_double[i]);
            if(!(error <= tolerance))
            {
                error_count++;
                if(error > max_error)
                    max_error = error;
                if(!found_first_error)
                {
                    first_error_idx = i;
                    found_first_error = true;
                }
                // Collect column number (column-major: index = row + col * lda)
                error_cols.insert(col);
            }
        }
        
        if(found_first_error)
        {
            // Convert linear index to matrix coordinates (column-major: index = row + col * lda)
            int64_t col = first_error_idx / lda;
            int64_t row = first_error_idx % lda;
            
            hipblaslt_cout << "=== allclose check failed ===" << std::endl;
            hipblaslt_cout << "Total elements: " << size << std::endl;
            hipblaslt_cout << "Error elements: " << error_count << std::endl;
            hipblaslt_cout << "Error ratio: " << (100.0 * error_count / size) << "%" << std::endl;
            hipblaslt_cout << "Max absolute error: " << max_error << std::endl;
            hipblaslt_cout << "First error at index " << first_error_idx 
                          << " (row=" << row << ", col=" << col << "): "
                          << "CPU=" << hCPU_double[first_error_idx] 
                          << ", GPU=" << hGPU_double[first_error_idx]
                          << ", diff=" << std::abs(hCPU_double[first_error_idx] - hGPU_double[first_error_idx]) 
                          << std::endl;
            
            // Print column statistics
            size_t total_cols = N;
            size_t error_col_count = error_cols.size();
            size_t correct_col_count = total_cols - error_col_count;
            hipblaslt_cout << "Column statistics: Total=" << total_cols 
                          << ", Error=" << error_col_count 
                          << ", Correct=" << correct_col_count << std::endl;
            
            // Print error columns (first 32 and last 32 if more than 64, otherwise all)
            if(!error_cols.empty())
            {
                std::vector<int64_t> error_cols_vec(error_cols.begin(), error_cols.end());
                hipblaslt_cout << "Error columns: ";
                bool first = true;
                
                if(error_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t err_col : error_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << err_col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << error_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (error_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = error_cols_vec.size() - 32; i < error_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << error_cols_vec[i];
                        if(i < error_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
            
            // Print columns with non-zero elements
            if(!non_zero_cols.empty())
            {
                std::vector<int64_t> non_zero_cols_vec(non_zero_cols.begin(), non_zero_cols.end());
                hipblaslt_cout << "Non-zero columns: ";
                bool first = true;
                
                if(non_zero_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t col : non_zero_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << non_zero_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (non_zero_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = non_zero_cols_vec.size() - 32; i < non_zero_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << non_zero_cols_vec[i];
                        if(i < non_zero_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
            
            // Print correct columns (first 32 and last 32 if more than 64, otherwise all)
            if(correct_col_count > 0)
            {
                std::vector<int64_t> correct_cols_vec;
                for(int64_t col = 0; col < static_cast<int64_t>(total_cols); col++)
                {
                    if(error_cols.find(col) == error_cols.end())
                    {
                        correct_cols_vec.push_back(col);
                    }
                }
                
                hipblaslt_cout << "Correct columns: ";
                bool first = true;
                
                if(correct_cols_vec.size() <= 64)
                {
                    // Print all if <= 64
                    for(int64_t corr_col : correct_cols_vec)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << corr_col;
                        first = false;
                    }
                }
                else
                {
                    // Print first 32
                    for(size_t i = 0; i < 32; i++)
                    {
                        if(!first)
                            hipblaslt_cout << ", ";
                        hipblaslt_cout << correct_cols_vec[i];
                        first = false;
                    }
                    hipblaslt_cout << ", ... (" << (correct_cols_vec.size() - 64) << " omitted), ";
                    // Print last 32
                    for(size_t i = correct_cols_vec.size() - 32; i < correct_cols_vec.size(); i++)
                    {
                        hipblaslt_cout << correct_cols_vec[i];
                        if(i < correct_cols_vec.size() - 1)
                            hipblaslt_cout << ", ";
                    }
                }
                hipblaslt_cout << std::endl;
            }
        }
        
        return false;
    }

    return true;
}
// For BF16 and half, we convert the results to double first
template <
    typename T,
    typename VEC,
    std::enable_if_t<std::is_same<T, hipblasLtHalf>{} || std::is_same<T, hip_bfloat16>{}, int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            VEC&&   hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = hCPU[idx];
            hGPU_double[idx] = hGPU[idx];
        }
    }

    return allclose_check_general<double>(
        allclose_type, M, N, lda, hCPU_double, hGPU_double, hipblaslt_atol, hipblaslt_rtol);
}

// For int8, we convert the results to int first
template <typename T, typename VEC, std::enable_if_t<std::is_same<T, int8_t>{}, int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            VEC&&   hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t           size = N * (size_t)lda;
    host_vector<int> hCPU_int(size);
    host_vector<int> hGPU_int(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx    = j + i * (size_t)lda;
            hCPU_int[idx] = hCPU[idx];
            hGPU_int[idx] = hGPU[idx];
        }
    }

    return allclose_check_general<int>(
        allclose_type, M, N, lda, hCPU_int, hGPU_int, hipblaslt_atol, hipblaslt_rtol);
}

/* ============== allclose check for strided_batched case ============= */
template <typename T, typename T_hpa>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            int64_t stride_a,
                            T_hpa*  hCPU,
                            T*      hGPU,
                            int64_t batch_count,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;

    for(size_t i = 0; i < batch_count; i++)
    {
        auto index = i * stride_a;
        bool close = allclose_check_general(
            allclose_type, M, N, lda, hCPU + index, hGPU + index, hipblaslt_atol, hipblaslt_rtol);
        if(!close)
            return false;
    }

    return true;
}

/* ============== allclose check for batched case ============= */
template <typename T>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            T*      hCPU[],
                            T*      hGPU[],
                            int64_t batch_count,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;

    for(int64_t i = 0; i < batch_count; i++)
    {
        auto index = i;
        bool close = allclose_check_general<T>(
            allclose_type, M, N, lda, hCPU[index], hGPU[index], hipblaslt_atol, hipblaslt_rtol);
        if(!close)
            return false;
    }

    return true;
}

bool allclose_check_general(char        allclose_type,
                            int64_t     M,
                            int64_t     N,
                            int64_t     lda,
                            int64_t     stride_a,
                            void*       hCPU,
                            void*       hGPU,
                            int64_t     batch_count,
                            double&     hipblaslt_atol,
                            double&     hipblaslt_rtol,
                            hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return allclose_check_general<float>(allclose_type,
                                             M,
                                             N,
                                             lda,
                                             stride_a,
                                             static_cast<float*>(hCPU),
                                             static_cast<float*>(hGPU),
                                             batch_count,
                                             hipblaslt_atol,
                                             hipblaslt_rtol);
    case HIP_R_64F:
        return allclose_check_general<double>(allclose_type,
                                              M,
                                              N,
                                              lda,
                                              stride_a,
                                              static_cast<double*>(hCPU),
                                              static_cast<double*>(hGPU),
                                              batch_count,
                                              hipblaslt_atol,
                                              hipblaslt_rtol);
    case HIP_R_16F:
        return allclose_check_general<hipblasLtHalf>(allclose_type,
                                                     M,
                                                     N,
                                                     lda,
                                                     stride_a,
                                                     static_cast<hipblasLtHalf*>(hCPU),
                                                     static_cast<hipblasLtHalf*>(hGPU),
                                                     batch_count,
                                                     hipblaslt_atol,
                                                     hipblaslt_rtol);
    case HIP_R_16BF:
        return allclose_check_general<hip_bfloat16>(allclose_type,
                                                    M,
                                                    N,
                                                    lda,
                                                    stride_a,
                                                    static_cast<hip_bfloat16*>(hCPU),
                                                    static_cast<hip_bfloat16*>(hGPU),
                                                    batch_count,
                                                    hipblaslt_atol,
                                                    hipblaslt_rtol);
    case HIP_R_8F_E4M3_FNUZ:
        return allclose_check_general<hipblaslt_f8_fnuz>(allclose_type,
                                                         M,
                                                         N,
                                                         lda,
                                                         stride_a,
                                                         static_cast<hipblaslt_f8_fnuz*>(hCPU),
                                                         static_cast<hipblaslt_f8_fnuz*>(hGPU),
                                                         batch_count,
                                                         hipblaslt_atol,
                                                         hipblaslt_rtol);
    case HIP_R_8F_E5M2_FNUZ:
        return allclose_check_general<hipblaslt_bf8_fnuz>(allclose_type,
                                                          M,
                                                          N,
                                                          lda,
                                                          stride_a,
                                                          static_cast<hipblaslt_bf8_fnuz*>(hCPU),
                                                          static_cast<hipblaslt_bf8_fnuz*>(hGPU),
                                                          batch_count,
                                                          hipblaslt_atol,
                                                          hipblaslt_rtol);
    case HIP_R_8F_E4M3:
        return allclose_check_general<hipblaslt_f8>(allclose_type,
                                                    M,
                                                    N,
                                                    lda,
                                                    stride_a,
                                                    static_cast<hipblaslt_f8*>(hCPU),
                                                    static_cast<hipblaslt_f8*>(hGPU),
                                                    batch_count,
                                                    hipblaslt_atol,
                                                    hipblaslt_rtol);
    case HIP_R_8F_E5M2:
        return allclose_check_general<hipblaslt_bf8>(allclose_type,
                                                     M,
                                                     N,
                                                     lda,
                                                     stride_a,
                                                     static_cast<hipblaslt_bf8*>(hCPU),
                                                     static_cast<hipblaslt_bf8*>(hGPU),
                                                     batch_count,
                                                     hipblaslt_atol,
                                                     hipblaslt_rtol);
    case HIP_R_32I:
        return allclose_check_general<int32_t>(allclose_type,
                                               M,
                                               N,
                                               lda,
                                               stride_a,
                                               static_cast<int32_t*>(hCPU),
                                               static_cast<int32_t*>(hGPU),
                                               batch_count,
                                               hipblaslt_atol,
                                               hipblaslt_rtol);
    case HIP_R_8I:
        return allclose_check_general<hipblasLtInt8>(allclose_type,
                                                     M,
                                                     N,
                                                     lda,
                                                     stride_a,
                                                     static_cast<hipblasLtInt8*>(hCPU),
                                                     static_cast<hipblasLtInt8*>(hGPU),
                                                     batch_count,
                                                     hipblaslt_atol,
                                                     hipblaslt_rtol);
    default:
        hipblaslt_cerr << "Error type in allclose_check_general" << std::endl;
        return false;
    }
}
