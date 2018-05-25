/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cutlass/matrix_traits.h>
#include <tools/util/command_line.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace perf {

/// Outcome of test
struct Disposition {
  enum Kind { Unknown = 0, NotRun, Passed, Incorrect, Failed, NotVerified, Invalid };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace perf

inline std::ostream &operator<<(std::ostream &out, perf::Disposition::Kind value) {
  char const *str[] = {
      "unknown", "not_run", "passed", "incorrect", "failed", "not_verified", "invalid"};
  if (value >= perf::Disposition::Unknown && value < perf::Disposition::Invalid) {
    out << str[value];
  } else {
    out << str[perf::Disposition::Invalid];
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Outputs matrix layout
inline std::ostream &operator<<(std::ostream &out, cutlass::MatrixLayout::Kind layout) {
  out << (layout == cutlass::MatrixLayout::kColumnMajor ? "column" : "row");
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Size and layout of a GEMM problem
struct GemmProblem {
  //
  // Data members
  //

  int m;
  int n;
  int k;
  cutlass::MatrixLayout::Kind layout_A;
  cutlass::MatrixLayout::Kind layout_B;

  double alpha;
  double beta;

  //
  // Static function members
  //

  /// Static method to print GemmProblem headers
  static std::string header() { return "M, N, K, Layout_A, Layout_B, Beta"; }

  //
  // Methods
  //

  GemmProblem(int _m = 0,
              int _n = 0,
              int _k = 0,
              cutlass::MatrixLayout::Kind _layout_A = cutlass::MatrixLayout::kColumnMajor,
              cutlass::MatrixLayout::Kind _layout_B = cutlass::MatrixLayout::kRowMajor,
              double _alpha = 1,
              double _beta = 0)
      : m(_m), n(_n), k(_k), layout_A(_layout_A), layout_B(_layout_B), alpha(_alpha), beta(_beta) {}

  /// leading dimension of A
  int lda() const {
    if (layout_A == cutlass::MatrixLayout::kColumnMajor) {
      return m;
    }
    return k;
  }

  /// leading dimension of B
  int ldb() const {
    if (layout_B == cutlass::MatrixLayout::kColumnMajor) {
      return k;
    }
    return n;
  }

  /// leading dimension of C
  int ldc() const { return m; }

  /// Pretty prints output
  std::ostream &pretty_print(std::ostream &out) const {
    out << m << "-by-" << n << "-by-" << k << ", A: " << layout_A << "-major, B: " << layout_B
        << "-major, beta: " << beta;

    return out;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace perf

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Prints a problem to an output stream
inline std::ostream &operator<<(std::ostream &out, perf::GemmProblem const &problem) {
  out << problem.m << ", " << problem.n << ", " << problem.k << ", " << problem.layout_A << ", "
      << problem.layout_B << ", " << problem.beta;

  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Result object
struct PerformanceResult {

  /// Name of kernel
  std::string kernel_name;

  /// Problem size
  GemmProblem problem;

  /// Outcome of test
  Disposition::Kind disposition;

  /// Runtime in ms
  double runtime;

  /// Throughput in units of GFLOPs
  double gflops;
  //
  // Methods
  //

  PerformanceResult(
                    std::string const &_kernel_name = "",
                    GemmProblem const &_problem = GemmProblem(),
                    Disposition::Kind _disposition = Disposition::NotRun,
                    double _runtime = 0,
                    double _gflops = 0)
      :
        kernel_name(_kernel_name),
        problem(_problem),
        disposition(_disposition),
        runtime(_runtime),
        gflops(_gflops) {}

  /// Displays headers
  static std::string header() {
    return std::string("Kernel, ") + GemmProblem::header() +
           ", Disposition, Runtime, GFLOPs";
  }

  /// Prints human-readable results
  std::ostream &pretty_print(std::ostream &out) const {

    out << "Kernel: \033[1m" << kernel_name << "\033[0m\n"
        << "    problem: ";

    std::stringstream disposition_str;
    if (disposition == Disposition::Passed) {
      disposition_str << "\033[1m";
    }
    else {
      disposition_str << "\033[1;31m";
    }
    disposition_str << disposition << "\033[0m";

    problem.pretty_print(out) << "\n"
                              << "    disposition: " << disposition_str.str() << "\n"
                              << "    runtime:     " << runtime << " ms\n\n"
                              << "    performance: \033[1m" << gflops << " GFLOPs\033[0m\n\n";

    return out;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace perf

/// Outputs result
inline std::ostream &operator<<(std::ostream &out, perf::PerformanceResult const &result) {

  out << result.kernel_name << ", " << result.problem << ", "
      << result.disposition << ", " << result.runtime << ", " << result.gflops;

  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
