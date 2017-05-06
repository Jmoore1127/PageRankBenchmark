/* Pagerank Pipeline Benchmark in C++                          */
/* Copyright 2015 Bradley C. Kuszmaul, bradley@mit.edu         */

#ifndef CSC_HH
#define CSC_HH

// This is a compressed sparse column (CSC or CCS) matrix.
//  * vals is all the nonzeros stored in column-major order.
//  * rows is a vector of the same length as vals that gives the row
//    index for each value.
//  * col-starts tells us the index in vals where each column starts.
// A CSC matrix is useful when performing a matrix*vector product.

#include <cassert>
#include <tuple>
#include <vector>

using namespace std;

template <class T>
struct csc_matrix {
  vector<T> col_starts;
  vector<T> rows;
  vector<double> vals;
  csc_matrix(T NNZ, T N /* matrix side length*/ ) {
  }

  csc_matrix(T N, const vector<tuple<T, T, double>> &nonzeros) {
    // nonzeros must be already sorted into column-major order.
    col_starts.reserve(N + 1);
    rows.reserve(nonzeros.size());
    vals.reserve(nonzeros.size());
    for (const auto & e : nonzeros) {
      const T &row = get<1>(e);
      const T &col = get<0>(e);
      assert(0 <= row && row < N);
      assert(0 <= col && col < N);
      const double &val = get<2>(e);
      while (col_starts.size() <= static_cast<size_t>(col)) {
        col_starts.push_back(rows.size());
      }
      rows.push_back(row);
      vals.push_back((double)val);
    }
    while (col_starts.size() < N+1) {
      col_starts.push_back(rows.size());
    }
  }
};
#endif
