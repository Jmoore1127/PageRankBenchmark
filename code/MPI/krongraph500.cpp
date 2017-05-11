/* Pagerank Pipeline Benchmark in C++                          */
/* Copyright 2015 Bradley C. Kuszmaul, bradley@mit.edu         */

#include "krongraph500.h"

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>

#include <fcntl.h>
#include <unistd.h>



int get_rand() {
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(0,RAND_MAX);
    return distribution(generator);
}

template <class T>
std::vector<std::tuple<T, T>> kronecker(int SCALE, int edges_per_vertex, int nodes) {
  // Half-hearted attempt to check that the T is big enough.
  // Doesn't try do a good job if SCALE and edges_per_vertex are close to 1ul<<64.
  assert(std::numeric_limits<T>::max() >= (1ul << SCALE)*edges_per_vertex);
  T N = T(1)<<SCALE;
  T M = edges_per_vertex * N / nodes;
  double A = 0.57, B = 0.19, C = 0.19;
  double ab = A+B;
  double c_norm = C/(1 - (ab));
  double a_norm = A/(ab);
  uint64_t ab_scaled = RAND_MAX * ab;
  uint64_t c_norm_scaled = RAND_MAX * c_norm;
  uint64_t a_norm_scaled = RAND_MAX * a_norm;

  std::vector<std::tuple<T,T>> edges;
  int chunk_size = M/omp_get_max_threads();
  edges.reserve(chunk_size + 10); //give a little padding to help ensure we don't resize

//#pragma omp declare reduction (merge : std::vector<std::tuple<T,T>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

//#pragma omp parallel for reduction(merge: edges)
  for (T i = 0; i < M; i++) {
    T ij_i = 0, ij_j = 0;
    for (int ib = 0; ib < SCALE; ib++) {
      T r1 = get_rand();
      T r2 = get_rand();
      T ii_bit = r1 > ab_scaled;
      T jj_bit = r2 > (c_norm_scaled * ii_bit + a_norm_scaled * !ii_bit);
      ij_i += ii_bit << ib;
      ij_j += jj_bit << ib;
    }
    //push transposed
    edges.push_back(std::tuple<T,T>(ij_j, ij_i));
  }
  return edges;
}

template std::vector<std::tuple<long, long>> kronecker(int SCALE, int edges_per_vertex, int nodes);

static int appendint_internal(int bufoff, char *buf, uint64_t v) {
  if (v<10) {
    buf[bufoff++] = '0'+v;
    return bufoff;
  } else {
    return appendint_internal(appendint_internal(bufoff, buf, v/10),
                              buf,
                              v%10);
  }
}
static int appendint(int bufoff, char *buf, uint64_t v) {
  if (v==0) {
    buf[bufoff++] = '0';
    return bufoff;
  } else {
    return appendint_internal(bufoff, buf, v);
  }
}