/* Pagerank Pipeline Benchmark in C++                          */
/* Copyright 2015 Bradley C. Kuszmaul, bradley@mit.edu         */

#ifndef KRONGRAPH500_HH
#define KRONGRAPH500_HH

#include <tuple>
#include <vector>

template <class T>
std::vector<std::tuple<T, T>> kronecker(int SCALE, int edges_per_vertex, int nodes);
// Effect: Create an edge list according to the Graph500 randomized Kronecker graph.
//   SCALE is the log (base 2) of the total number of vertice.s
//   edges_per_vertex is the average number of edges per node
//   Duplicate edges may be returned
//  Requires that ((1<<SCALE) * edges_per_vertex) fits in T.
#endif
