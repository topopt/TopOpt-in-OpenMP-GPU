#pragma once

#include "definitions.h"
#include "gpu_definitions.h"

#include "stencil_assembly.h"

void smoothDampedJacobi_halo(const struct gpuNode * gpu_node, const DTYPE *x,
                             const uint_fast32_t nswp, const CTYPE omega,
                             const MTYPE *invD, CTYPE *u, const CTYPE *b,
                             CTYPE *tmp);

void smoothDampedJacobiSubspaceMatrix_halo(
    const struct gridContext * gc, const struct CSRMatrix M, const int l,
    const uint_fast32_t nswp, const CTYPE omega, const MTYPE *invD, CTYPE *u,
    const CTYPE *b, CTYPE *tmp);

void smoothDampedJacobiSubspace_halo(const struct gpuNode * gpu_node,
                                     const DTYPE *x, const int l,
                                     const uint_fast32_t nswp,
                                     const CTYPE omega, const MTYPE *invD,
                                     CTYPE *u, const CTYPE *b, CTYPE *tmp);

void solveStateMG_halo(const struct gpuGrid * gpu_gc, DTYPE *x, const int nswp,
                       const int nl, const CTYPE tol,
                       struct SolverDataBuffer *data, int *finalIter,
                       float *finalRes, CTYPE *b, STYPE *u);

void allocateSolverData(const struct gridContext gc, const int nl,
                        struct SolverDataBuffer *data);

void freeSolverData(struct SolverDataBuffer *data, const int nl);

// compute the norm of two vectors
// temperature: cold-medium, called 2 x number of cg iterations
__force_inline inline CTYPE norm(CTYPE *v, uint_fast32_t size) {
  CTYPE val = 0.0;
  // long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
#pragma omp target teams distribute parallel for reduction(+ : val) map(always,tofrom: val)
  for (uint_fast32_t i = 0; i < size; i++)
    val += v[i] * v[i];
  val = sqrt(val);
  return val;
}

// compute the inner product of two vectors
// temperature: cold-medium, called 2 x number of cg iterations
__force_inline inline CTYPE innerProduct(CTYPE *a, CTYPE *b,
                                         uint_fast32_t size) {
  CTYPE val = 0.0;
  // long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
#pragma omp target teams distribute parallel for reduction(+ : val) map(always,tofrom: val) //firstprivate(size)
  for (uint_fast32_t i = 0; i < size; i++)
    val += a[i] * b[i];
  return val;
}
