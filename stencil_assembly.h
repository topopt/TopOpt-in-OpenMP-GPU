#pragma once

#include "definitions.h"

// #include "/zhome/c3/7/127558/SuiteSparse/SuiteSparse-5.10.1/include/cholmod.h"
#include "cholmod.h"

struct CSRMatrix {
  uint64_t nnz;
  int32_t nrows;

  int *rowOffsets;
  int *colIndex;
  MTYPE *vals;
};

struct CoarseSolverData {
  cholmod_common *cholmodCommon;
  cholmod_sparse *sparseMatrix;
  cholmod_factor *factoredMatrix;

  cholmod_dense *rhs;
  cholmod_dense *solution;

  cholmod_dense *Y_workspace;
  cholmod_dense *E_workspace;
};

void allocateSubspaceMatrix(const struct gridContext gc, const int l,
                            struct CSRMatrix *M);

void freeSubspaceMatrix(struct CSRMatrix *M);

void assembleSubspaceMatrix(const struct gridContext gc, const int l,
                            const DTYPE *x, struct CSRMatrix M, MTYPE *tmp);

void applyStateOperatorSubspaceMatrix(const struct gridContext gc, const int l,
                                      const struct CSRMatrix M, const CTYPE *in,
                                      CTYPE *out);

void initializeCholmod(const struct gridContext gc, const int l,
                       struct CoarseSolverData *ssolverData,
                       const struct CSRMatrix M);

void finishCholmod(const struct gridContext gc, const int l,
                   struct CoarseSolverData *solverData,
                   const struct CSRMatrix M, const int verbose);

void factorizeSubspaceMatrix(const struct gridContext gc, const int l,
                             struct CoarseSolverData solverData,
                             const struct CSRMatrix M);

void solveSubspaceMatrix(const struct gridContext gc, const int l,
                         struct CoarseSolverData solverData, const CTYPE *in,
                         CTYPE *out);
