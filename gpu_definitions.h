#ifndef GPU_DEFINITIONS
#define GPU_DEFINITIONS
#include "definitions.h"
#include "stencil_grid_utility.h"
#include "stencil_assembly.h"

struct SolverDataBuffer {
  // cg data
  CTYPE *r;
  CTYPE *p;
  CTYPE *q;
  CTYPE *z;

  // jacobi + mg data
  MTYPE **invD;
  CTYPE **dmg;
  CTYPE **rmg;
  CTYPE **zmg;

  // explicitly assembled matrices
  struct CSRMatrix *coarseMatrices;
  struct CoarseSolverData bottomSolver;
};

struct gpuNode {
    int id;
    struct gpuNode * prev;
    struct gpuNode * next;
    struct gridContext * gc;
    int offset_x;
    int x_global;
};

struct gpuGrid {
  int num_targets;
  struct gpuNode * target;
};

void gridContextToGPUGrid(
    struct gridContext * gc,
    struct gpuGrid * gpu_gc,
    const int nl,
    const int verbose);

void freeGPUGrid(
    struct gpuGrid * gpu_gc,
    const int nl);

void computePadding(
    const struct gridContext * gc,
    const int l,
    int * ncell,
    int * wrapx,
    int * wrapy,
    int * wrapz,
    uint_fast32_t * ndof);

DTYPE compute_volume(
    const struct gridContext gc,
    DTYPE * xPhys);

float compute_change(
    DTYPE * x,
    DTYPE * xnew,
    const int nelem);

void update_solution(
    const DTYPE * x,
    DTYPE * xnew,
    const DTYPE * dv, 
    const DTYPE * dc, 
    const int nelem,
    const DTYPE g);

void enter_data(
    const struct gridContext gridContext,
    const struct SolverDataBuffer solverData,
    const struct gpuGrid gpu_gc,
    const DTYPE * xPhys,
    const DTYPE * x,
    const DTYPE * xnew,
    const STYPE * U,
    const CTYPE * F,
    const DTYPE * dc,
    const int nelem,
    const int nl);

#endif
