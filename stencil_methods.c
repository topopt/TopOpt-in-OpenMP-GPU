#include "stencil_methods.h"

#include "stencil_utility.h"

#include <cblas.h>

void applyStateOperator_stencil(const struct gpuNode * gpu_node, const DTYPE *x,
                                const CTYPE *in, CTYPE *out) {
  const struct gridContext gc = *(gpu_node->gc);
  const uint32_t nx = gc.nelx + 1;
  const uint32_t ny = gc.nely + 1;
  const uint32_t nz = gc.nelz + 1;

  // this is necessary for omp to recognize that gc.precomputedKE[0] is already
  // mapped
  const MTYPE *precomputedKE = gc.precomputedKE[0];
  const int wrapy = gc.wrapy;
  const int wrapz = gc.wrapz;

  // loop over elements, depends on the which level you are on. For the finest
  // (level 0) nelx*nely*nelz = 100.000 or more, but for every level you go down
  // the number of iterations reduce by a factor of 8. i.e. level 2 will only
  // have ~1000. This specific loop accounts for ~90% runtime
  //#pragma omp teams distribute parallel for collapse(3) schedule(static)

#pragma omp target teams distribute parallel for schedule(static) collapse(3) device(gpu_node->id)
  for (int32_t i = 1; i < nx + 1; i += STENCIL_SIZE_X) {
    for (int32_t k = 1; k < nz + 1; k += STENCIL_SIZE_Z) {
      for (int32_t j = 1; j < ny + 1; j += STENCIL_SIZE_Y) {

        alignas(__alignBound) MTYPE out_x[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE out_y[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE out_z[STENCIL_SIZE_Y];

        alignas(__alignBound) MTYPE in_x[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE in_y[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE in_z[STENCIL_SIZE_Y];

// zero the values about to be written in this
        #pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)               \
            aligned(out_x, out_y, out_z                                                \
            : __alignBound)
        for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {
          out_x[jj] = 0.0;
          out_y[jj] = 0.0;
          out_z[jj] = 0.0;
        }

        // center line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){0, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){0, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){0, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){1, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){-1, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){-1, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){1, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, -1, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 1, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){1, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){-1, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        #pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)
        for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {
          const uint_fast32_t offset =
              3 * (i * wrapy * wrapz + k * wrapy + j + jj);
          out[offset + 0] = out_x[jj];
          out[offset + 1] = out_y[jj];
          out[offset + 2] = out_z[jj];
        }
      }
    }
  }
  
// zero out the extra padded nodes
#pragma omp target teams distribute parallel for collapse(3) schedule(static) device(gpu_node->id)
  for (int32_t i = 0; i < gc.wrapx; i++)
    for (int32_t k = 0; k < gc.wrapz; k++)
      for (int32_t j = ny + 1; j < gc.wrapy; j++) {

        const uint_fast32_t offset =
            3 * (i * wrapy * wrapz + k * wrapy + j);

        out[offset + 0] = 0.0;
        out[offset + 1] = 0.0;
        out[offset + 2] = 0.0;
      }

  const uint_fast32_t n = gc.fixedDofs[0].n;
  const uint_fast32_t *fidx = gc.fixedDofs[0].idx;

// apply boundaryConditions
#pragma omp target teams distribute parallel for schedule(static) device(gpu_node->id)
  for (int i = 0; i < n; i++) {
    out[fidx[i]] = in[fidx[i]];
  }
}


// Apply the global matrix vector product out = K * in
// temperature: very hot, called ~25 x (number of mg levels [1-5]) x
// (number of cg iterations [125-2500]) = [125-12500]  times pr design
// iteration

void applyStateOperatorSubspace_halo(const struct gpuNode * gpu_node, const int l,
                                     const DTYPE *x, CTYPE *in, CTYPE *out) {
  const struct gridContext gc = *(gpu_node->gc);
  
  // Computing dimensions for coarse grid
  int ncell,wrapxc,wrapyc,wrapzc;
  uint_fast32_t ndofc;
  computePadding(&gc,l,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndofc);

  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  MTYPE* KE = gc.precomputedKE[l];
  uint_fast32_t * idx = gc.fixedDofs[l].idx;
  uint_fast32_t n = gc.fixedDofs[l].n;

  #pragma omp target teams distribute parallel for schedule(static) device(gpu_node->id)
  for (uint32_t i = 0; i < ndofc; i++)
    out[i] = 0.0;

  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)
              #pragma omp target teams distribute parallel for collapse(4) schedule(static) device(gpu_node->id)
              for (int32_t i = bx + 1; i < nelxc + 1; i += 2)
                for (int32_t k = bz + 1; k < nelzc + 1; k += 2)
                  for (int32_t j = by + 1; j < nelyc + 1; j += 2) 
                    for (int imv=0;imv<24;imv++){
                      MTYPE out_local = 0.0;
                      uint_fast32_t idx_update = 0;
                      for (int kt=0;kt<2;kt++){
                        for (int jt=0;jt<2;jt++){
                          for (int it=0;it<2;it++){
                              int nx = i + it;
                              if (jt == 1){
                                nx = i + 1-it;
                              }
                              const int pos = 3*(kt*4+jt*2+it);
                              const int nz = k + kt;
                              const int ny = j + 1-jt;
                              const uint_fast32_t nIndex = nx * wrapyc * wrapzc + nz * wrapyc + ny;
                              uint_fast32_t edof = 3 * nIndex;
                              for (int off1 = 0;off1<3;off1++){
                                if (pos + off1 == imv){
                                  idx_update = edof+off1;
                                }
                              }
                              for (int ii = 0; ii < ncell; ii++)
                                for (int kk = 0; kk < ncell; kk++)
                                  for (int jj = 0; jj < ncell; jj++){
                                    const int ifine = ((i - 1) * ncell) + ii + 1;
                                    const int jfine = ((j - 1) * ncell) + jj + 1;
                                    const int kfine = ((k - 1) * ncell) + kk + 1;

                                    const int cellidx = 24 * 24 * (ncell * ncell * ii + ncell * kk + jj);

                                    const uint_fast32_t elementIndex =
                                        ifine * (gc.wrapy - 1) * (gc.wrapz - 1) +
                                        kfine * (gc.wrapy - 1) + jfine;
                                    const MTYPE elementScale =
                                        gc.Emin + x[elementIndex] * x[elementIndex] *
                                        x[elementIndex] * (gc.E0 - gc.Emin);
                                    for (int off = 0;off<3;off++){
                                      out_local += elementScale*KE[cellidx+imv*24+pos+off]*((MTYPE)in[edof+off]);
                                    }
                                  }
                              
                          }
                        }
                      }
                      #pragma omp atomic update
                      out[idx_update] += (CTYPE)out_local;
                    }
                  
            

            // apply boundaryConditions
//#pragma omp parallel for schedule(static) default(none) shared(gc,in,out)
#pragma omp target teams distribute parallel for schedule(static) device(gpu_node->id)//default(none) shared(gc,in,out)
  for (int i = 0; i < n; i++) {
    out[idx[i]] = in[idx[i]];
  }
}

void projectToFinerGrid_halo(const struct gpuNode * gpu_node,
                             /*in*/ const int l,   /*in*/
                             const CTYPE *ucoarse, /*in*/
                             CTYPE *ufine /*out*/) {
  const struct gridContext * gc = gpu_node->gc;

  // Computing dimensions for fine grid
  int ncellf,wrapxf,wrapyf,wrapzf;
  uint_fast32_t ndoff;
  computePadding(gc,l,&ncellf,&wrapxf,&wrapyf,&wrapzf,&ndoff);

  // Computing dimensions for coarse grid
  int ncellc,wrapxc,wrapyc,wrapzc;
  uint_fast32_t ndofc;
  computePadding(gc,l+1,&ncellc,&wrapxc,&wrapyc,&wrapzc,&ndofc);

  const int nelxf = gc->nelx / ncellf;
  const int nelyf = gc->nely / ncellf;
  const int nelzf = gc->nelz / ncellf;

  const int nxf = nelxf + 1;
  const int nyf = nelyf + 1;
  const int nzf = nelzf + 1;

  // loop over nodes, usually very large with nx*ny*nz = 100.000 or more
  #pragma omp target teams distribute parallel for collapse(4) schedule(static) device(gpu_node->id)
  for (int32_t ifine = 1; ifine < nxf + 1; ifine++)
    for (int32_t kfine = 1; kfine < nzf + 1; kfine++)
      for (int32_t jfine = 1; jfine < nyf + 1; jfine++) {
        for(int offset = 0;offset<3;offset++){

        const uint32_t fineIndex =
            ifine * wrapyf * wrapzf + kfine * wrapyf + jfine;

        const uint32_t icoarse1 = (ifine - 1) / 2 + 1;
        const uint32_t icoarse2 = (ifine) / 2 + 1;
        const uint32_t jcoarse1 = (jfine - 1) / 2 + 1;
        const uint32_t jcoarse2 = (jfine) / 2 + 1;
        const uint32_t kcoarse1 = (kfine - 1) / 2 + 1;
        const uint32_t kcoarse2 = (kfine) / 2 + 1;

        // Node indices on coarse grid
        const uint_fast32_t coarseIndex1 =
            icoarse1 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex2 =
            icoarse2 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex3 =
            icoarse2 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex4 =
            icoarse1 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex5 =
            icoarse1 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex6 =
            icoarse2 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex7 =
            icoarse2 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex8 =
            icoarse1 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse1;

        ufine[3 * fineIndex + offset] = 0.125 * ucoarse[3 * coarseIndex1 + offset] +
                                   0.125 * ucoarse[3 * coarseIndex2 + offset] +
                                   0.125 * ucoarse[3 * coarseIndex3 + offset] +
                                   0.125 * ucoarse[3 * coarseIndex4 + offset] +
                                   0.125 * ucoarse[3 * coarseIndex5 + offset] +
                                   0.125 * ucoarse[3 * coarseIndex6 + offset] +
                                   0.125 * ucoarse[3 * coarseIndex7 + offset] +
                                   0.125 * ucoarse[3 * coarseIndex8 + offset];
        }
      }
}

// projects a field to a coarser grid ufine -> ucoarse
// temperature: medium, called (number of mg levels [1-5]) x (number of cg
// iterations [5-100]) = [5-500]  times pr design iteration
void projectToCoarserGrid_halo(const struct gpuNode * gpu_node,//in
                                const int l,//in
                               const CTYPE *ufine, //in
                               CTYPE *ucoarse //out
                               ) {
  const struct gridContext * gc = gpu_node->gc;

  // Computing dimensions for fine grid
  int ncellf,wrapxf,wrapyf,wrapzf;
  uint_fast32_t ndoff;
  computePadding(gc,l,&ncellf,&wrapxf,&wrapyf,&wrapzf,&ndoff);

  // Computing dimensions for coarse grid
  int ncellc,wrapxc,wrapyc,wrapzc;
  uint_fast32_t ndofc;
  computePadding(gc,l+1,&ncellc,&wrapxc,&wrapyc,&wrapzc,&ndofc);

  const int nelxc = gc->nelx / ncellc;
  const int nelyc = gc->nely / ncellc;
  const int nelzc = gc->nelz / ncellc;

  const int nxc = nelxc + 1;
  const int nyc = nelyc + 1;
  const int nzc = nelzc + 1;

  const MTYPE vals[4] = {1.0, 0.5, 0.25, 0.125};
  //const int n_threads = 8;
  //const int n_teams = 5120/n_threads;

  // loop over nodes, usually very large with nx*ny*nz = 100.000 or more
  //#pragma omp target data map(to:ufine[:ndofc_fine],vals[:4]) map(tofrom:ucoarse[:ndofc_coarse])
  #pragma omp target data map(to:vals[:4]) device(gpu_node->id)
  {
  //#pragma omp target teams num_teams(n_teams) distribute collapse(3)
  #pragma omp target teams distribute parallel for schedule(static) collapse(3) device(gpu_node->id)
  for (int32_t icoarse = 1; icoarse < nxc + 1; icoarse++)
    for (int32_t kcoarse = 1; kcoarse < nzc + 1; kcoarse++)
      for (int32_t jcoarse = 1; jcoarse < nyc + 1; jcoarse++) {

        const int coarseIndex =
            icoarse * wrapyc * wrapzc + kcoarse * wrapyc + jcoarse;

        ucoarse[3 * coarseIndex] = 0;
        ucoarse[3 * coarseIndex+1] = 0;
        ucoarse[3 * coarseIndex+2] = 0;
      }
  #pragma omp target teams distribute parallel for schedule(static) collapse(3) device(gpu_node->id)
  for (int32_t icoarse = 1; icoarse < nxc + 1; icoarse++)
    for (int32_t kcoarse = 1; kcoarse < nzc + 1; kcoarse++)
      for (int32_t jcoarse = 1; jcoarse < nyc + 1; jcoarse++) {

        const int coarseIndex =
            icoarse * wrapyc * wrapzc + kcoarse * wrapyc + jcoarse;


        // Node indices on fine grid
        const int nx1 = (icoarse - 1) * 2 + 1;
        const int ny1 = (jcoarse - 1) * 2 + 1;
        const int nz1 = (kcoarse - 1) * 2 + 1;

        const int xmin = nx1 - 1;
        const int ymin = ny1 - 1;
        const int zmin = nz1 - 1;

        const int xmax = nx1 + 2;
        const int ymax = ny1 + 2;
        const int zmax = nz1 + 2;

        /*ucoarse[3 * coarseIndex] = 0;
        ucoarse[3 * coarseIndex+1] = 0;
        ucoarse[3 * coarseIndex+2] = 0;*/

        // this can be done faster by writing out the 27 iterations by hand,
        // do it when necessary.
        //#pragma omp parallel for num_threads(n_threads) collapse(4) schedule(static)
        for (int32_t ifine = xmin; ifine < xmax; ifine++)
          for (int32_t kfine = zmin; kfine < zmax; kfine++)
            for (int32_t jfine = ymin; jfine < ymax; jfine++) 
              for (int32_t offset = 0; offset < 3; offset++){

              const uint32_t fineIndex =
                  ifine * wrapyf * wrapzf + kfine * wrapyf + jfine;

              const int ind = (nx1 - ifine) * (nx1 - ifine) +
                              (ny1 - jfine) * (ny1 - jfine) +
                              (nz1 - kfine) * (nz1 - kfine);
              int idx = 3 * coarseIndex+offset;
              CTYPE prod = vals[ind] * ufine[3 * fineIndex + offset];
              #pragma omp atomic update
              ucoarse[idx] += prod;
            }
      }
  }
}

// generate the matrix diagonal for jacobi smoothing.
// temperature: low-medium, called number of levels for every design
// iteration.
void assembleInvertedMatrixDiagonalSubspace_halo(const struct gpuNode * gpu_node,
                                                 const DTYPE *x, const int l,
                                                 MTYPE *diag) {
  const struct gridContext * gc = gpu_node->gc;
  int ncell,wrapxc,wrapyc,wrapzc;
  uint_fast32_t ndofc;
  computePadding(gc,l,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndofc);
  const int32_t nelxc = gc->nelx / ncell;
  const int32_t nelyc = gc->nely / ncell;
  const int32_t nelzc = gc->nelz / ncell;
  const int wrapy = gc->wrapy;
  const int wrapz = gc->wrapz;

  uint_fast32_t* idx = gc->fixedDofs[l].idx;
  uint_fast32_t n = gc->fixedDofs[l].n;
  const double Emin = gc->Emin;
  const double E0_Emin = gc->E0-gc->Emin;
  const MTYPE * KE = gc->precomputedKE[l];

  //#pragma omp target data map(to:idx[:n])
  //{
  #pragma omp target teams distribute parallel for schedule(static) device(gpu_node->id)
  for (unsigned int i = 0; i < ndofc; i++)
    diag[i] = 0.0;
  
  #pragma omp target teams distribute parallel for collapse(9) schedule(static) device(gpu_node->id)
  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)
        for (int32_t it = 1; it < nelxc + 1; it += 2)
          for (int32_t kt = 1; kt < nelzc + 1; kt += 2)
            for (int32_t jt = 1; jt < nelyc + 1; jt += 2) 
              for (int ii = 0; ii < ncell; ii++)
                for (int kk = 0; kk < ncell; kk++)
                  for (int jj = 0; jj < ncell; jj++) {
                    int32_t i = bx + it;
                    int32_t k = bz + kt;
                    int32_t j = by + jt;
                    if ((i >= nelxc + 1) || (k >= nelzc + 1) || (j >= nelyc + 1)){
                      break;
                    }
                    uint_fast32_t edof[24];
                    getEdof_halo(edof, i, j, k, wrapyc, wrapzc);
                    const int ifine = ((i - 1) * ncell) + ii + 1;
                    const int jfine = ((j - 1) * ncell) + jj + 1;
                    const int kfine = ((k - 1) * ncell) + kk + 1;

                    const int cellidx = ncell * ncell * ii + ncell * kk + jj;

                    const uint_fast32_t elementIndex =
                        ifine * (wrapy - 1) * (wrapz - 1) +
                        kfine * (wrapy - 1) + jfine;
                    const MTYPE elementScale =
                        Emin + x[elementIndex] * x[elementIndex] *
                                      x[elementIndex] * E0_Emin;

                    for (int iii = 0; iii < 24; iii++){
                      #pragma omp atomic update
                      diag[edof[iii]] += elementScale * KE[24 * 24 * cellidx + iii * 24 + iii];
                    }
                  }
//printf("(wrapxc,wrapyc,wrapzc)=(%d,%d,%d)\n",wrapxc,wrapyc,wrapzc);

// apply boundaryConditions
#pragma omp target teams distribute parallel for schedule(static) device(gpu_node->id)
  for (int i = 0; i < n; i++){
    diag[idx[i]] = 1.0;
  }

#pragma omp target teams distribute parallel for collapse(3) schedule(static) device(gpu_node->id)
  for (int i = 1; i < nelxc + 2; i++)
    for (int k = 1; k < nelzc + 2; k++)
      for (int j = 1; j < nelyc + 2; j++) {
        const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

        diag[3 * nidx + 0] = 1.0 / diag[3 * nidx + 0];
        diag[3 * nidx + 1] = 1.0 / diag[3 * nidx + 1];
        diag[3 * nidx + 2] = 1.0 / diag[3 * nidx + 2];
      }
  //}
}

__force_inline inline void getEdof_halo_8(uint_fast32_t edof[8], const int i,
                                        const int j, const int k,
                                        const int wrapy, const int wrapz) {

  const int nx_1 = i;
  const int nx_2 = i + 1;
  const int nz_1 = k;
  const int nz_2 = k + 1;
  const int ny_1 = j;
  const int ny_2 = j + 1;

  const uint_fast32_t nIndex1 = nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_2;
  const uint_fast32_t nIndex2 = nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_2;
  const uint_fast32_t nIndex3 = nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_1;
  const uint_fast32_t nIndex4 = nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_1;
  const uint_fast32_t nIndex5 = nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_2;
  const uint_fast32_t nIndex6 = nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_2;
  const uint_fast32_t nIndex7 = nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_1;
  const uint_fast32_t nIndex8 = nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_1;

  edof[0] = 3 * nIndex1;
  edof[1] = 3 * nIndex2;
  edof[2] = 3 * nIndex3;
  edof[3] = 3 * nIndex4;

  edof[4] = 3 * nIndex5;
  edof[5] = 3 * nIndex6;
  edof[6] = 3 * nIndex7;
  edof[7] = 3 * nIndex8;
}

// generate elementwise gradients from displacement.
// temperature: cold, called once for every design iteration.
void getComplianceAndSensetivity_halo(const struct gpuNode * gpu_node,
                                      const DTYPE *x, STYPE *u, DTYPE *c,
                                      DTYPE *dcdx) {
  const struct gridContext gc = *(gpu_node->gc);

  c[0] = 0.0;
  DTYPE cc = 0.0;

  #pragma omp target teams distribute parallel for schedule(static) collapse(3) device(gpu_node->id)
  for (int32_t i = 1; i < gc.nelx + 1; i++)
    for (int32_t k = 1; k < gc.nelz + 1; k++)
      for (int32_t j = 1; j < gc.nely + 1; j++) {
        const uint_fast32_t elementIndex =
              i * (gc.wrapy - 1) * (gc.wrapz - 1) + k * (gc.wrapy - 1) + j;
        dcdx[elementIndex] = 0.0;
      }

// loops over all elements, typically 100.000 or more. Note that there are no
// write dependencies, other than the reduction.
  MTYPE * KE = gc.precomputedKE[0];
  double E_diff = gc.E0 - gc.Emin;
  double Emin = gc.Emin;
  //#pragma omp target data map(always,tofrom:cc)
  {
  for (int ii=0;ii<8;ii++)
  #pragma omp target teams distribute parallel for schedule(static) collapse(3) map(always,tofrom:cc) reduction(+ : cc) firstprivate(Emin,E_diff) device(gpu_node->id)
  for (int32_t i = 1; i < gc.nelx + 1; i++)
    for (int32_t k = 1; k < gc.nelz + 1; k++)
      for (int32_t j = 1; j < gc.nely + 1; j++) {
          uint_fast32_t edof[8];

          getEdof_halo_8(edof, i, j, k, gc.wrapy, gc.wrapz);
          const uint_fast32_t elementIndex =
              i * (gc.wrapy - 1) * (gc.wrapz - 1) + k * (gc.wrapy - 1) + j;

          // clocal = ulocal^T * ke * ulocal
          STYPE clocal = 0.0;
          for(int iii=0;iii<3;iii++){
            MTYPE tmp = 0.0;
            for (int jj=0;jj<8;jj++){
              tmp += KE[(ii*3+iii)*24+3*jj]*u[edof[jj]];
              tmp += KE[(ii*3+iii)*24+3*jj+1]*u[edof[jj]+1];
              tmp += KE[(ii*3+iii)*24+3*jj+2]*u[edof[jj]+2];
            }
            clocal += u[edof[ii]+iii] * tmp;
          }
          // apply contribution to c and dcdx
          cc += clocal * (Emin + x[elementIndex] * x[elementIndex] *
                                        x[elementIndex] * E_diff);
          dcdx[elementIndex] += clocal * (-3.0 * E_diff *
                                        x[elementIndex] * x[elementIndex]);
          }
  }
  c[0] = cc;
}
