#include "stencil_grid_utility.h"

#include "local_matrix.h"

#include "gpu_definitions.h"

void setFixedDof_halo(struct gridContext *gc, const int l) {

  // Computing dimensions for grid
  int ncell,wrapxc,wrapyc,wrapzc;
  uint_fast32_t ndofc;
  computePadding(gc,l,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndofc);
  const int32_t nelyc = (*gc).nely / ncell;
  const int32_t nelzc = (*gc).nelz / ncell;

  const int nyc = (nelyc + 1);
  const int nzc = (nelzc + 1);

  // classic cantilever
  // (*gc).fixedDofs[l].n = 3 * nyc * nzc;
  // (*gc).fixedDofs[l].idx = malloc(sizeof(uint_fast32_t) *
  // (*gc).fixedDofs[l].n); int offset = 0; for (uint_fast32_t k = 1; k < (nzc +
  // 1); k++)
  //   for (uint_fast32_t j = 1; j < (nyc + 1); j++) {
  //     (*gc).fixedDofs[l].idx[offset + 0] =
  //         3 * (wrapyc * wrapzc + wrapyc * k + j) + 0;
  //     (*gc).fixedDofs[l].idx[offset + 1] =
  //         3 * (wrapyc * wrapzc + wrapyc * k + j) + 1;
  //     (*gc).fixedDofs[l].idx[offset + 2] =
  //         3 * (wrapyc * wrapzc + wrapyc * k + j) + 2;
  //     offset += 3;
  //   }

  // new cantilever
  const int nodelimit = (nelyc / 4) + 1;
  (*gc).fixedDofs[l].n = 3 * nzc * 2 * nodelimit;
  (*gc).fixedDofs[l].idx = malloc(sizeof(uint_fast32_t) * (*gc).fixedDofs[l].n);
  int offset = 0;
  const int i = 1;
  for (uint_fast32_t k = 1; k < (nzc + 1); k++) {
    for (uint_fast32_t j = 1; j < nodelimit + 1; j++) {
      (*gc).fixedDofs[l].idx[offset + 0] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 0;
      (*gc).fixedDofs[l].idx[offset + 1] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 1;
      (*gc).fixedDofs[l].idx[offset + 2] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 2;
      offset += 3;
    }
    for (uint_fast32_t j = (nyc + 1) - nodelimit; j < (nyc + 1); j++) {
      (*gc).fixedDofs[l].idx[offset + 0] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 0;
      (*gc).fixedDofs[l].idx[offset + 1] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 1;
      (*gc).fixedDofs[l].idx[offset + 2] =
          3 * (i * wrapyc * wrapzc + wrapyc * k + j) + 2;
      offset += 3;
    }
  }

  const uint_fast32_t n = (*gc).fixedDofs[l].n;
  const uint_fast32_t *pidx = (*gc).fixedDofs[l].idx;

  #pragma omp target enter data map(to : pidx[:n])
}

void setupGC(struct gridContext *gc, const int nl, const int nelx, const int nely, const int nelz) {

  (*gc).E0 = 1;
  (*gc).Emin = 1e-6;
  (*gc).nu = 0.3;
  (*gc).penal = 3; // dummy variable, does nothing

  (*gc).elementSizeX = 0.5;
  (*gc).elementSizeY = 0.5;
  (*gc).elementSizeZ = 0.5;

  (*gc).nelx = nelx;
  (*gc).nely = nely;
  (*gc).nelz = nelz;

  const int paddingx =
      (STENCIL_SIZE_X - (((*gc).nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - (((*gc).nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - (((*gc).nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  (*gc).wrapx = (*gc).nelx + paddingx + 3;
  (*gc).wrapy = (*gc).nely + paddingy + 3;
  (*gc).wrapz = (*gc).nelz + paddingz + 3;

  (*gc).precomputedKE = malloc(sizeof(MTYPE *) * nl);
  (*gc).fixedDofs = malloc(sizeof(struct FixedDofs) * nl);

  for (int l = 0; l < nl; l++) {
    const int ncell = pow(2, l);
    const int pKESize = 24 * 24 * ncell * ncell * ncell;
    (*gc).precomputedKE[l] = malloc(sizeof(MTYPE) * pKESize);
    getKEsubspace((*gc).precomputedKE[l], (*gc).nu, l);

    setFixedDof_halo(gc, l);

    //MTYPE *pKE = (*gc).precomputedKE[l];

//#pragma omp target enter data map(to : pKE[:pKESize])
  }
}

void freeGC(struct gridContext *gc, const int nl) {

  for (int l = 0; l < nl; l++) {
    free((*gc).precomputedKE[l]);
    free((*gc).fixedDofs[l].idx);
  }

  free((*gc).precomputedKE);
  free((*gc).fixedDofs);
}

void allocateZeroPaddedStateField(const struct gridContext gc, const int l,
                                  CTYPE **v) {

  // Computing dimensions for coarse grid
  int ncell,wrapx,wrapy,wrapz;
  uint_fast32_t ndof;
  computePadding(&gc,l,&ncell,&wrapx,&wrapy,&wrapz,&ndof);

  (*v) = malloc(sizeof(CTYPE) * ndof);

#pragma omp parallel for schedule(static) default(none) firstprivate(ndof) shared(v)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}

void allocateZeroPaddedStateField_MTYPE(const struct gridContext gc,
                                        const int l, MTYPE **v) {

  // Computing dimensions for coarse grid
  int ncell,wrapx,wrapy,wrapz;
  uint_fast32_t ndof;
  computePadding(&gc,l,&ncell,&wrapx,&wrapy,&wrapz,&ndof);

  (*v) = malloc(sizeof(MTYPE) * ndof);

#pragma omp parallel for schedule(static) default(none) firstprivate(ndof) shared(v)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}

void allocateZeroPaddedStateField_STYPE(const struct gridContext gc,
                                        const int l, STYPE **v) {

  // Computing dimensions for coarse grid
  int ncell,wrapx,wrapy,wrapz;
  uint_fast32_t ndof;
  computePadding(&gc,l,&ncell,&wrapx,&wrapy,&wrapz,&ndof);

  (*v) = malloc(sizeof(STYPE) * ndof);

#pragma omp parallel for schedule(static) default(none) firstprivate(ndof) shared(v)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}
