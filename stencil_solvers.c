#include "stencil_solvers.h"

#include "stencil_grid_utility.h"
#include "stencil_methods.h"

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.

void smoothDampedJacobiSubspace_halo(const struct gpuNode * gpu_node,
                                     const DTYPE *x, const int l,
                                     const uint_fast32_t nswp,
                                     const CTYPE omega, const MTYPE *invD,
                                     CTYPE *u, const CTYPE *b, CTYPE *tmp) {

  const struct gridContext * gc = gpu_node->gc;

  int ncell,wrapxc,wrapyc,wrapzc;
  uint_fast32_t ndof;
  computePadding(gc,l,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndof);

  const int32_t nelxc = gc->nelx / ncell;
  const int32_t nelyc = gc->nely / ncell;
  const int32_t nelzc = gc->nelz / ncell;

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperatorSubspace_halo(gpu_node, l, x, u, tmp);
  

    // long for loop, as ndof is typically 300.000 or more, but also trivially
    // parallel.
    #pragma omp target teams distribute parallel for collapse(3) schedule(static) device(gpu_node->id)
    for (int i = 1; i < nelxc + 2; i++)
      for (int k = 1; k < nelzc + 2; k++)
        for (int j = 1; j < nelyc + 2; j++) {
          const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          u[idx1] += omega * invD[idx1] * (b[idx1] - tmp[idx1]);
          u[idx2] += omega * invD[idx2] * (b[idx2] - tmp[idx2]);
          u[idx3] += omega * invD[idx3] * (b[idx3] - tmp[idx3]);
        }
  }
}

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.
void smoothDampedJacobiSubspaceMatrix_halo(
    const struct gridContext * gc, const struct CSRMatrix M, const int l,
    const uint_fast32_t nswp, const CTYPE omega, const MTYPE *invD, CTYPE *u,
    const CTYPE *b, CTYPE *tmp) {

    int ncell,wrapxc,wrapyc,wrapzc;
    uint_fast32_t ndof;
    computePadding(gc,l,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndof);

    const int32_t nelxc = gc->nelx / ncell;
    const int32_t nelyc = gc->nely / ncell;
    const int32_t nelzc = gc->nelz / ncell;

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperatorSubspaceMatrix(*gc, l, M, u, tmp);

// long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
    #pragma omp parallel for collapse(3) schedule(static) default(none) \
      firstprivate(nelxc,nelzc,nelyc,omega,wrapyc,wrapzc) shared(u,invD,b,tmp)    
    for (int i = 1; i < nelxc + 2; i++)
      for (int k = 1; k < nelzc + 2; k++)
        for (int j = 1; j < nelyc + 2; j++) {
          const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          u[idx1] += omega * invD[idx1] * (b[idx1] - tmp[idx1]);
          u[idx2] += omega * invD[idx2] * (b[idx2] - tmp[idx2]);
          u[idx3] += omega * invD[idx3] * (b[idx3] - tmp[idx3]);
        }
  }
}

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.

void smoothDampedJacobi_halo(const struct gpuNode * gpu_node, const DTYPE *x,
                             const uint_fast32_t nswp, const CTYPE omega,
                             const MTYPE *invD, CTYPE *u, const CTYPE *b,
                             CTYPE *tmp) {
  const struct gridContext gc = *(gpu_node->gc);
  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperator_stencil(gpu_node, x, u, tmp);

  // it is not faster to make an even simpler kernel with four loops
#pragma omp target teams distribute parallel for collapse(3) schedule(static) device(gpu_node->id)
    for (int i = 1; i < gc.nelx + 2; i++)
      for (int k = 1; k < gc.nelz + 2; k++)
        for (int j = 1; j < gc.nely + 2; j++) {
          const int nidx = i * gc.wrapy * gc.wrapz + gc.wrapy * k + j;

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          u[idx1] += omega * invD[idx1] * (b[idx1] - tmp[idx1]);
          u[idx2] += omega * invD[idx2] * (b[idx2] - tmp[idx2]);
          u[idx3] += omega * invD[idx3] * (b[idx3] - tmp[idx3]);
        }
  }
}


// Vcycle preconditioner. recursive function.
// temperature: medium, called (number of levels)x(number of cg iterations ~
// 5 - 100) every design iteration. Much of the compute time is spent in
// this function, although in children functions.
void VcyclePreconditioner(const struct gpuGrid * gpu_gc, const DTYPE *x,
                          const int nl, const int l, MTYPE **const invD,
                          struct CoarseSolverData *bottomSolverData,
                          const struct CSRMatrix *coarseMatrices, CTYPE omega,
                          const int nswp, CTYPE **r, CTYPE **z, CTYPE **d) {
  const struct gridContext * gc = gpu_gc->target[0].gc;

  int ncell,wrapxc,wrapyc,wrapzc;
  uint_fast32_t ndofc;
  computePadding(gc,l,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndofc);

  int ncell_nl,wrapx_nl,wrapy_nl,wrapz_nl;
  uint_fast32_t ndof_nl;
  computePadding(gc,l+1,&ncell_nl,&wrapx_nl,&wrapy_nl,&wrapz_nl,&ndof_nl);

  CTYPE *zptr = z[l];
  CTYPE *dptr = d[l];
  CTYPE *rptr = r[l];
  CTYPE * next_rptr = r[l + 1];
  CTYPE * next_zptr = z[l + 1];
  MTYPE *invDptr = invD[l];

    #pragma omp target teams distribute parallel for schedule(static)
    for (int i = 0; i < ndofc; i++)
        zptr[i] = 0.0;
    // smooth
    if (l == 0) {
      //printf("Case 1 - enter\n");

      smoothDampedJacobi_halo(&(gpu_gc->target[0]), x, nswp, omega, invDptr, zptr, rptr, dptr);

      applyStateOperator_stencil(&(gpu_gc->target[0]), x, zptr, dptr);
      //#pragma omp target update from(dptr[:ndofc])

    } else if (l < number_of_matrix_free_levels) {
      //printf("Case 2 -enter \n");
      //#pragma omp target update to(invDptr[:ndofc],rptr[:ndofc],zptr[:ndofc])
      smoothDampedJacobiSubspace_halo(&(gpu_gc->target[0]), x, l, nswp, omega, invDptr, zptr, rptr,
                                      dptr);
      
      applyStateOperatorSubspace_halo(&(gpu_gc->target[0]), l, x, zptr, dptr);
      
      //#pragma omp target update from(zptr[:ndofc])
    } else {
      //printf("Case 3 -enter \n");
      #pragma omp target update from(zptr[:ndofc],rptr[:ndofc],dptr[:ndofc])
      smoothDampedJacobiSubspaceMatrix_halo(gc, coarseMatrices[l], l, nswp, omega,
                                            invDptr, zptr, rptr, dptr); // Updates z[l] and d[l]
      applyStateOperatorSubspaceMatrix(*gc, l, coarseMatrices[l], zptr, dptr); // Updates coarseMatrices[l] and d[l]
      #pragma omp target update to(zptr[:ndofc],dptr[:ndofc])
    }
  
                          
  // long for loop, as ndof is typically 300.000 or more, but also trivially
  // parallel
    #pragma omp target teams distribute parallel for schedule(static)
    for (int i = 0; i < ndofc; i++)
      dptr[i] = rptr[i] - dptr[i];
    
    //#pragma omp target update from(dptr[:ndofc])
    

    // project residual down
    projectToCoarserGrid_halo(&(gpu_gc->target[0]), l, dptr, next_rptr);
    //printf("To coarser grid\n");

    // smooth coarse
    if (nl == l + 2) {
      #pragma omp parallel for schedule(static) default(none) firstprivate(ndof_nl,l) shared(next_zptr)
      for (int i = 0; i < ndof_nl; i++)
        next_zptr[i] = 0.0;

      // Must run on CPU due to Cholesky factorization
      #pragma omp target update from(next_rptr[:ndof_nl])
      solveSubspaceMatrix(*gc, l + 1, *bottomSolverData, next_rptr, next_zptr);
      #pragma omp target update to(next_zptr[:ndof_nl])
      //printf("Solved on CPU\n");

    } else
      VcyclePreconditioner(gpu_gc, x, nl, l + 1, invD, bottomSolverData,
                          coarseMatrices, omega, nswp, r, z, d);

    // project residual up
    projectToFinerGrid_halo(&(gpu_gc->target[0]), l, next_zptr, dptr);
    //printf("To finer grid\n");

    // smooth
    if (l == 0) {
      //printf("Case 1 exit\n");
      //#pragma omp target update to(dptr[:ndofc])

      #pragma omp target teams distribute parallel for schedule(static)
      for (int i = 0; i < ndofc; i++)
        zptr[i] += dptr[i];

      smoothDampedJacobi_halo(&(gpu_gc->target[0]), x, nswp, omega, invDptr, zptr, rptr, dptr);

      //#pragma omp target update from(zptr[:ndofc])
    }
    else if (l < number_of_matrix_free_levels) {
      //printf("Case 2 exit\n");

      #pragma omp target teams distribute parallel for schedule(static)
      for (int i = 0; i < ndofc; i++)
        zptr[i] += dptr[i];
      
      //#pragma omp target update to(invDptr[:ndofc])
      smoothDampedJacobiSubspace_halo(&(gpu_gc->target[0]), x, l, nswp, omega, invDptr, zptr, rptr,
                                      dptr);
      //#pragma omp target update from(zptr[:ndofc])
    } else {
      //printf("Case 3 - exit\n");

      #pragma omp target teams distribute parallel for schedule(static)
      for (int i = 0; i < ndofc; i++)
        zptr[i] += dptr[i];

      #pragma omp target update from(zptr[:ndofc],rptr[:ndofc],dptr[:ndofc])

      smoothDampedJacobiSubspaceMatrix_halo(gc, coarseMatrices[l], l, nswp, omega,
                                            invD[l], zptr, rptr, dptr);
      #pragma omp target update to(dptr[:ndofc],zptr[:ndofc])
    }
}

// solves the linear system of Ku = b.
// temperature: medium, accounts for 95% or more of runtime, but this time is
// spent in children functions. The iter loop of this funciton is a good
// candidate for GPU parallel region scope, as it is only performed once every
// design iteration (and thus only 100 times during a program)
void solveStateMG_halo(const struct gpuGrid * gpu_gc, DTYPE *x, const int nswp,
                       const int nl, const CTYPE tol,
                       struct SolverDataBuffer *data, int *finalIter,
                       float *finalRes, CTYPE *b, STYPE *u) {

  const struct gridContext * gc = gpu_gc->target[0].gc;

  const uint_fast32_t ndof = 3 * gc->wrapx * gc->wrapy * gc->wrapz;

  CTYPE *r = data->r;
  CTYPE *p = data->p;
  CTYPE *q = data->q;
  CTYPE *z = data->z;

  MTYPE **invD = data->invD;
  CTYPE **dmg = data->dmg;
  CTYPE **rmg = data->rmg;
  CTYPE **zmg = data->zmg;

  // setup residual vector
  #pragma omp target teams distribute parallel for schedule(static) nowait
  for (uint_fast32_t i = 0; i < ndof; i++)
    z[i] = (CTYPE)u[i];

  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    // printf("assemble mat l:%i\n", l);
    assembleSubspaceMatrix(*gc, l, x, data->coarseMatrices[l], invD[l]);
    MTYPE *invDptr = invD[l];
    int ncell,wrapxc,wrapyc,wrapzc;
    uint_fast32_t ndofc;
    computePadding(gc,l,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndofc);
    #pragma omp target update to(invDptr[:ndofc])
  }

  for (int l = 0; l < nl; l++) {
    assembleInvertedMatrixDiagonalSubspace_halo(&(gpu_gc->target[0]), x, l, invD[l]);
    int ncell,wrapxc,wrapyc,wrapzc;
    uint_fast32_t ndofc;
    computePadding(gc,l,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndofc);
    #pragma omp target update from(invD[l][:ndofc]) nowait
  }

  factorizeSubspaceMatrix(*gc, nl - 1, data->bottomSolver,
                          data->coarseMatrices[nl - 1]);

  //const uint32_t designSize = (gc.wrapx - 1) * (gc.wrapy - 1) * (gc.wrapz - 1);
  CTYPE rhoold = 0.0;
  CTYPE dpr;
  CTYPE alpha;
  CTYPE rho;
  //CTYPE *dptr = dmg[0];
  //MTYPE *invDptr = invD[0];

//#pragma omp target data map(to:b                                \
                            [:ndof])//, z                                         \
                            [:ndof], r                                         \
                            [:ndof], p                                         \
                            [:ndof], q                                         \
                            [:ndof], dptr                                      \
                            [:ndof], invDptr                                   \
                            [:ndof]) map(tofrom                                \
                                         : u[:ndof],x[:designSize])
  {

    #pragma omp taskwait

    applyStateOperator_stencil(&(gpu_gc->target[0]), x, z, r);

    #pragma omp target teams distribute parallel for schedule(static)
    for (uint_fast32_t i = 0; i < ndof; i++)
      r[i] = b[i] - r[i];

    // setup scalars
    const MTYPE omega = 0.6;
    const CTYPE bnorm = norm(b, ndof);
    const int maxIter = 1000;

    // begin cg loop - usually spans 5 - 300 iterations will be reduced to 5 -
    // 20 iterations once direct solver is included for coarse subproblem.
    for (int iter = 0; iter < maxIter; iter++) {

      //#pragma omp target update from(r[:ndof])

      // get preconditioned vector
      VcyclePreconditioner(gpu_gc, x, nl, 0, invD, &data->bottomSolver,
                           data->coarseMatrices, omega, nswp, rmg, zmg, dmg);
      //printf("Exit VcyclePreconditioner\n");
      //#pragma omp target update to(z[:ndof])

      rho = innerProduct(r, z, ndof);

      if (iter == 0) {

        #pragma omp target teams distribute parallel for schedule(static)
        for (uint_fast32_t i = 0; i < ndof; i++)
          p[i] = z[i];

      } else {

        CTYPE beta = rho / rhoold;
        #pragma omp target teams distribute parallel for firstprivate(beta) schedule(static)
        for (uint_fast32_t i = 0; i < ndof; i++)
          p[i] = beta * p[i] + z[i];
      }

      applyStateOperator_stencil(&(gpu_gc->target[0]), x, p, q);

      dpr = innerProduct(p, q, ndof);

      alpha = rho / dpr;
      rhoold = rho;

      #pragma omp target teams distribute parallel for firstprivate(alpha) schedule(static) nowait
      for (uint_fast32_t i = 0; i < ndof; i++)
        u[i] += (STYPE)(alpha * p[i]);

      #pragma omp target teams distribute parallel for firstprivate(alpha) schedule(static)
      for (uint_fast32_t i = 0; i < ndof; i++)
        r[i] -= alpha * q[i];


      CTYPE rnorm = 0.0;

      rnorm = norm(r, ndof);

      const CTYPE relres = rnorm / bnorm;

      (*finalIter) = iter;
      (*finalRes) = relres;

      #pragma omp taskwait

      //printf("it: %i, res=%e\n", iter, relres);

      if (relres < tol)
        break;
    }
  }
}

void allocateSolverData(const struct gridContext gc, const int nl,
                        struct SolverDataBuffer *data) {

  allocateZeroPaddedStateField(gc, 0, &(*data).r);
  allocateZeroPaddedStateField(gc, 0, &(*data).p);
  allocateZeroPaddedStateField(gc, 0, &(*data).q);
  allocateZeroPaddedStateField(gc, 0, &(*data).z);

  (*data).invD = malloc(sizeof(MTYPE *) * nl);
  (*data).dmg = malloc(sizeof(CTYPE *) * nl);
  (*data).rmg = malloc(sizeof(CTYPE *) * nl);
  (*data).zmg = malloc(sizeof(CTYPE *) * nl);

  allocateZeroPaddedStateField(gc, 0, &((*data).dmg[0]));
  allocateZeroPaddedStateField_MTYPE(gc, 0, &((*data).invD[0]));
  (*data).rmg[0] = (*data).r;
  (*data).zmg[0] = (*data).z;

  for (int l = 1; l < nl; l++) {
    allocateZeroPaddedStateField(gc, l, &((*data).dmg[l]));
    allocateZeroPaddedStateField(gc, l, &((*data).rmg[l]));
    allocateZeroPaddedStateField(gc, l, &((*data).zmg[l]));
    allocateZeroPaddedStateField_MTYPE(gc, l, &((*data).invD[l]));
  }

  // allocate for all levels for easy indces
  (*data).coarseMatrices = malloc(sizeof(struct CSRMatrix) * nl);
  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    allocateSubspaceMatrix(gc, l, &((*data).coarseMatrices[l]));
  }
}

void freeSolverData(struct SolverDataBuffer *data, const int nl) {

  free((*data).r);
  free((*data).z);
  free((*data).p);
  free((*data).q);

  free((*data).invD[0]);
  free((*data).dmg[0]);

  for (int l = 1; l < nl; l++) {
    free((*data).invD[l]);
    free((*data).dmg[l]);
    free((*data).rmg[l]);
    free((*data).zmg[l]);
  }

  free((*data).invD);
  free((*data).dmg);
  free((*data).zmg);
  free((*data).rmg);

  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    freeSubspaceMatrix(&((*data).coarseMatrices[l]));
  }
  free((*data).coarseMatrices);
}
