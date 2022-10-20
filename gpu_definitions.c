#include "gpu_definitions.h"

void gridContextToGPUGrid(
    struct gridContext * gc,
    struct gpuGrid * gpu_gc,
    const int nl,
    const int verbose)
{
    const int num_targets = gpu_gc->num_targets;
    gpu_gc->target = (struct gpuNode *) malloc(num_targets*sizeof(struct gpuNode));

    // Finding size of x per target
    const int nelx = gc->nelx;
    const int block_x = (int) nelx/num_targets;

    for (int i = 0;i<num_targets;i++){
        // Initializing node
        (gpu_gc->target[i]).gc = (struct gridContext *) malloc(sizeof(struct gridContext));
        struct gpuNode * node = &(gpu_gc->target[i]);
        node -> id = i;
        if (i>0){
            node -> prev = &(gpu_gc->target[i-1]);
            node -> offset_x = 1;
        }
        else {
            node -> prev = NULL;
            node -> offset_x = 0;
        }
        if (i<num_targets-1){
            node -> next = &(gpu_gc->target[i+1]);
        }
        else {
            node -> next = NULL;
        }

        // Global offset in the grid
        node -> x_global = i *block_x;

        // Calculating x size
        int nelx_local = ( i== num_targets-1) ? nelx - i*num_targets : block_x;
        if (node -> prev != NULL){
            nelx_local += 1;
        }
        if (node -> next != NULL){
            nelx_local += 1;
        }
        setupGC(node->gc,nl,nelx_local,gc->nely,gc->nelz);
        if (verbose == 1) {
            printf("GPU %d:\n\tGlobal index: %d\n\tOffset: %d\n",node->id,node->x_global,node->offset_x);
            printf("\tPrev is null? %d\n\tNext is null? %d\n",node->prev == NULL,node->next==NULL);
        }
    }
}

void freeGPUGrid(
    struct gpuGrid * gpu_gc,
    const int nl)
{
    const int num_targets = gpu_gc->num_targets;
    for (int i=0;i<num_targets;i++){
        struct gpuNode * node = &(gpu_gc->target[i]);
        freeGC(node->gc,nl);
    }
    free(gpu_gc->target);
}

void computePadding(
    const struct gridContext * gc,
    const int l,
    int * ncell,
    int * wrapx,
    int * wrapy,
    int * wrapz,
    uint_fast32_t * ndof
){
    *ncell = pow(2, l);
    const int32_t nelx = gc->nelx / (*ncell);
    const int32_t nely = gc->nely / (*ncell);
    const int32_t nelz = gc->nelz / (*ncell);
    const int paddingx =
        (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
    const int paddingy =
        (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
    const int paddingz =
        (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

    // Writing the result
    *wrapx = nelx + paddingx + 3;
    *wrapy = nely + paddingy + 3;
    *wrapz = nelz + paddingz + 3;
    *ndof = 3 * (*wrapy) * (*wrapz) * (*wrapx);
}

DTYPE compute_volume(
    const struct gridContext gc,
    DTYPE * xPhys
){
    DTYPE vol = 0.0;
    #pragma omp target teams distribute parallel for collapse(3) map(always,tofrom:vol) reduction(+ : vol)
    for (int i = 1; i < gc.nelx + 1; i++)
      for (int k = 1; k < gc.nelz + 1; k++)
        for (int j = 1; j < gc.nely + 1; j++) {
          const int idx =
              i * (gc.wrapy - 1) * (gc.wrapz - 1) +
              k * (gc.wrapy - 1) + j;

          vol += xPhys[idx];
        }
    return vol;
}

float compute_change(
    DTYPE * x,
    DTYPE * xnew,
    const int nelem
){
    float change = 0.0;

    #pragma omp target teams distribute parallel for schedule(static) map(always,tofrom:change) reduction(max : change)
    for (uint_least32_t i = 0; i < nelem; i++) {
      change = MAX(change, fabs(x[i] - xnew[i]));
      x[i] = xnew[i];
    }
    return change;
}

void update_solution(
    const DTYPE * x,
    DTYPE * xnew,
    const DTYPE * dv, 
    const DTYPE * dc, 
    const int nelem,
    const DTYPE g
){
    DTYPE l1 = 0.0, l2 = 1e9, move = 0.2;

    while ((l2 - l1) / (l1 + l2) > 1e-6) {
      DTYPE lmid = 0.5 * (l2 + l1);
      DTYPE gt = 0.0;
      

      #pragma omp target teams distribute parallel for schedule(static) map(always,tofrom:gt) firstprivate(move,lmid) reduction(+:gt)
      for (uint_least32_t i = 0; i < nelem; i++) {
        xnew[i] =
              MAX(0.0, MAX(x[i] - move,
                          MIN(1.0, MIN(x[i] + move,
                                        x[i] * sqrt(-dc[i] / (dv[i] * lmid))))));
        gt += dv[i] * (xnew[i] - x[i]);
      }

      gt += g;
      if (gt > 0)
        l1 = lmid;
      else
        l2 = lmid;
    }
}

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
    const int nl
){
    int u_size = 0;
    const int ndof = 3 * gridContext.wrapx * gridContext.wrapy * gridContext.wrapz;
    for (int l_tmp = 0;l_tmp < nl; l_tmp++){
        int ncell;
        int wrapxc;
        int wrapyc;
        int wrapzc;
        uint_fast32_t ndofc;
        computePadding(&gridContext,l_tmp,&ncell,&wrapxc,&wrapyc,&wrapzc,&ndofc);
        if (l_tmp == 0){
            u_size = ndofc;
        }
        CTYPE *zptr = solverData.zmg[l_tmp];
        CTYPE *dptr = solverData.dmg[l_tmp];
        CTYPE *rptr = solverData.rmg[l_tmp];
        MTYPE *invDptr = solverData.invD[l_tmp];
        const MTYPE * KE = gpu_gc.target[0].gc->precomputedKE[l_tmp];
        #pragma omp target enter data map(to:zptr[:ndofc],invDptr[:ndofc],rptr[:ndofc],\
                                    dptr[:ndofc],KE[:24*24*ncell*ncell*ncell]) device(gpu_gc.target[0].id) nowait
    }
    #pragma omp target enter data map(to:xPhys[:nelem],x[:nelem],dc[:nelem],xnew[:nelem],U[:u_size],F[:ndof],\
    solverData.r[:ndof],solverData.p[:ndof],solverData.q[:ndof],solverData.z[:ndof]) device(gpu_gc.target[0].id) nowait
    #pragma omp taskwait
}
