#include "stencil_optimization.h"

#include "stencil_grid_utility.h"
#include "stencil_methods.h"
#include "stencil_solvers.h"
#include "gpu_definitions.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// main function
void top3dmgcg(const uint_fast32_t nelx, const uint_fast32_t nely,
               const uint_fast32_t nelz, const DTYPE volfrac, const DTYPE rmin,
               const uint_fast32_t nl, const float cgtol,
               const uint_fast32_t cgmax, const int verbose,const int write_result,const int max_iterations) {

  struct gridContext gridContext;
  setupGC(&gridContext, nl,nelx,nely,nelz);

  struct gpuGrid gpu_gc;
  const int num_targets = 1; //omp_get_num_devices();
  if (verbose == 1){
    printf("OpenMP enabled with %d devices.\n",num_targets);
    printf("OpenMP default device: %d\n",omp_get_default_device());
  }
  gpu_gc.num_targets = num_targets;

  gridContextToGPUGrid(&gridContext,&gpu_gc,nl,verbose);

  const uint_fast64_t nelem = (gridContext.wrapx - 1) *
                              (gridContext.wrapy - 1) * (gridContext.wrapz - 1);

  CTYPE *F;
  STYPE *U;
  allocateZeroPaddedStateField(gridContext, 0, &F);
  allocateZeroPaddedStateField_STYPE(gridContext, 0, &U);

  double forceMagnitude = -1;

  { // setup cantilever load
    const int ny = nely + 1;

    const int k = 0;

    const double radius = ((double)ny) / 5.0; // snap
    const double radius2 = radius * radius;
    const double center_x = (double)nelx;
    const double center_y = ((double)nely - 1.0) / 2.0;

    int num_elements = 0;
    for (int i = 0; i < nelx; i++) {
      for (int j = 0; j < nely; j++) {
        const double dx = (double)i - center_x;
        const double dy = (double)j - center_y;
        const double dist2 = dx * dx + dy * dy;
        if (dist2 < radius2) {
          num_elements++;
        }
      }
    }

    double nodalForce = forceMagnitude / (4.0 * (double)num_elements);
    for (int i = 0; i < nelx; i++) {
      for (int j = 0; j < nely; j++) {

        const int ii = i + 1;
        const int jj = j + 1;
        const int kk = k + 1;

        const double dx = (double)i - center_x;
        const double dy = (double)j - center_y;
        const double dist2 = dx * dx + dy * dy;

        if (dist2 < radius2) {
          const uint_fast32_t nidx1 =
              (ii + 1) * gridContext.wrapy * gridContext.wrapz +
              gridContext.wrapy * kk + (jj + 1);
          const uint_fast32_t nidx2 =
              (ii + 1) * gridContext.wrapy * gridContext.wrapz +
              gridContext.wrapy * kk + jj;
          const uint_fast32_t nidx3 =
              ii * gridContext.wrapy * gridContext.wrapz +
              gridContext.wrapy * kk + (jj + 1);
          const uint_fast32_t nidx4 =
              ii * gridContext.wrapy * gridContext.wrapz +
              gridContext.wrapy * kk + jj;
          F[3 * nidx1 + 2] += nodalForce;
          F[3 * nidx2 + 2] += nodalForce;
          F[3 * nidx3 + 2] += nodalForce;
          F[3 * nidx4 + 2] += nodalForce;
        }
      }
    }
  }

  DTYPE *dc = malloc(sizeof(DTYPE) * nelem);
  DTYPE *dv = malloc(sizeof(DTYPE) * nelem);
  DTYPE *x = malloc(sizeof(DTYPE) * nelem);
  DTYPE *xPhys = malloc(sizeof(DTYPE) * nelem);
  DTYPE *xnew = malloc(sizeof(DTYPE) * nelem);
  DTYPE c = 0.0;

  #pragma omp parallel for schedule(static) default(none) firstprivate(nelem) shared(x,xPhys,dv)
  for (uint_fast64_t i = 0; i < nelem; i++) {
    x[i] = 0.0;
    xPhys[i] = 0.0;
    dv[i] = 1.0;
  }

  #pragma omp parallel for collapse(3) schedule(static) default(none) firstprivate(volfrac) shared(gridContext,x,xPhys)
  for (int i = 1; i < gridContext.nelx + 1; i++)
    for (int k = 1; k < gridContext.nelz + 1; k++)
      for (int j = 1; j < gridContext.nely + 1; j++) {
        const int idx = i * (gridContext.wrapy - 1) * (gridContext.wrapz - 1) +
                        k * (gridContext.wrapy - 1) + j;

        x[idx] = volfrac;
        xPhys[idx] = volfrac;
      }

  // allocate needed memory for solver
  struct SolverDataBuffer solverData;
  allocateSolverData(gridContext, nl, &solverData);
  initializeCholmod(gridContext, nl - 1, &solverData.bottomSolver,
                    solverData.coarseMatrices[nl - 1]);

  #pragma omp target enter data map(to:dv[:nelem]) device(gpu_gc.target[0].id)
  applyDensityFilterGradient(gridContext, rmin, dv);

  unsigned int loop = 0;
  float change = 1;

  #ifdef _OPENMP
    if (verbose ==1){ 
      printf(" OpenMP enabled with %d threads\n", omp_get_max_threads());
    }

    const double start_wtime = omp_get_wtime();
  #endif
  
  // Mapping data to the GPU grid
  enter_data(gridContext,solverData,gpu_gc,xPhys,x,xnew,U,F,dc,nelem,nl);

  /* %% START ITERATION */
  DTYPE vol = 0.0;
  int gciter_total = 0;
  while ((change > 1e-2) && (loop < max_iterations)) {

    loop++;

    int cgiter;
    float cgres;
    const int nswp = 4;
    solveStateMG_halo(&gpu_gc, xPhys, nswp, nl, cgtol, &solverData, &cgiter,
                      &cgres, F, U);
    //printf("Exit solveStateMG_halo\n");
    vol = compute_volume(gridContext,xPhys);

    getComplianceAndSensetivity_halo(&(gpu_gc.target[0]), xPhys, U, &c, dc);
    //printf("Exit getComplianceAndSensetivity_halo\n");
    applyDensityFilterGradient(gridContext, rmin, dc);

    //#pragma omp taskwait
    vol /= (DTYPE)(gridContext.nelx * gridContext.nely * gridContext.nelz);
    DTYPE g = vol - volfrac;

    // Iteratively stepping solution forward
    update_solution(x,xnew,dv,dc,nelem,g);
    
    // Computing the maximum change over all elements
    change = compute_change(x,xnew,nelem);

    #pragma omp target update from(x[:nelem])
    applyDensityFilter(gridContext, rmin, x, xPhys);
    #pragma omp target update to(xPhys[:nelem])

    gciter_total += cgiter;

    if (verbose ==1){ 
      printf("It.:%4i Obj.:%6.3e Vol.:%6.3f ch.:%4.2f relres: %4.2e iters: %4i ",
           loop, c, vol, change, cgres, cgiter);
      #ifdef _OPENMP
        printf("time: %6.3f \n", omp_get_wtime() - start_wtime);
      #else
        printf(" \n");
      #endif
    }
  }

  //#pragma omp target update from(xPhys[:nelem])

  printf("%4i %12.6f %6.3f %4.2f %9i %9i",loop, c, vol, change,nelx*nely*nelz,gciter_total);
  #ifdef _OPENMP
  printf(" %6.3f \n", omp_get_wtime() - start_wtime);
  #else 
  printf("\n");
  #endif
  if (verbose ==1){ 
    #ifdef _OPENMP
      printf("End time: %6.3f \n", omp_get_wtime() - start_wtime);
    #endif
  }
  char name1[60];
  char name2[60];
  if (write_result == 1){
    snprintf(name1, 60, "out_%d_%d_%d.vtu",(int)nelx, (int)nely, (int)nelz);
    writeDensityVtkFile(nelx, nely, nelz, xPhys,name1);
    snprintf(name2, 60, "out_%d_%d_%d_halo.vtu",(int)nelx, (int)nely, (int)nelz);
    writeDensityVtkFileWithHalo(nelx, nely, nelz, xPhys,name2);
  }
  finishCholmod(gridContext, nl - 1, &solverData.bottomSolver,
                solverData.coarseMatrices[nl - 1],verbose);
  freeSolverData(&solverData, nl);
  freeGC(&gridContext, nl);
  freeGPUGrid(&gpu_gc,nl);

  free(dc);
  free(dv);
  free(x);
  free(xPhys);
  free(xnew);
}

// this function acts as a matrix-free replacement for out = (H*rho(:))./Hs
// note that rho and out cannot be the same pointer!
// temperature: cold, called once pr design iteration
void applyDensityFilter(const struct gridContext gc, const DTYPE rmin,
                        const DTYPE *rho, DTYPE *out) {

  const uint32_t nelx = gc.nelx;
  const uint32_t nely = gc.nely;
  const uint32_t nelz = gc.nelz;

  const uint32_t elWrapy = gc.wrapy - 1;
  const uint32_t elWrapz = gc.wrapz - 1;

  //#pragma omp target teams distribute parallel for firstprivate(rmin,elWrapy,elWrapz)
  // It was found that this function was hard to parallelize in an efficient way on the GPU

// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
  #pragma omp parallel for collapse(3) default(none) firstprivate(nelx,nely,nelz,rmin,elWrapy,elWrapz) shared(out,rho)
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {

        const uint32_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        double oute1 = 0.0;
        double unityScale = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil(rmin) + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil(rmin) - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil(rmin) + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil(rmin) - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil(rmin) + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil(rmin) - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const uint32_t e2 = i2 * elWrapy * elWrapz + k2 * elWrapy + j2;

              const double filterWeight =
                  MAX(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              oute1 += filterWeight * rho[e2];
              unityScale += filterWeight;
            }
          }
        }
        out[e1] = oute1 / unityScale;
      }
}

// this function acts as a matrix-free replacement for v = H* (v(:)./Hs)
// note that rho and out cannot be the same pointer!
// temperature: cold, called twice pr design iteration
void applyDensityFilterGradient(const struct gridContext gc, const DTYPE rmin,
                                DTYPE *v) {
  const uint32_t nelx = gc.nelx;
  const uint32_t nely = gc.nely;
  const uint32_t nelz = gc.nelz;
  const uint32_t elWrapy = gc.wrapy - 1;
  const uint32_t elWrapz = gc.wrapz - 1;
  const uint32_t N = (gc.wrapx - 1) * elWrapy * elWrapz;
  DTYPE *tmp = malloc(sizeof(DTYPE) * N);

// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
  const uint_fast64_t nelem = (gc.wrapx - 1) *
                              (gc.wrapy - 1) * (gc.wrapz - 1);
  const double ceil_rmin = ceil(rmin);
  #pragma omp target data map(to:tmp[:N]) //map(always,tofrom:v[:nelem])
  {
  #pragma omp target teams distribute parallel for
  for(int i=0;i<N;i++){
    tmp[i] = 0;
  }
  #pragma omp target teams distribute parallel for collapse(3) schedule(static) firstprivate(ceil_rmin,rmin) // default(none) firstprivate(ceil_rmin,rmin,nelx,nely,nelz) shared(tmp,v)
  for (uint32_t i1 = 1; i1 < nelx + 1; i1++)
    for (uint32_t k1 = 1; k1 < nelz + 1; k1++)
      for (uint32_t j1 = 1; j1 < nely + 1; j1++) {

        const uint32_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        double unityScale = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil_rmin + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil_rmin - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil_rmin + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil_rmin - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil_rmin + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil_rmin - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const double filterWeight =
                  MAX(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              unityScale += filterWeight;
            }
          }
        }
        //#pragma omp atomic write
        tmp[e1] = ((double) v[e1]) / unityScale;
      }
  
// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
  #pragma omp target teams distribute parallel for collapse(3) schedule(static) firstprivate(ceil_rmin,rmin,elWrapy,elWrapz) //default(none) firstprivate(ceil_rmin,nelx,nely,nelz,rmin,elWrapy,elWrapz) shared(tmp,v)
  for (uint32_t i1 = 1; i1 < nelx + 1; i1++)
    for (uint32_t k1 = 1; k1 < nelz + 1; k1++)
      for (uint32_t j1 = 1; j1 < nely + 1; j1++) {

        const uint32_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        double ve1 = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil_rmin + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil_rmin - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil_rmin + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil_rmin - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil_rmin + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil_rmin - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const uint32_t e2 = i2 * elWrapy * elWrapz + k2 * elWrapy + j2;

              const double  filterWeight =
                  MAX(0.0, ((double) rmin) - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              ve1 += filterWeight * tmp[e2];
            }
          }
        }
        v[e1] = ve1;
      }
  }
  free(tmp);
}

// writes a file with a snapshot of the density field (x,xPhys), can be opened
// with paraview temperature: very cold, usually called once only
void writeDensityVtkFile(const int nelx, const int nely, const int nelz,
                         const DTYPE *densityArray, const char *filename) {
  int nx = nelx + 1;
  int ny = nely + 1;
  int nz = nelz + 1;

  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int elWrapy = nely + paddingy + 3 - 1;
  const int elWrapz = nelz + paddingz + 3 - 1;

  int numberOfNodes = nx * ny * nz;
  int numberOfElements = nelx * nely * nelz;

  FILE *fid = fopen(filename, "w");

  // write header
  fprintf(fid, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n");
  fprintf(fid, "<UnstructuredGrid>\n");
  fprintf(fid, "<Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">\n",
          numberOfNodes, numberOfElements);

  // points
  fprintf(fid, "<Points>\n");
  fprintf(fid,
          "<DataArray type=\"Float32\" NumberOfComponents=\"%i\" "
          "format=\"ascii\">\n",
          3);
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        fprintf(fid, "%e %e %e\n", (float)i, (float)j, (float)k);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Points>\n");

  fprintf(fid, "<Cells>\n");

  fprintf(
      fid,
      "<DataArray type=\"Int32\" Name=\"connectivity\" format= \"ascii\">\n");
  for (int i = 0; i < nelx; i++)
    for (int k = 0; k < nelz; k++)
      for (int j = 0; j < nely; j++) {
        const int nx_1 = i;
        const int nx_2 = i + 1;
        const int nz_1 = k;
        const int nz_2 = k + 1;
        const int ny_1 = j;
        const int ny_2 = j + 1;
        fprintf(fid, "%d %d %d %d %d %d %d %d\n",
                nx_1 * ny * nz + nz_1 * ny + ny_2,
                nx_2 * ny * nz + nz_1 * ny + ny_2,
                nx_2 * ny * nz + nz_1 * ny + ny_1,
                nx_1 * ny * nz + nz_1 * ny + ny_1,
                nx_1 * ny * nz + nz_2 * ny + ny_2,
                nx_2 * ny * nz + nz_2 * ny + ny_2,
                nx_2 * ny * nz + nz_2 * ny + ny_1,
                nx_1 * ny * nz + nz_2 * ny + ny_1);
      }

  fprintf(fid, "</DataArray>\n");

  fprintf(fid,
          "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  for (int i = 1; i < numberOfElements + 1; i++)
    fprintf(fid, "%d\n", i * 8);
  fprintf(fid, "</DataArray>\n");

  fprintf(fid, "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int i = 0; i < numberOfElements; i++)
    fprintf(fid, "%d\n", 12);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Cells>\n");

  fprintf(fid, "<CellData>\n");
  fprintf(fid, "<DataArray type=\"Float32\" NumberOfComponents=\"1\" "
               "Name=\"density\" format=\"ascii\">\n");
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {
        const uint64_t idx = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;
        fprintf(fid, "%e\n", densityArray[idx]);
      }
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</CellData>\n");

  fprintf(fid, "</Piece>\n");
  fprintf(fid, "</UnstructuredGrid>\n");
  fprintf(fid, "</VTKFile>\n");

  fclose(fid);
}

// writes a file with a snapshot of the density field (x,xPhys), can be opened
// with paraview temperature: very cold, usually called once only
void writeDensityVtkFileWithHalo(const int nelx, const int nely, const int nelz,
                                 const DTYPE *densityArray,
                                 const char *filename) {

  const int paddingx =
      (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapx = nelx + paddingx + 3;
  const int wrapy = nely + paddingy + 3;
  const int wrapz = nelz + paddingz + 3;

  const int elWrapx = wrapx - 1;
  const int elWrapy = wrapy - 1;
  const int elWrapz = wrapz - 1;

  int numberOfNodes = wrapx * wrapy * wrapz;
  int numberOfElements = elWrapx * elWrapy * elWrapz;

  FILE *fid = fopen(filename, "w");

  // write header
  fprintf(fid, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n");
  fprintf(fid, "<UnstructuredGrid>\n");
  fprintf(fid, "<Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">\n",
          numberOfNodes, numberOfElements);

  // points
  fprintf(fid, "<Points>\n");
  fprintf(fid,
          "<DataArray type=\"Float32\" NumberOfComponents=\"%i\" "
          "format=\"ascii\">\n",
          3);
  for (int i = 0; i < wrapx; i++)
    for (int k = 0; k < wrapz; k++)
      for (int j = 0; j < wrapy; j++)
        fprintf(fid, "%e %e %e\n", (float)i, (float)j, (float)k);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Points>\n");

  fprintf(fid, "<Cells>\n");

  fprintf(
      fid,
      "<DataArray type=\"Int32\" Name=\"connectivity\" format= \"ascii\">\n");
  for (int i = 0; i < elWrapx; i++)
    for (int k = 0; k < elWrapz; k++)
      for (int j = 0; j < elWrapy; j++) {
        const int nx_1 = i;
        const int nx_2 = i + 1;
        const int nz_1 = k;
        const int nz_2 = k + 1;
        const int ny_1 = j;
        const int ny_2 = j + 1;
        fprintf(fid, "%d %d %d %d %d %d %d %d\n",
                nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_1);
      }

  fprintf(fid, "</DataArray>\n");

  fprintf(fid,
          "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  for (int i = 1; i < numberOfElements + 1; i++)
    fprintf(fid, "%d\n", i * 8);
  fprintf(fid, "</DataArray>\n");

  fprintf(fid, "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int i = 0; i < numberOfElements; i++)
    fprintf(fid, "%d\n", 12);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Cells>\n");

  fprintf(fid, "<CellData>\n");
  fprintf(fid, "<DataArray type=\"Float32\" NumberOfComponents=\"1\" "
               "Name=\"density\" format=\"ascii\">\n");
  for (unsigned int i1 = 0; i1 < elWrapx; i1++)
    for (unsigned int k1 = 0; k1 < elWrapz; k1++)
      for (unsigned int j1 = 0; j1 < elWrapy; j1++) {
        const uint64_t idx = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;
        fprintf(fid, "%e\n", densityArray[idx]);
      }
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</CellData>\n");

  fprintf(fid, "</Piece>\n");
  fprintf(fid, "</UnstructuredGrid>\n");
  fprintf(fid, "</VTKFile>\n");

  fclose(fid);
}
