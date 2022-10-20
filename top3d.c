#include "getopt.h"

#include "definitions.h"

#include "local_matrix.h"
#include "stencil_grid_utility.h"
#include "stencil_methods.h"

#include "stencil_assembly.h"
#include "stencil_optimization.h"
#include "stencil_solvers.h"

// todo: accept parameters as command line arguments
int main(int argc, char *argv[]) {

  int nelx_coarse = 12;
  int nely_coarse = 6;
  int nelz_coarse = 6;
  float volfrac = 0.2;
  float rmin = 1.5;
  int iters = 20;
  int nl = 4;
  int verbose = 0;
  int write_result = 0;
  int opt;

  while ((opt = getopt(argc, argv, "x:y:z:r:f:i:l:v:w:")) != -1) {
    switch (opt) {
    case 'x':
      nelx_coarse = atoi(optarg);
      break;
    case 'y':
      nely_coarse = atoi(optarg);
      break;
    case 'z':
      nelz_coarse = atoi(optarg);
      break;
    case 'r':
      rmin = atof(optarg);
      break;
    case 'f':
      volfrac = atof(optarg);
      break;
    case 'i':
      iters = atoi(optarg);
      break;
    case 'l':
      nl = atoi(optarg);
      break;
    case 'v':
      verbose = atoi(optarg);
      break;
    case 'w':
      write_result = atoi(optarg);
      break;
    default:
      fprintf(stderr, "Usage: %s [-xyzrfilvw]\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  int sizeIncr = 2;
  for (int i = 2; i < nl; i++)
    sizeIncr *= 2;

  const int nelx = nelx_coarse * sizeIncr;
  const int nely = nely_coarse * sizeIncr;
  const int nelz = nelz_coarse * sizeIncr;

  if (verbose) {
    printf(
        "Running topopt with:\n number of coarse els x (-x): %i (fine els = "
        "%i)\n number of coarse els y (-y): %i (fine els = %i)\n number of "
        "coarse els z (-z): %i (fine els = %i)\n total number of elements: "
        "%i\n volume fraction (-f): %f\n filter radius in elements (-r): %f\n "
        "number of design iterations (-i): %i\n verbose (-v): %i\n write file "
        "(-w): %i\n\n",
        nelx_coarse, nelx, nely_coarse, nely, nelz_coarse, nelz,
        nelx * nely * nelz, volfrac, rmin, iters, verbose, write_result);
  }

  const float cgtol = 1e-5;
  const int cgmax = 200;

  top3dmgcg(nelx, nely, nelz, volfrac, rmin, nl, cgtol, cgmax, verbose,
            write_result, iters);
  return 0;
}
