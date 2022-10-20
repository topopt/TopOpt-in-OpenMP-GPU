#pragma once

#include "definitions.h"

void setFixedDof_halo(struct gridContext *gc, const int l);

void setupGC(struct gridContext *gc, const int nl, const int nelx, const int nely, const int nelz);

void freeGC(struct gridContext *gc, const int nl);

void allocateZeroPaddedStateField(const struct gridContext gc, const int l,
                                  CTYPE **v);
void allocateZeroPaddedStateField_MTYPE(const struct gridContext gc,
                                        const int l, MTYPE **v);

void allocateZeroPaddedStateField_STYPE(const struct gridContext gc,
                                        const int l, STYPE **v);
