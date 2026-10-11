/* SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
 * SPDX-License-Identifier: Apache-2.0
 * Probe-only LD_PRELOAD observer. Calls the original LAPACK without modifying
 * parameters/results. No product dependency or eigensolver replacement.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void record(const char *symbol, int n, double elapsed, int info) {
    const char *path = getenv("BFGS_LAPACK_LOG");
    if (!path) return;
    FILE *file = fopen(path, "a");
    if (file) {
        fprintf(file, "%s n=%d wall_ms=%.9f info=%d\n", symbol, n, elapsed * 1e3, info);
        fclose(file);
    }
}

int LAPACKE_dstedc(int layout, char compz, int n, double *d, double *e, double *z, int ldz) {
    typedef int (*fn)(int, char, int, double *, double *, double *, int);
    fn original = (fn)dlsym(RTLD_NEXT, "LAPACKE_dstedc");
    if (!original) abort();
    double start = now();
    int info = original(layout, compz, n, d, e, z, ldz);
    record("LAPACKE_dstedc", n, now() - start, info);
    return info;
}

int LAPACKE_dstedc_work(int layout, char compz, int n, double *d, double *e, double *z,
                      int ldz, double *work, int lwork, int *iwork, int liwork) {
    typedef int (*fn)(int, char, int, double *, double *, double *, int, double *, int, int *, int);
    fn original = (fn)dlsym(RTLD_NEXT, "LAPACKE_dstedc_work");
    if (!original) abort();
    double start = now();
    int info = original(layout, compz, n, d, e, z, ldz, work, lwork, iwork, liwork);
    record("LAPACKE_dstedc_work", n, now() - start, info);
    return info;
}
