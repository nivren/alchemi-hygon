/* SPDX-License-Identifier: Apache-2.0
 * M3 observer adds host solve/LU symbols to the E1 eigen observer.
 * Does not modify arguments or results. Linux LP64 LAPACKE ABI.
 */
#include "ase_bfgs_lapack_trace.c"

void dgetrf_(const int *m, const int *n, double *a, const int *lda,
             int *ipiv, int *info) {
    typedef void (*fn)(const int *, const int *, double *, const int *, int *, int *);
    fn original = (fn)dlsym(RTLD_NEXT, "dgetrf_");
    if (!original) abort();
    double start = now();
    original(m, n, a, lda, ipiv, info);
    record("dgetrf_", *n, now() - start, *info);
}

void dgesv_(const int *n, const int *nrhs, double *a, const int *lda,
            int *ipiv, double *b, const int *ldb, int *info) {
    typedef void (*fn)(const int *, const int *, double *, const int *, int *, double *, const int *, int *);
    fn original = (fn)dlsym(RTLD_NEXT, "dgesv_");
    if (!original) abort();
    double start = now();
    original(n, nrhs, a, lda, ipiv, b, ldb, info);
    record("dgesv_", *n, now() - start, *info);
}

int LAPACKE_dgesv(int layout, int n, int nrhs, double *a, int lda,
                 int *ipiv, double *b, int ldb) {
    typedef int (*fn)(int, int, int, double *, int, int *, double *, int);
    fn original = (fn)dlsym(RTLD_NEXT, "LAPACKE_dgesv");
    if (!original) abort();
    double start = now();
    int info = original(layout, n, nrhs, a, lda, ipiv, b, ldb);
    record("LAPACKE_dgesv", n, now() - start, info);
    return info;
}

int LAPACKE_dgetrf(int layout, int m, int n, double *a, int lda, int *ipiv) {
    typedef int (*fn)(int, int, int, double *, int, int *);
    fn original = (fn)dlsym(RTLD_NEXT, "LAPACKE_dgetrf");
    if (!original) abort();
    double start = now();
    int info = original(layout, m, n, a, lda, ipiv);
    record("LAPACKE_dgetrf", n, now() - start, info);
    return info;
}

int LAPACKE_dgetrs(int layout, char trans, int n, int nrhs, const double *a,
                  int lda, const int *ipiv, double *b, int ldb) {
    typedef int (*fn)(int, char, int, int, const double *, int, const int *, double *, int);
    fn original = (fn)dlsym(RTLD_NEXT, "LAPACKE_dgetrs");
    if (!original) abort();
    double start = now();
    int info = original(layout, trans, n, nrhs, a, lda, ipiv, b, ldb);
    record("LAPACKE_dgetrs", n, now() - start, info);
    return info;
}
