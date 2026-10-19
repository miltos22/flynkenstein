#include <stddef.h>
long long lz76(const unsigned char *s, long long n) {
    if (n <= 1) return n;
    long long c = 1, l = 1, i = 0, k = 1, kmax = 1;
    while (1) {
        if (s[i + k - 1] == s[l + k - 1]) {
            k++;
            if (l + k > n) { c++; break; }
        } else {
            if (k > kmax) kmax = k;
            i++;
            if (i == l) {
                c++;
                l += kmax;
                if (l + 1 > n) break;
                i = 0; k = 1; kmax = 1;
            } else {
                k = 1;
            }
        }
    }
    return c;
}
#include <stdlib.h>
long long lz76_linear(const unsigned char *s, long long n) {
    if (n <= 1) return n;
    long long cap = 2 * n + 2;
    int *nx0 = malloc(cap * sizeof(int)), *nx1 = malloc(cap * sizeof(int));
    int *lnk = malloc(cap * sizeof(int)), *len = malloc(cap * sizeof(int));
    if (!nx0 || !nx1 || !lnk || !len) { free(nx0); free(nx1); free(lnk); free(len); return -1; }
    long long sz = 1; int last = 0;
    nx0[0] = nx1[0] = -1; lnk[0] = -1; len[0] = 0;
    long long built = 0, pos = 0, c = 0;
    int cur = 0; long long L = 0;
    while (pos < n) {
        if (pos + L >= n) { c++; break; }
        while (built < pos + L) {
            int ch = s[built], z = (int)sz++;
            len[z] = len[last] + 1; nx0[z] = nx1[z] = -1;
            int p = last;
            while (p != -1 && (ch ? nx1[p] : nx0[p]) == -1) { if (ch) nx1[p] = z; else nx0[p] = z; p = lnk[p]; }
            if (p == -1) lnk[z] = 0;
            else {
                int q = ch ? nx1[p] : nx0[p];
                if (len[p] + 1 == len[q]) lnk[z] = q;
                else {
                    int cl = (int)sz++;
                    len[cl] = len[p] + 1; nx0[cl] = nx0[q]; nx1[cl] = nx1[q]; lnk[cl] = lnk[q];
                    while (p != -1 && (ch ? nx1[p] : nx0[p]) == q) { if (ch) nx1[p] = cl; else nx0[p] = cl; p = lnk[p]; }
                    lnk[q] = lnk[z] = cl;
                    if (cur == q && L <= len[cl]) cur = cl;
                }
            }
            last = z; built++;
        }
        int ch = s[pos + L];
        int t = ch ? nx1[cur] : nx0[cur];
        if (t != -1) { cur = t; L++; }
        else { c++; pos += L + 1; cur = 0; L = 0; }
    }
    free(nx0); free(nx1); free(lnk); free(len);
    return c;
}
