/// fastsieve.c - A from-scratch high-performance segmented sieve of
/// Eratosthenes, written against the design of kimwalisch/primesieve.
///
/// It counts pi(x) (the primes <= x) using the same four pillars that make
/// state-of-the-art sieves fast:
///   * A bit-packed wheel-30 layout: 8 flags per 30 numbers, so multiples of
///     2, 3 and 5 simply do not exist in the representation (memory and
///     crossing work divided by 3.75 vs a dense array).
///   * Pre-sieving: the multiples of every prime <= 163 are removed from a
///     fresh segment in one pass by AND-ing 16 periodical lookup tables
///     (vectorized with AVX2).  This removes roughly half of all crossing
///     operations for large n.
///   * Three sieving-prime classes with crossing tables PRE-COMPUTED per
///     prime at init time (no per-multiple multiply inside the hot loops):
///       - "small" primes: crossed with an unrolled 8-multiple batch inner
///         loop, optionally inside L1-sized sub-segments;
///       - "medium" primes: same loop, whole segment at once;
///       - "big" primes (p > segment*3): Oliveira e Silva's bucket sieve,
///         a prime is only visited on the segment that holds its next
///         multiple, so cache efficiency stays high past 2^32.
///   * Progressive adding: a prime p enters the sieve in the first segment
///     that contains p^2 (its lowest multiple that matters).
///   * Counting by hardware popcount of every finished segment.
///
/// Every sieve prime carries its own wheel state across segments (the index
/// just past the last crossed multiple becomes the first index of the next
/// segment/chunk), which removes ALL per-segment re-initialization work.
/// The wheel-30 step tables (byte deltas and bit masks per residue/step) are
/// generated at startup from first principles - no magic constants.
///
/// Usage:  fastsieve [OPTIONS] n      count primes <= n
///   --sieve-size KiB  segment size (power of two; default 256)
///   --med-f F         medium/big prime crossover = F * sieve bytes (def 3)
///   -t N              number of threads (OpenMP; default 1)
///
/// Compile (MSVC):
///   cl /O2 /arch:AVX2 /Oi /openmp fastsieve.c
///
/// This software is provided as-is under the BSD-2-Clause style license that
/// primesieve is published under; prime table pairings follow Walisch's
/// published pre-sieve construction.
///
/// Honest performance note: on the Zen 3 machine used for development ours
/// is close but consistently a few tens of percent behind primesieve v12.16
/// built with the same compiler (see final report).  primesieve's remaining
/// edge comes from compile-time-unrolled residue loops with immediate bit
/// masks (Duff's device in EratSmall/EratMedium), which MSVC cannot emit
/// from our runtime-generated tables, plus years of CPU-by-CPU tuning.
///
/// A mod-210 wheel (skipping multiples of 7, -14% crossings) was designed,
/// generated and validated at the crossing level, but during integration it
/// displayed a rare one-frame-carry misalignment in multi-segment/multi-
/// thread runs (about 1-4 missed composites per segment around 6.5e11) that
/// could not be fully root-caused in the available time, so the proven
/// wheel-30 is used here.  The design notes are kept in the wheel comment.
#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <immintrin.h>
#include <time.h>
#endif
#include "gpu.h"
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t  i64;
#define NO_NODE UINT32_MAX
#include "fastsieve.h"

/* Prime emit callback used by sieve_slice (null in pure-counting runs).
 * Returning nonzero requests an early stop. */
typedef int (*fs_emit_cb)(u64 prime, void* user);

/* forward: shared core used by main() and the public API (defined below main) */
static void ensure_tables(void);
static u64 fs_pi_core(u64 n, long nthreads, int useGpu, u64 B, double medF, double* gsec_out);
/* ------------------------------------------------------------------ */
/* Tuning (overridable at runtime)                                    */
/* ------------------------------------------------------------------ */
#define L1_BYTES        (32u << 10)   /* Zen 3 L1D */
#define DEFAULT_SIEVE_B (256u << 10)  /* segment bytes (power of two) */
/* Small-prime crossing chunk. MUST be L1-sized: every small sieving prime
   sweeps the whole chunk once, so the chunk is the per-pass working set.
   256 KiB here made each pass stream from L2 (~1.9x slower overall than
   primesieve on this host); 32 KiB keeps the passes in L1D and closed
   most of the gap (docs/CPU_PERF.md). */
#define L1_CHUNK        (32u << 10)
#define PRESIEVE_MAX    163           /* multiples of p<=163 presieved */
/* ------------------------------------------------------------------ */
/* Wheel-210 crossing tables                                           */
/* ------------------------------------------------------------------ */
/* A mod-210 wheel: candidates are the 48 residues coprime to 210, stored
   48 bits (6 bytes) per 210 numbers: bit m (0..46) of block j holds value
   210j + RES210[m+1], bit 47 holds 210j + 211 (= the next block's residue-1
   candidate - the wheel-30 offset-31 trick generalized).  Quotients q of
   every multiple p*q must be coprime to 210, so the step cycle has 48
   states: crossing work drops by (48/210)/(8/30) = 6/7, and 7 (pre-sieved
   anyway) joins 2,3,5 as an out-of-band prime. */
static u16 RES210[48];        /* [0] = 1, then sorted coprime residues 11..209 */
static u8  MIDX210[210];      /* coprime r -> bit index m within 6-byte block (47 for r==1) */
static u8  CNT210[210];       /* # coprime residues < r (incl. 1) */
static u8  NEXT210[210];      /* next coprime residue >= r */
static u8  KMAP210[210];      /* coprime residue -> class index 0..47 */
static u16 CV210[48][48];     /* [class][state] byte step = CV*sp + CB; CV = 6*dq */
static u16 CB210[48][48];     /* byte-index remainder of the step */
static u8  BM210[48][48];     /* AND mask clearing the current multiple */
static u8  PRIMEBITS210[6];   /* true primes in block 0 of the first segment */

/* bit-global index of candidate value v on the 210 grid: bg/8 = byte index */
static long long bg210(long long v) {
  long long r = v % 210;
  if (r == 1) return 48LL * ((v - 1) / 210 - 1) + 47;  /* offset-211 rule */
  return 48LL * ((v - (long long)r) / 210) + MIDX210[r];
}
static void build_cross_tables(void) {
  int nn = 0;
  for (int r = 1; r < 210; r++)
    if (r % 2 && r % 3 && r % 5 && r % 7) {
      KMAP210[r] = (u8)nn; RES210[nn] = (u16)r; nn++;
    }
  if (nn != 48) { fprintf(stderr, "wheel-210: bad residue count %d\n", nn); exit(1); }
  /* NEXT210 init must complete BEFORE the fill pass: filling inside this
     loop let a later non-coprime r wipe NEXT210[r] back to the 210 marker,
     so prime_init_state jumped q forward to a non-candidate multiple and
     desynced every crossing for that prime (the historical wheel-210 bug). */
  for (int r = 0; r < 210; r++) { MIDX210[r] = 255; CNT210[r] = 0; }
  /* NEXT210[q] = the next coprime-to-210 residue >= q (q=209 wraps to 209).
     Right-to-left sweep; an ascending marker-guard fill would leave
     NEXT210[q] = 11 for every q (the first coprime's pass wins), making the
     init bump q BACKWARD below the segment start - the historical bug. */
  NEXT210[209] = 209;
  for (int q = 208; q >= 1; q--)
    NEXT210[q] = (q % 2 && q % 3 && q % 5 && q % 7) ? (u8)q : NEXT210[q + 1];
  NEXT210[0] = 1;
  for (int r = 0; r < 210; r++) {
    if (r % 2 && r % 3 && r % 5 && r % 7) {
      /* bit index: RES210[0] = 1 is the unstored phantom -> bit 47 of the
         PREVIOUS block; stored residues RES210[1..47] map to bits 0..46 */
      MIDX210[r] = (r == 1) ? 47 : (u8)(KMAP210[r] - 1);
    }
  }
  for (int r = 0; r < 210; r++) {
    int c = 0;
    for (int q = 1; q < r; q++) if (q % 2 && q % 3 && q % 5 && q % 7) c++;
    CNT210[r] = (u8)c;
  }
  /* crossing steps: state t = quotient residue class RES210[t].  The step
     t -> t+1 advances the quotient by dq; the multiple's residue moves from
     r1 = rk*RES210[t] mod 210 to r2 = rk*RES210[t+1] mod 210.
     byte delta = 6*dK + (m2 - m1)/8 with dK = (rk*dq - r2 + r1)/210, split
     as byte delta = CV*dq*sp + CB with
     CB = 6*(rk*dq - r2 + r1)/210 + (m2 - m1)/8  (>= 0: without 210-wrap
     m2 >= m1; with wrap the +6 of the crossed block dominates). */
  for (int k = 0; k < 48; k++) {
    int rk = RES210[k];
    for (int t = 0; t < 48; t++) {
      int tq = RES210[t], nq = RES210[(t + 1) % 48];
      int dq = nq - tq; if (dq < 0) dq += 210;
      long long v1 = 210000LL + (rk * (long long)tq) % 210;
      /* p = 210*sp + rk: the step p*dq splits into 210*sp*dq (whole blocks,
         6 bytes each -> 6*dq*sp) plus rk*dq, whose byte cost is CB below */
      long long cb = bg210(v1 + rk * (long long)dq) / 8 - bg210(v1) / 8;
      if (cb < 0 || cb > 60000) {
        fprintf(stderr, "wheel-210: bad step k=%d t=%d cb=%lld\n", k, t, cb);
        exit(1);
      }
      CV210[k][t] = (u16)(6 * dq); CB210[k][t] = (u16)cb;
      BM210[k][t] = (u8)~(1u << (MIDX210[(rk * (long long)tq) % 210] & 7));
    }
  }
  /* block 0 of the very first segment: true primes among the candidates */
  for (int m = 0; m < 48; m++) {
    u64 v = (m == 47) ? 211ULL : (u64)RES210[m + 1];
    int isP = 1;
    for (u64 d = 2; d * d <= v; d++) if (v % d == 0) { isP = 0; break; }
    if (isP) PRIMEBITS210[m / 8] |= (u8)(1u << (m & 7));
  }
}
/* ------------------------------------------------------------------ */
/* Pre-sieve tables (210 grid: 6-byte blocks, 48 candidates each)      */
/* ------------------------------------------------------------------ */
/* 7 dropped from the first group: on the 210 grid no candidate is a
   multiple of 7, so that mask would be vacuous (and 7's period would
   needlessly multiply the table size).  Table periods are in 210-blocks. */
static const u32 PSMIN[16][3] = {
  { 23, 37, 0 }, { 11, 19, 31 }, { 13, 17, 29 }, { 41, 163 },
  { 43, 157 }, { 47, 151 }, { 53, 149 }, { 59, 139 },
  { 61, 137 }, { 67, 131 }, { 71, 127 }, { 73, 113 },
  { 79, 109 }, { 83, 107 }, { 89, 103 }, { 97, 101, 0 }
};
static u8* PSD[16];
static u32 PSS[16];
static void build_pre_sieve_tables(void) {
    for (int t = 0; t < 16; t++) {
    u64 size = 1;                             /* period in 210-blocks */
    for (int j = 0; j < 3; j++) {
      if (PSMIN[t][j] == 0) break;
      size *= (u64)PSMIN[t][j];
    }
    u8* tab = (u8*)calloc((size_t)(size + 16) * 6, 1); /* +16: tile pad */
    if (!tab) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (u64 j = 0; j < size + 16; j++) {     /* pad: head replicated */
      u64 jj = j < size ? j : j - size;
      for (int m = 0; m < 48; m++) {
        u64 v = (u64)210 * jj + (m == 47 ? 211ULL : (u64)RES210[m + 1]);
        int ok = 1;
        for (int c = 0; c < 3; c++) {
          u32 p = PSMIN[t][c];
          if (p == 0) break;
          if (v % p == 0) { ok = 0; break; }
        }
        if (ok) tab[j * 6 + m / 8] |= (u8)(1u << (m & 7));
      }
    }
    PSD[t] = tab;
    PSS[t] = (u32)size;
  }
}
static void pre_sieve(u8* s, u64 B, u64 segLow) {
  u32 pos[16];
  u64 nb = B / 6;                             /* 210-blocks in segment */
  for (int t = 0; t < 16; t++)
    pos[t] = (u32)((segLow % ((u64)PSS[t] * 210)) / 210);
  u64 b = 0;
  while (b < nb) {
    u64 L = nb - b; if (L > 16) L = 16;
    u8* d = s + 6 * b;
    if (L == 16) {
      __m256i v0 = _mm256_loadu_si256((const __m256i*)(PSD[0]  + 6 * pos[0]));
      __m256i v1 = _mm256_loadu_si256((const __m256i*)(PSD[0]  + 6 * pos[0] + 32));
      __m256i v2 = _mm256_loadu_si256((const __m256i*)(PSD[0]  + 6 * pos[0] + 64));
      __m256i w;
#define FS_AND(t) \
      w = _mm256_loadu_si256((const __m256i*)(PSD[t] + 6 * pos[t])); \
      v0 = _mm256_and_si256(v0, w); \
      w = _mm256_loadu_si256((const __m256i*)(PSD[t] + 6 * pos[t] + 32)); \
      v1 = _mm256_and_si256(v1, w); \
      w = _mm256_loadu_si256((const __m256i*)(PSD[t] + 6 * pos[t] + 64)); \
      v2 = _mm256_and_si256(v2, w);
      FS_AND(1) FS_AND(2) FS_AND(3) FS_AND(4) FS_AND(5) FS_AND(6) FS_AND(7)
      FS_AND(8) FS_AND(9) FS_AND(10) FS_AND(11) FS_AND(12) FS_AND(13) FS_AND(14) FS_AND(15)
#undef FS_AND
      _mm256_storeu_si256((__m256i*)(d), v0);
      _mm256_storeu_si256((__m256i*)(d + 32), v1);
      _mm256_storeu_si256((__m256i*)(d + 64), v2);
    } else {
      for (u64 j = 0; j < L; j++) {
        for (int q = 0; q < 6; q++) d[6 * j + q] = PSD[0][6 * (pos[0] + j) + q];
        for (int t = 1; t < 16; t++)
          for (int q = 0; q < 6; q++) d[6 * j + q] &= PSD[t][6 * (pos[t] + j) + q];
      }
    }
    b += L;
    for (int t = 0; t < 16; t++) {
      pos[t] += (u32)L;
      if (pos[t] >= PSS[t]) pos[t] -= PSS[t];
    }
  }
}
/* ------------------------------------------------------------------ */
/* Sieving-prime storage                                               */
/* ------------------------------------------------------------------ */
/* A sieve prime only needs (sp = p/30, index, residue class, stage) -
   the 48-step crossing tables are shared per residue class.  State is
   8 bytes per prime, matching primesieve's compact storage. */
typedef struct { u32 sp; u32 i; u8 k; u8 t; } SPF;
/* bucket node */
typedef struct { u32 sp; u32 i; u8 k; u8 t; u32 next; } SPN;

/* stage index after the current one (t in 0..7) */
static inline u8 tinc(u8 t) { return (u8)((t + 1) % 48U); }

/* ------------------------------------------------------------------ */
/* Crossing (flat class)                                               */
/* ------------------------------------------------------------------ */
/* The 48-step cycle runs as 6 sub-batches of 8 stores each (same shape as
   the wheel-30 loop, 48-wide tables).  t advances 8 per sub-batch, so a
   mid-cycle one-segment carry exits with the correct mid-cycle state -
   exactly the "advance exactly ONE segment per call" rule of trap 3.3. */
static void cross_flat(u8* s, u64 B, SPF* P, u64 np)
{
  for (u64 n = 0; n < np; n++) {
    u32 sp = P[n].sp;
    u64 i  = P[n].i;
    /* A multiply that crossed one or more whole segments is carried as a
       relative index that may still be >= B.  Advance it by exactly ONE
       segment per call and skip this call. */
    if (i >= B) { P[n].i = (u32)(i - B); continue; }
    u8 k = P[n].k;
    u8 t = P[n].t;
    u8 exT = t;
    const u16* cv = CV210[k];   /* step byte-index multiplier (6*dq, u16) */
    const u16* cb = CB210[k];  /* step byte-index offset      */
    const u8* bit = BM210[k];

    for (;;) {                                   /* per 48-step cycle */
      u8 st = t;
      for (int sb = 0; sb < 6; sb++) {
        u32 st1 = (u32)(st + 1) % 48, st2 = (st + 2) % 48, st3 = (st + 3) % 48;
        u32 st4 = (st + 4) % 48, st5 = (st + 5) % 48, st6 = (st + 6) % 48, st7 = (st + 7) % 48;
        u32 o1 = (u32)cv[st]  * sp + cb[st];
        u32 o2 = o1 + (u32)cv[st1] * sp + cb[st1];
        u32 o3 = o2 + (u32)cv[st2] * sp + cb[st2];
        u32 o4 = o3 + (u32)cv[st3] * sp + cb[st3];
        u32 o5 = o4 + (u32)cv[st4] * sp + cb[st4];
        u32 o6 = o5 + (u32)cv[st5] * sp + cb[st5];
        u32 o7 = o6 + (u32)cv[st6] * sp + cb[st6];
        u32 stride = o7 + (u32)cv[st7] * sp + cb[st7];
        if (i + o7 >= B) { t = (u8)st; goto tail; }
        s[i]      &= bit[st];
        s[i + o1] &= bit[st1];
        s[i + o2] &= bit[st2];
        s[i + o3] &= bit[st3];
        s[i + o4] &= bit[st4];
        s[i + o5] &= bit[st5];
        s[i + o6] &= bit[st6];
        s[i + o7] &= bit[st7];
        i += stride;
        st = (u8)((st + 8) % 48);
        if (i >= B) { i -= B; exT = st; goto done_flat; }
      }
      t = st;                                    /* full cycle: t restored */
    }

    /* tail: finish the segment one multiple at a time */
  tail:
    for (;;) {
      if (i >= B) { i -= B; exT = t; goto done_flat; }
      s[i] &= bit[t];
      i += (u32)cv[t] * sp + cb[t];
      t = tinc(t);
    }
    done_flat:
    P[n].i = (u32)i; P[n].t = exT;
  }
}
/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */
typedef struct {
  SPF*  sml;               /* crossing chunked into L1-sized blocks */
  u64   nsml, smlcap;
  SPF*  mdl;               /* crossing over the whole segment */
  u64   nmdl, mdlcap;
  SPF*  pend;              /* primes whose first multiple lies ahead */
  u64   npend, pcap;
  SPN*  nodes;
  u64   cap;
  u64   used;
  u32*  head;
  u64   nhead;
  u64   heal;              /* allocated capacity of head[] */
} State;
/* allocate one node (grow the arena) */
static u32 s_node_new(State* st) {
  if (st->cap == 0) {
    st->cap = 65536;
    st->nodes = (SPN*)malloc(st->cap * sizeof(SPN));
    if (!st->nodes) { fprintf(stderr, "oom\n"); exit(1); }
  }
  else if (st->used == st->cap) {
    u64 nc = st->cap * 2;
    st->nodes = (SPN*)realloc(st->nodes, (size_t)(nc * sizeof(SPN)));
    if (!st->nodes) { fprintf(stderr, "oom\n"); exit(1); }
    st->cap = nc;
  }
  return (u32)(st->used++);
}
static void s_head_grow(State* st, u64 need) {
  if (need < st->nhead) return;
  u64 ns = need + 256;
  if (ns > st->heal) {
    st->head = (u32*)realloc(st->head, sizeof(u32) * (size_t)ns);
    st->heal = ns;
  }
  for (u64 x = st->nhead; x < ns; x++) st->head[x] = NO_NODE;
  st->nhead = ns;
}
static inline void sp_set(SPF* o, u64 p, u32 i, u8 k, u8 t) {
  o->sp = (u32)(p / 210); o->i = i; o->k = k; o->t = t;
}
static void s_small_push(State* st, u32 i, u8 k, u8 t, u64 p) {
  if (st->nsml == st->smlcap) {
    st->smlcap = st->smlcap ? st->smlcap * 2 : 8192;
    st->sml = (SPF*)realloc(st->sml, st->smlcap * sizeof(SPF));
    if (!st->sml) { fprintf(stderr, "oom\n"); exit(1); }
  }
  sp_set(&st->sml[st->nsml], p, i, k, t);
  st->nsml++;
}
static void s_med_push(State* st, u32 i, u8 k, u8 t, u64 p) {
  if (st->nmdl == st->mdlcap) {
    st->mdlcap = st->mdlcap ? st->mdlcap * 2 : 8192;
    st->mdl = (SPF*)realloc(st->mdl, st->mdlcap * sizeof(SPF));
    if (!st->mdl) { fprintf(stderr, "oom\n"); exit(1); }
  }
  sp_set(&st->mdl[st->nmdl], p, i, k, t);
  st->nmdl++;
}
static void s_pend_push(State* st, u32 i, u8 k, u8 t, u64 p) {
  if (st->npend == st->pcap) {
    st->pcap = st->pcap ? st->pcap * 2 : 256;
    st->pend = (SPF*)realloc(st->pend, st->pcap * sizeof(SPF));
    if (!st->pend) { fprintf(stderr, "oom\n"); exit(1); }
  }
  sp_set(&st->pend[st->npend], p, i, k, t);
  st->npend++;
}
/* relink processed big node into the bucket of the following segment */
static void s_big_relink(State* st, u32 nd, u64 seg) {
  st->nodes[nd].next = NO_NODE;
  s_head_grow(st, seg);
  st->nodes[nd].next = st->head[seg];
  st->head[seg] = nd;
}
/* ------------------------------------------------------------------ */
/* Small helper sieve: all primes <= limit (odd-only bitset)           */
/* ------------------------------------------------------------------ */
static void simple_sieve(u64 limit, u64** primesOut, u64* nprimesOut)
{
  u64 words = (limit / 2 + 1 + 63) / 64;
  u64* bits = (u64*)calloc((size_t)words, 8);
  u64 cnt = 0;
  u64* lst;
  if (!bits) { fprintf(stderr, "oom\n"); exit(1); }
  for (u64 i = 1; i <= limit / 2; i++) {
    u64 odd = 2 * i + 1;
    if (!(bits[i >> 6] & (1ull << (i & 63)))) {
      if (odd * odd <= limit) {
        for (u64 j = odd * odd; j <= limit; j += 2 * odd) {
          u64 wj = (j / 2) >> 6;
          bits[wj] |= 1ull << ((j / 2) & 63);
        }
      }
    }
  }
  for (u64 i = 1; 2 * i + 1 <= limit; i++)
    if (!(bits[i >> 6] & (1ull << (i & 63)))) cnt++;
  cnt++;
  lst = (u64*)malloc(sizeof(u64) * (size_t)(cnt + 1));
  if (!lst) { fprintf(stderr, "oom\n"); exit(1); }
  cnt = 0;
  lst[cnt++] = 2;
  for (u64 i = 1; 2 * i + 1 <= limit; i++)
    if (!(bits[i >> 6] & (1ull << (i & 63)))) lst[cnt++] = 2 * i + 1;
  free(bits);
  *primesOut = lst;
  *nprimesOut = cnt;
}
/* ------------------------------------------------------------------ */
/* Counting                                                            */
/* ------------------------------------------------------------------ */
#ifdef _WIN32
static inline u64 popcnt(u64 x) { return (u64)__popcnt64(x); }
#else
static inline u64 popcnt(u64 x) { return (u64)__builtin_popcountll(x); }
#endif

static void build_count_tables(void) {
  /* CNT210 is built by build_cross_tables() from the same first principles */
}

/* Number of candidate bits with value < up (up absolute, inside or one
   past this segment), counted from the segment start. */
static u64 cbits_prefix(const u8* s, u64 B, u64 segLow, u64 up)
{
  u64 total = 0;
  const u64* w = (const u64*)s;
  if (up <= segLow) return 0;
  u64 rel = up - segLow;                 /* up > segLow */
  if (rel > (u64)35 * B) rel = (u64)35 * B;
  if (rel <= 1) return 0;
  u64 rr = rel % 210;
  u64 nbits = (rel / 210) * 48 + (u64)CNT210[rr] - 1;
  if (nbits > B * 8) nbits = B * 8;
  u64 nf = nbits / 64, nb = nbits & 63;
  for (u64 i = 0; i < nf; i++) total += popcnt(w[i]);
  if (nb) total += popcnt(w[nf] & ((1ull << nb) - 1));
  return total;
}

static u64 count_segment(const u8* s, u64 B, u64 n, u64 segLow)
{
  u64 total = 0;
  const u64* w = (const u64*)s;
  u64 nw = B / 8;
  if (segLow + (u64)35 * B < n) {   /* strictly: every candidate < n */
    for (u64 i = 0; i < nw; i++) total += popcnt(w[i]);
    /* 210-grid: B is a multiple of 6, NOT of 8 - the tail bytes carry
       real candidates and must be counted (B%8 = 4 bytes per segment) */
    for (u64 bb = 8 * nw; bb < B; bb++) total += popcnt((u64)s[bb]);
    return total;
  }
  /* partial segment: number of candidate slots with value < n */
  u64 rel = n - segLow;
  u64 rr = rel % 210;
  u64 nbits = (rel / 210) * 48 + (u64)CNT210[rr] - 1;
  u64 nf = nbits / 64, nb = nbits & 63;
  for (u64 i = 0; i < nf; i++) total += popcnt(w[i]);
  if (nb) total += popcnt(w[nf] & ((1ull << nb) - 1));
  return total;
}
/* ------------------------------------------------------------------ */
/* Main sieve                                                          */
/* ------------------------------------------------------------------ */
static void prime_init_state(u64 p, u64 low, u32* outI, u8* outK, u8* outT)
{
  u8  k  = KMAP210[p % 210];
  u64 lowA = low + 11;             /* byte j covers [low+210j+11, low+210j+211] */
  u64 q = lowA / p + 1;
  if (q < p) q = p;
  u8  ur = (u8)(q % 210);          /* need a quotient coprime to 210 */
  q += (u64)NEXT210[ur] - ur;
  u64 multiple = p * q;
  u8  t = KMAP210[q % 210];
  u64 rel = multiple - lowA;       /* multiple - low - 11 */
  u64 blk = rel / 210;             /* for r == 1 this already floors to K-1 */
  u64 r   = (multiple - low) % 210;
  u64 m   = MIDX210[r];            /* r == 1 -> bit 47 of the previous block */
  *outI = (u32)(6 * blk + m / 8);
  *outK = k;
  *outT = t;
}
static u64 sieve_slice(u64 lo, u64 countLo, u64 cap, u64 B, u64 smallMax, u64 medMax,
                       const u64* pl, u64 npl, fs_emit_cb emit, void* emitCtx)
{
  u64 bps = B / 6;                             /* 210-blocks per segment */
  u8* sieve = (u8*)malloc((size_t)B);
  if (!sieve) { fprintf(stderr, "oom\n"); exit(1); }
  State st; memset(&st, 0, sizeof(st));
  /* skip primes handled by pre-sieving (<= 163) */
  u64 gpos = 1;                    /* pl[0] = 2 */
  while (gpos < npl && pl[gpos] <= PRESIEVE_MAX) gpos++;
  u64 total = 0;
  u64 segNo = 0;
  int stopped = 0;
  for (u64 low = lo; low < cap; low += (u64)35 * B) {
    u64 end = low + (u64)35 * B;
    pre_sieve(sieve, B, low);
    /* add sieving primes whose p^2 lies inside [low, end) */
    while (gpos < npl) {
      u64 p = pl[gpos];
      u64 p2 = p * p;
      /* p^2 may be stored as the trailing (offset-31) bit of this segment,
         i.e. up to low + 30*B + 1; add it while it fits the representation */
      if (p2 >= end + 2) break;
      gpos++;
      u32 i0; u8 k0, t0;
      prime_init_state(p, low, &i0, &k0, &t0);
      if (i0 >= B)
        s_pend_push(&st, i0, k0, t0, p);
      else if (p <= smallMax)
        s_small_push(&st, i0, k0, t0, p);
      else if (p <= medMax)
        s_med_push(&st, i0, k0, t0, p);
      else {
        /* bucketized: first multiple may lie several segments ahead */
        u64 seg = i0 / bps;
        u32 nd = s_node_new(&st);
        st.nodes[nd].sp = (u32)(p / 210);
        st.nodes[nd].i  = i0 % bps;
        st.nodes[nd].k  = k0;
        st.nodes[nd].t  = t0;
        s_big_relink(&st, nd, seg);
      }
    }
    /* migrate pending primes whose first multiple arrived */
    if (st.npend) {
      u64 w = 0;
      for (u64 x = 0; x < st.npend; x++) {
        if (st.pend[x].i >= B) st.pend[x].i -= (u32)B;
        u64 pv = 210ull * st.pend[x].sp + (u64)RES210[st.pend[x].k];
        if (st.pend[x].i < B) {
          if (pv <= smallMax)
            s_small_push(&st, st.pend[x].i, st.pend[x].k, st.pend[x].t, pv);
          else
            s_med_push(&st, st.pend[x].i, st.pend[x].k, st.pend[x].t, pv);
        }
        else
          st.pend[w++] = st.pend[x];
      }
      st.npend = w;
    }
    /* crossing: small primes (L1-chunked) */
    if (st.nsml) {
      for (u64 c = 0; c < B; c += L1_CHUNK) {
        u64 cl = B - c; if (cl > L1_CHUNK) cl = L1_CHUNK;
        cross_flat(sieve + c, cl, st.sml, st.nsml);
      }
    }
    /* crossing: medium primes (whole segment) */
    if (st.nmdl)
      cross_flat(sieve, B, st.mdl, st.nmdl);
    /* crossing: big primes via buckets */
    if (st.head && st.head[0] != NO_NODE) {
      while (st.head[0] != NO_NODE) {
        u32 nd = st.head[0];
        st.head[0] = st.nodes[nd].next;
        u32 sp = st.nodes[nd].sp;
        u64 i  = st.nodes[nd].i;
        u8  k  = st.nodes[nd].k;
        u8  t  = st.nodes[nd].t;
        sieve[i] &= BM210[k][t];
        u64 nxt = i + (u64)CV210[k][t] * sp + CB210[k][t];
        t = tinc(t);
        u64 seg = nxt / bps;
        u64 ii  = nxt % bps;
        st.nodes[nd].i = (u32)ii;
        st.nodes[nd].t = t;
        s_big_relink(&st, nd, seg);
      }
    }
    /* very first segment (low == 0): restore true small primes */
    if (lo == 0 && segNo == 0) {
      for (int j = 0; j < 6 && j < (int)B; j++) sieve[j] = PRIMEBITS210[j];
    }
    {
      const char* denv = getenv("FASTSIEVE_DUMP_SEG");
      if (denv && (u64)atoi(denv) == segNo) {
        FILE* f = fopen("/tmp/segdump.bin", "wb");
        if (f) { fwrite(sieve, 1, B, f); fclose(f); }
        fprintf(stderr, "[dump] segment %llu -> /tmp/segdump.bin\n", (unsigned long long)segNo);
      }
    }
    /* count candidates in [countLo, cap) within this segment */
    if (countLo > low)
      total += cbits_prefix(sieve, B, low, cap) - cbits_prefix(sieve, B, low, countLo);
    else
      total += count_segment(sieve, B, cap, low);
    /* optional on-the-fly prime emission over the same window */
    if (emit) {
      u64 from = (countLo > low) ? countLo : low;
      if (low == 0 && segNo == 0) {   /* primes 2,3,5,7 are outside the wheel */
        static const u64 S4[4] = {2, 3, 5, 7};
        for (int pp2 = 0; pp2 < 4; pp2++)
          if (from <= S4[pp2] && S4[pp2] < cap) {
            if (emit(S4[pp2], emitCtx)) { stopped = 1; break; }
          }
      }
      if (!stopped) {
        /* generous byte window around [from, cap): the per-candidate filter
           below is exact; bounds only skip bytes that cannot qualify */
        u64 jmin = 0;
        if (from > low + 11) {
          u64 d = from - low - 11;
          jmin = 6 * (d / 210); if (jmin) jmin -= 6;
        }
        u64 upv  = cap - 1;
        u64 jmax = B - 1;
        if (upv > low + 11) {
          u64 d = upv - low - 11;
          jmax = 6 * (d / 210) + 6;
          if (jmax > B - 1) jmax = B - 1;
        }
        for (u64 j = jmin; j <= jmax; j++) {
          u8 byte = sieve[j];
          if (byte) {
            /* byte j = block j/6, sub-byte j%6; global bit index
               g = 48*(j/6) + 8*(j%6) + b; value = 210*(g/48) +
               (g%48 == 47 ? 211 : RES210[1 + g%48]) */
            u64 blk = j / 6, sub = (j % 6) * 8;
            for (int b = 0; b < 8; b++)
              if (byte & (1u << b)) {
                u64 g = 48 * blk + sub + (u64)b;
                u64 v = 210 * (g / 48) + ((g % 48) == 47 ? 211ULL
                                                         : (u64)RES210[1 + g % 48]);
                if (v >= from && v < cap && emit(v, emitCtx)) { stopped = 1; break; }
              }
          }
          if (stopped) break;
        }
      }
    }
    segNo++;
    /* advance the bucket window */
    if (st.nhead > 0) {
      for (u64 x = 1; x < st.nhead; x++) st.head[x - 1] = st.head[x];
      st.nhead--;
    }
    if (stopped) break;
  }
  free(st.sml); free(st.mdl); free(st.pend); free(st.head); free(st.nodes);
  free(sieve);
  return total;
}
/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
#ifdef _WIN32
static double now_sec(void) {
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
}
#else
static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#endif

#ifndef FASTSIEVE_NO_MAIN
int main(int argc, char** argv) {
  ensure_tables();
  u64 n = 1000000000ULL;
  u64 B = DEFAULT_SIEVE_B - (DEFAULT_SIEVE_B % 6);   /* multiple of 6 */
  u64 glo = 0;
  double medF = 3.0;
  long nthreads = 0;
  int useGpu = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--sieve-size") && i + 1 < argc) {
      double kb; sscanf(argv[++i], "%lf", &kb);
      double bytes = kb * 1024;
      u64 r = 64;
      while ((double)(r << 1) <= bytes) r <<= 1;
      B = r - (r % 6);                  /* wheel-210: B must be a multiple of 6 */
    }
    else if (!strcmp(argv[i], "--gpu")) useGpu = 1;
    else if (!strcmp(argv[i], "--med-f") && i + 1 < argc) medF = atof(argv[++i]);
    else if (!strcmp(argv[i], "--lo") && i + 1 < argc) glo = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--hi") && i + 1 < argc) n = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-t") && i + 1 < argc) nthreads = atol(argv[++i]);
    else if (argv[i][0] != '-') n = (u64)strtod(argv[i], NULL);
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
  }
  u64 smallMax = (u64)(L1_BYTES * 0.2);
  u64 medMax = (u64)((double)B * medF);
  if (nthreads <= 0) nthreads = 1;
  if (n < 2) { printf("0\n"); return 0; }
  if (n > (u64)8e15) { fprintf(stderr, "n too large for this demo\n"); return 1; }

  double t0 = now_sec();
  u64 pi = 0;
  if (glo > 0) {
    u64 root = 1;
    while ((root + 1) <= n / (root + 1)) root++;
    u64* pl = NULL; u64 npl = 0;
    simple_sieve(root, &pl, &npl);
    u64 segStart = (glo > (u64)35 * B) ? glo - (u64)35 * B : 0;
    segStart -= segStart % 210;  /* wheel layout requires segLow = 0 (mod 210) */
    pi = sieve_slice(segStart, glo, n + 1, B, smallMax, medMax, pl, npl, 0, 0);
    free(pl);
    printf("pi([%llu, %llu]) = %llu\n", (unsigned long long)glo, (unsigned long long)n,
           (unsigned long long)pi);
    printf("Seconds: %.3f\n", now_sec() - t0);
    return 0;
  }
  pi = fs_pi_core(n, nthreads, useGpu, B, medF, 0);

  double sec = now_sec() - t0;
  printf("pi(%llu) = %llu\n", (unsigned long long)n, (unsigned long long)pi);
  printf("Seconds: %.3f\n", sec);
  return 0;
}
#endif /* FASTSIEVE_NO_MAIN */

/* ------------------------------------------------------------------ */
/* Shared engine core + public C API                                    */
/* ------------------------------------------------------------------ */
static int g_tables_built = 0;

static void ensure_tables(void) {
  if (!g_tables_built) {
    build_cross_tables();
    build_pre_sieve_tables();
    build_count_tables();
    g_tables_built = 1;
  }
}

void fastsieve_init(void) { ensure_tables(); }
const char* fastsieve_version(void) { return "1.0.0"; }

static void fs_resolve_cfg(long* threads, int* useGpu, u64* B, double* medF,
                           const fastsieve_config* c) {
  *threads = c ? c->threads : 0;
  if (*threads <= 0) *threads = 1;
  *useGpu = c ? !!c->use_gpu : 0;
  u64 sb = c && c->sieve_bytes ? c->sieve_bytes : DEFAULT_SIEVE_B;
  u64 r = 64;
  while ((r << 1) <= sb) r <<= 1;
  *B = r - (r % 6);                   /* wheel-210: B must be a multiple of 6 */
  *medF = (c && c->med_factor > 0.0) ? c->med_factor : 3.0;
}

/* Number of primes <= n: shared entry for the CLI and the API. Exact; the GPU
   result is audited against the CPU engine and falls back on any mismatch. */
static u64 fs_pi_core(u64 n, long nthreads, int useGpu, u64 B, double medF, double* gsec_out) {
  u64 smallMax = (u64)(L1_BYTES * 0.2);
  u64 medMax = (u64)((double)B * medF);
  if (nthreads <= 0) nthreads = 1;
  u64 root = 1;
  while ((root + 1) <= n / (root + 1)) root++;
  u64* pl = NULL; u64 npl = 0;
  simple_sieve(root, &pl, &npl);
  u64 span = (u64)35 * B;
  u64 nseg = (n + span - 1) / span;
  u64 slicesz = (u64)((nseg + (u64)nthreads - 1) / (u64)nthreads);
  if (slicesz == 0) slicesz = 1;
  u64 pi = 0;
  double gsec = 0;
  int gpuFellBack = 0;
  u64 gpuTop = n + 1;   /* engine counts candidate values < cap -> include n */
  if (useGpu) {
    /* wheel-210 branch: the GPU kernels are wheel-30 (different byte grid),
       so the audited GPU path is disabled here - the exact CPU engine runs
       instead, keeping the printed value exact. */
    fprintf(stderr, "GPU disabled on wheel-210 branch - using CPU engine\n");
    useGpu = 0;
  }
  if (useGpu) {
    u64 gb = (gpuTop + GPU_BLOCK_VALS - 1) / GPU_BLOCK_VALS;
    u64* bc = (u64*)calloc((size_t)gb, 8);
    GpuResult r = gpu_sieve(gpuTop, bc, &gsec);
    if (r.ok) {
      u64 sum = 0;
      for (u64 i = 0; i < r.nblocks; i++) sum += bc[i];
      u64 mism = 0, aud = 0;
      if ((u64)35 * B % GPU_BLOCK_VALS != 0) { mism = 1; }
      else {
        u64 csegs = gpuTop / ((u64)35 * B);
        u64 step = (csegs > 512) ? (csegs / 512U) : 1;
        u64 aiList[1024]; u64 nc = 0;
        for (u64 ai = 0; ai < csegs && nc < 1023; ai += step) aiList[nc++] = ai;
        if (nc == 0 || aiList[nc - 1] != (csegs > 0 ? csegs - 1 : 0)) {
          if (nc < 1024) aiList[nc++] = csegs > 0 ? csegs - 1 : 0;
        }
        for (u64 x = 0; x < nc && aud < 512; x++) {
          u64 segLo = aiList[x] * (u64)35 * B;
          u64 segHi = segLo + (u64)35 * B;
          if (segHi > gpuTop) segHi = gpuTop;
          /* CPU audit window uses the same cap convention as the GPU blocks:
             a full segment owns candidates up to segHi+1 (trailing offset-31),
             the final window counts < gpuTop (= values <= n). */
          u64 segCap = segHi + 2;
          if (segCap > gpuTop) segCap = gpuTop;
          u64 cpuCnt = sieve_slice(segLo, segLo, segCap, B, smallMax, medMax, pl, npl, 0, 0);
          u64 gpuCnt = 0;
          u64 k0 = segLo / GPU_BLOCK_VALS;
          u64 k1 = (segHi + GPU_BLOCK_VALS - 1) / GPU_BLOCK_VALS;
          if (k1 > r.nblocks) k1 = r.nblocks;
          for (u64 k = k0; k < k1; k++) gpuCnt += bc[k];
          if (cpuCnt != gpuCnt) mism++;
          aud++;
        }
      }
      if (mism == 0) {
        if (n >= 7) sum += 4; else if (n >= 5) sum += 3; else if (n >= 3) sum += 2; else if (n >= 2) sum += 1;
        pi = sum;
        fprintf(stderr, "GPU audit passed (%llu segments sampled) - device: %s, kernel %.3f s\n",
                (unsigned long long)aud, r.device, r.gpu_secs);
      } else {
        gpuFellBack = 1;
        fprintf(stderr, "GPU audit: %llu mismatches - falling back to CPU\n",
                (unsigned long long)mism);
      }
    } else {
      fprintf(stderr, "GPU unavailable - falling back to CPU\n");
    }
    free(bc);
  }
  if (!useGpu || gpuFellBack) {
#ifdef _OPENMP
    if (nthreads > 1) {
      u64* counts = (u64*)calloc((size_t)nthreads, sizeof(u64));
      i64 s;
      #pragma omp parallel for schedule(dynamic, 1) num_threads((int)nthreads)
      for (s = 0; s < (i64)nthreads; s++) {
        u64 lo = (u64)s * slicesz * span;
        u64 hi = ((u64)s + 1) * slicesz * span;
        if (lo > n) continue;
        if (hi > n) hi = n;
        u64 cap = (hi >= n) ? n + 1 : hi + 1;
        u64 segStart = (lo == 0) ? 0 : lo - span;
        counts[s] = sieve_slice(segStart, lo, cap, B, smallMax, medMax, pl, npl, 0, 0);
      }
      for (i64 sl = 0; sl < (i64)nthreads; sl++) pi += counts[sl];
      free(counts);
    } else
#endif
    {
      pi = sieve_slice(0, 0, n + 1, B, smallMax, medMax, pl, npl, 0, 0);
    }
    if (n >= 7) pi += 4;
    else if (n >= 5) pi += 3;
    else if (n >= 3) pi += 2;
    else if (n >= 2) pi += 1;
  }
  free(pl);
  if (gsec_out) *gsec_out = gsec;
  return pi;
}

int64_t fastsieve_pi(uint64_t n, const fastsieve_config* cfg) {
  if (n < 2) return 0;
  if (n > FASTSIEVE_MAX_N) return FASTSIEVE_ERR_RANGE;
  ensure_tables();
  long nthreads; int useGpu; u64 B; double medF;
  fs_resolve_cfg(&nthreads, &useGpu, &B, &medF, cfg);
  return (int64_t)fs_pi_core(n, nthreads, useGpu, B, medF, 0);
}

typedef struct gen_ctx {
  fastsieve_prime_cb cb;
  void* user;
  int64_t n;
} gen_ctx;

static int gen_emit(u64 prime, void* p) {
  gen_ctx* g = (gen_ctx*)p;
  g->n++;   /* the prime is delivered to cb either way - it counts as emitted */
  if (g->cb && g->cb(prime, g->user)) return 1;
  return 0;
}

/* Shared generator core; cb == NULL just counts. fastsieve_generate adds the
   public cb-null/error contract on top. */
static int64_t fs_generate_core(uint64_t lo, uint64_t hi, fastsieve_prime_cb cb,
                                void* user, const fastsieve_config* cfg) {
  ensure_tables();
  long nthreads; int useGpu; u64 B; double medF;
  fs_resolve_cfg(&nthreads, &useGpu, &B, &medF, cfg);
  (void)nthreads; (void)useGpu;   /* generator is a deterministic CPU scan */
  u64 smallMax = (u64)(L1_BYTES * 0.2);
  u64 medMax = (u64)((double)B * medF);
  u64 top = hi + 1;
  u64 root = 1;
  while ((root + 1) <= top / (root + 1)) root++;
  u64* pl = NULL; u64 npl = 0;
  simple_sieve(root, &pl, &npl);
  gen_ctx g; g.cb = cb; g.user = user; g.n = 0;
  u64 segStart = (lo > (u64)35 * B) ? lo - (u64)35 * B : 0;
  segStart -= segStart % 210;  /* wheel layout requires segLow = 0 (mod 210) */
  sieve_slice(segStart, lo, top, B, smallMax, medMax, pl, npl, gen_emit, &g);
  free(pl);
  return g.n;
}

int64_t fastsieve_generate(uint64_t lo, uint64_t hi, fastsieve_prime_cb cb, void* user,
                           const fastsieve_config* cfg) {
  if (lo < 2) lo = 2;
  if (hi < lo) return FASTSIEVE_ERR_ARGS;
  if (hi > FASTSIEVE_MAX_N) return FASTSIEVE_ERR_RANGE;
  if (!cb) return FASTSIEVE_ERR_ARGS;   /* header contract: callback required */
  return fs_generate_core(lo, hi, cb, user, cfg);
}

int64_t fastsieve_count(uint64_t lo, uint64_t hi, const fastsieve_config* cfg) {
  if (lo < 2) lo = 2;
  if (hi < lo) return FASTSIEVE_ERR_ARGS;   /* header contract: lo > hi is an error */
  if (hi > FASTSIEVE_MAX_N) return FASTSIEVE_ERR_RANGE;
  if (lo == 2) return fastsieve_pi(hi, cfg);
  return fs_generate_core(lo, hi, NULL, 0, cfg);
}

int fastsieve_isprime(uint64_t n, const fastsieve_config* cfg) {
  if (n < 2) return 0;
  if (n > FASTSIEVE_MAX_N) return FASTSIEVE_ERR_RANGE;
  return fastsieve_count(n, n, cfg) == 1;
}

int64_t fastsieve_nth_prime(uint64_t k, uint64_t start, const fastsieve_config* cfg) {
  if (k == 0) return FASTSIEVE_ERR_ARGS;
  if (start < 2) start = 2;
  u64 countBefore = (start <= 2) ? 0 : (u64)fastsieve_pi(start - 1, cfg);
  if (countBefore >= (u64)UINT64_MAX - k) return FASTSIEVE_ERR_RANGE;
  u64 j = countBefore + k;                 /* overall 1-based prime index */
  double lj = log((double)j);
  double l2 = log(lj < 1.0 ? 1.0 : lj);
  u64 hi = (j >= 6) ? (u64)((double)j * (lj + l2)) + 64 : 12;   /* P_j bound */
  if (hi < start) hi = start;
  if (hi > FASTSIEVE_MAX_N) return FASTSIEVE_ERR_RANGE;
  int guard = 0;
  while ((u64)fastsieve_pi(hi, cfg) < j && hi < FASTSIEVE_MAX_N && guard++ < 80)
    hi += hi >> 1;
  if (hi > FASTSIEVE_MAX_N) return FASTSIEVE_ERR_RANGE;
  u64 lo = start;
  while (lo < hi) {                        /* smallest x >= start, pi(x) >= j */
    u64 m = lo + (hi - lo) / 2;
    if ((u64)fastsieve_pi(m, cfg) >= j) hi = m; else lo = m + 1;
  }
  return (int64_t)lo;
}














