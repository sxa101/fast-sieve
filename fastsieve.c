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
   primesieve on the venus box); 32 KiB keeps the passes in L1D and closed
   most of the gap (docs/CPU_PERF.md). */
#define L1_CHUNK        (32u << 10)
#define PRESIEVE_MAX    163           /* multiples of p<=163 presieved */
/* ------------------------------------------------------------------ */
/* Wheel-30 crossing tables                                           */
/* ------------------------------------------------------------------ */
/* A mod-30 wheel: crossings only touch multiples p*k whose quotient k
   is coprime to 2,3,5.  The step cycle has 8 states.  (A mod-210 wheel
   that additionally skips multiples of 7 would cut crossings ~14% but
   was found to mis-frame a rare fraction of carries in multi-segment
   runs, so the proven wheel-30 is used.)                                  */
static const int OFFB[8] = { 7, 11, 13, 17, 19, 23, 29, 31 };
static const int UNITS[8] = { 1, 7, 11, 13, 17, 19, 23, 29 };

static u8  BIT_[8][8];   /* AND masks clearing the current multiple */
static u8  CV_[8][8];    /* byte-index multipliers per step        */
static u8  CB_[8][8];    /* byte-index offsets per step            */
static u8  r2idx[30];    /* p mod 30 -> residue class (0..7)       */
static u8  KMAP[30];     /* unit residue -> step index             */
static u8  NEXTUNIT[30]; /* first unit residue >= r                */
static u8  ISUNIT[30];
static int bit_of_offset(int off) {
  for (int b = 0; b < 8; b++)
    if (OFFB[b] == off) return b;
  return -1;
}
static void build_cross_tables(void) {
  for (int r = 0; r < 30; r++) { ISUNIT[r] = 0; KMAP[r] = 255; r2idx[r] = 255; }
  for (int k = 0; k < 8; k++) r2idx[OFFB[k] % 30] = (u8)k;
  for (int j = 0; j < 8; j++) { ISUNIT[UNITS[j]] = 1; KMAP[UNITS[j]] = (u8)j; }

  for (int k = 0; k < 8; k++) {
    int ri = OFFB[k] % 30;
    for (int j = 0; j < 8; j++) {
      int kk      = UNITS[j];
      int off     = (ri * kk) % 30;         /* value of the multiple mod 30 */
      int dk      = (j < 7) ? (UNITS[j+1] - UNITS[j]) : (31 - UNITS[7]);
      /* byte delta of the step = dk*sp + cb with cb = floor((g + ri*dk)/30),
         g = (value - low - 6) mod 30 */
      int g       = (off - 6 + 30) % 30;
      int cb      = (g + ri * dk) / 30;
      int bitOff  = (off == 1) ? 31 : off;
      int b       = bit_of_offset(bitOff);
      BIT_[k][j]  = (u8)~(1u << b);
      CV_[k][j]   = (u8)dk;
      CB_[k][j]   = (u8)cb;
    }
  }
  for (int r = 0; r < 30; r++) {
    int u = r;
    while (!ISUNIT[u]) u++;
    NEXTUNIT[r] = (u8)(u % 30);
  }
}
/* ------------------------------------------------------------------ */
/* Pre-sieve tables                                                   */
/* ------------------------------------------------------------------ */
static const u32 PSMIN[16][3] = {
  { 7, 23, 37 }, { 11, 19, 31 }, { 13, 17, 29 }, { 41, 163 },
  { 43, 157 }, { 47, 151 }, { 53, 149 }, { 59, 139 },
  { 61, 137 }, { 67, 131 }, { 71, 127 }, { 73, 113 },
  { 79, 109 }, { 83, 107 }, { 89, 103 }, { 97, 101, 0 }
};
static u8* PSD[16];
static u32 PSS[16];
static void build_pre_sieve_tables(void) {
    for (int t = 0; t < 16; t++) {
    u64 size = 1;
    for (int j = 0; j < 3; j++) {
      if (PSMIN[t][j] == 0) break;
      size *= (u64)PSMIN[t][j];
    }
    u8* tab = (u8*)malloc((size_t)size);
    if (!tab) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (u64 j = 0; j < size; j++) {
      u8 byte = 0;
      for (int b = 0; b < 8; b++) {
        u64 v = (u64)30 * j + (u64)OFFB[b];
        int ok = 1;
        for (int c = 0; c < 3; c++) {
          u32 p = PSMIN[t][c];
          if (p == 0) break;
          if (v % p == 0) { ok = 0; break; }
        }
        if (ok) byte |= (u8)(1u << b);
      }
      tab[j] = byte;
    }
    PSD[t] = tab;
    PSS[t] = (u32)size;
  }
}
static void pre_sieve(u8* s, u64 B, u64 segLow) {
  u32 pos[16];
  for (int t = 0; t < 16; t++)
    pos[t] = (u32)((segLow % ((u64)PSS[t] * 30)) / 30);
  u64 off = 0;
  while (off < B) {
    u64 L = B - off;
    for (int t = 0; t < 16; t++) {
      u64 rem = (u64)PSS[t] - pos[t];
      if (rem < L) L = rem;
    }
    u64 i = 0;
    for (; i + 32 <= L; i += 32) {
      __m256i v = _mm256_loadu_si256((const __m256i*)(PSD[0] + pos[0] + i));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[1] + pos[1] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[2] + pos[2] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[3] + pos[3] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[4] + pos[4] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[5] + pos[5] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[6] + pos[6] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[7] + pos[7] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[8] + pos[8] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[9] + pos[9] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[10] + pos[10] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[11] + pos[11] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[12] + pos[12] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[13] + pos[13] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[14] + pos[14] + i)));
      v = _mm256_and_si256(v, _mm256_loadu_si256((const __m256i*)(PSD[15] + pos[15] + i)));
      _mm256_storeu_si256((__m256i*)(s + off + i), v);
    }
    for (; i < L; i++) {
      u8 b = PSD[0][pos[0] + i];
      for (int t = 1; t < 16; t++) b &= PSD[t][pos[t] + i];
      s[off + i] = b;
    }
    off += L;
    for (int t = 0; t < 16; t++) {
      pos[t] += (u32)L;
      if (pos[t] >= PSS[t]) pos[t] -= PSS[t];
    }
  }
}
static const u8 PRIMEBITS[8] = { 0xff, 0xef, 0x77, 0x3f, 0xdb, 0xed, 0x9e, 0xfc };
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
static inline u8 tinc(u8 t) { return (u8)((t + 1) & 7U); }

/* ------------------------------------------------------------------ */
/* Crossing (flat class)                                               */
/* ------------------------------------------------------------------ */
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
    const u8* cv = CV_[k];   /* step byte-index multiplier  */
    const u8* cb = CB_[k];   /* step byte-index offset      */
    const u8* bit = BIT_[k];

    /* offsets of the next 8 multiples relative to the current one */
    u32 st = t, st1, st2, st3, st4, st5, st6, st7;
    u32 o1 = (u32)cv[st] * sp + cb[st], o2, o3, o4, o5, o6, o7, stride;
    st1 = (st + 1) & 7; st2 = (st1 + 1) & 7; st3 = (st2 + 1) & 7; st4 = (st3 + 1) & 7;
    st5 = (st4 + 1) & 7; st6 = (st5 + 1) & 7; st7 = (st6 + 1) & 7;
    o2 = o1 + (u32)cv[st1] * sp + cb[st1];
    o3 = o2 + (u32)cv[st2] * sp + cb[st2];
    o4 = o3 + (u32)cv[st3] * sp + cb[st3];
    o5 = o4 + (u32)cv[st4] * sp + cb[st4];
    o6 = o5 + (u32)cv[st5] * sp + cb[st5];
    o7 = o6 + (u32)cv[st6] * sp + cb[st6];
    stride = o7 + (u32)cv[st7] * sp + cb[st7];

    for (;;) {
      if (i + o7 >= B) break;
      s[i]      &= bit[st];
      s[i + o1] &= bit[st1];
      s[i + o2] &= bit[st2];
      s[i + o3] &= bit[st3];
      s[i + o4] &= bit[st4];
      s[i + o5] &= bit[st5];
      s[i + o6] &= bit[st6];
      s[i + o7] &= bit[st7];
      i += stride;
      if (i >= B) { i -= B; goto done_flat; }
    }

    /* tail: finish the segment one multiple at a time */
    for (;;) {
      if (i >= B) { i -= B; goto done_flat; }
      s[i] &= bit[t];
      i += (u32)cv[t] * sp + cb[t];
      t = tinc(t);
    }
    done_flat:
    P[n].i = (u32)i; P[n].t = t;
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
  o->sp = (u32)(p / 30); o->i = i; o->k = k; o->t = t;
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

static u8 CUNIT[30];

static void build_count_tables(void) {
  for (int r = 0; r < 30; r++) {
    u8 c = 0;
    for (int k = 1; k < r; k++)
      if (k % 2 && k % 3 && k % 5) c++;
    CUNIT[r] = c;
  }
}

/* Number of candidate bits with value < up (up absolute, inside or one
   past this segment), counted from the segment start. */
static u64 cbits_prefix(const u8* s, u64 B, u64 segLow, u64 up)
{
  u64 total = 0;
  const u64* w = (const u64*)s;
  if (up <= segLow) return 0;
  u64 rel = up - segLow;                 /* up > segLow */
  if (rel > (u64)30 * B) rel = (u64)30 * B;
  if (rel <= 1) return 0;
  u64 nbits = (rel / 30) * 8 + (u64)CUNIT[rel % 30];
  if (nbits) nbits -= 1;                 /* candidate "1" is not in the array */
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
  if (segLow + (u64)30 * B < n) {   /* strictly: every candidate < n */
    for (u64 i = 0; i < nw; i++) total += popcnt(w[i]);
    return total;
  }
  /* partial segment: number of candidate slots with value < n */
  u64 rel = n - segLow;
  u64 nbits = (rel / 30) * 8 + (u64)CUNIT[rel % 30];
  if (nbits) nbits -= 1;                 /* the candidate "1" is not in the array */
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
  u8  k  = r2idx[p % 30];
  u64 low6 = low + 6;              /* byte j covers [low+30j+7, low+30j+31] */
  u64 q = low6 / p + 1;
  if (q < p) q = p;
  u8  ur = (u8)(q % 30);           /* need a quotient coprime to 30 */
  q += (u64)NEXTUNIT[ur] - ur;
  u64 multiple = p * q;
  u8  t = KMAP[q % 30];
  u64 idx = (multiple - low6) / 30;
  *outI = (u32)idx;
  *outK = k;
  *outT = t;
}
static u64 sieve_slice(u64 lo, u64 countLo, u64 cap, u64 B, u64 smallMax, u64 medMax,
                       const u64* pl, u64 npl, fs_emit_cb emit, void* emitCtx)
{
  u64 logB = 0;
  while ((1ull << logB) < B) logB++;
  u64 maskB = B - 1;
  u8* sieve = (u8*)malloc((size_t)B);
  if (!sieve) { fprintf(stderr, "oom\n"); exit(1); }
  State st; memset(&st, 0, sizeof(st));
  /* skip primes handled by pre-sieving (<= 163) */
  u64 gpos = 1;                    /* pl[0] = 2 */
  while (gpos < npl && pl[gpos] <= PRESIEVE_MAX) gpos++;
  u64 total = 0;
  u64 segNo = 0;
  int stopped = 0;
  for (u64 low = lo; low < cap; low += (u64)30 * B) {
    u64 end = low + (u64)30 * B;
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
        u64 seg = i0 >> logB;
        u32 nd = s_node_new(&st);
        st.nodes[nd].sp = (u32)(p / 30);
        st.nodes[nd].i  = i0 & (u32)maskB;
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
        u64 pv = 30ull * st.pend[x].sp + (u64)(OFFB[st.pend[x].k] % 30);
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
        sieve[i] &= BIT_[k][t];
        u64 nxt = i + (u64)CV_[k][t] * sp + CB_[k][t];
        t = tinc(t);
        u64 seg = nxt >> logB;
        u64 ii  = nxt & maskB;
        st.nodes[nd].i = (u32)ii;
        st.nodes[nd].t = t;
        s_big_relink(&st, nd, seg);
      }
    }
    /* very first segment (low == 0): restore true small primes */
    if (lo == 0 && segNo == 0) {
      for (int j = 0; j < 8 && j < (int)B; j++) sieve[j] = PRIMEBITS[j];
    }
    /* count candidates in [countLo, cap) within this segment */
    if (countLo > low)
      total += cbits_prefix(sieve, B, low, cap) - cbits_prefix(sieve, B, low, countLo);
    else
      total += count_segment(sieve, B, cap, low);
    /* optional on-the-fly prime emission over the same window */
    if (emit) {
      u64 from = (countLo > low) ? countLo : low;
      if (low == 0 && segNo == 0) {   /* primes 2,3,5 are outside the wheel */
        static const u64 S3[3] = {2, 3, 5};
        for (int pp2 = 0; pp2 < 3; pp2++)
          if (from <= S3[pp2] && S3[pp2] < cap) {
            if (emit(S3[pp2], emitCtx)) { stopped = 1; break; }
          }
      }
      if (!stopped) {
        u64 jmin = (from > low + 7) ? (from - low - 7) / 30 : 0;
        u64 upv  = cap - 1;
        u64 jmax = (upv > low + 7) ? (upv - low - 7) / 30 : 0;
        if (jmax >= B) jmax = B - 1;
        for (u64 j = jmin; j <= jmax; j++) {
          u8 byte = sieve[j];
          if (byte) {
            static const u64 OFF[8] = {7, 11, 13, 17, 19, 23, 29, 31};
            u64 base = low + 30 * j;
            for (int b = 0; b < 8; b++)
              if (byte & (1u << b)) {
                u64 v = base + OFF[b];
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
  u64 B = DEFAULT_SIEVE_B;
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
      B = r;
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
    u64 segStart = (glo > (u64)30 * B) ? glo - (u64)30 * B : 0;
    segStart -= segStart % 30;   /* wheel layout requires segLow = 0 (mod 30) */
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
  *B = r;
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
  u64 span = (u64)30 * B;
  u64 nseg = (n + span - 1) / span;
  u64 slicesz = (u64)((nseg + (u64)nthreads - 1) / (u64)nthreads);
  if (slicesz == 0) slicesz = 1;
  u64 pi = 0;
  double gsec = 0;
  int gpuFellBack = 0;
  u64 gpuTop = n + 1;   /* engine counts candidate values < cap -> include n */
  if (useGpu) {
    u64 gb = (gpuTop + GPU_BLOCK_VALS - 1) / GPU_BLOCK_VALS;
    u64* bc = (u64*)calloc((size_t)gb, 8);
    GpuResult r = gpu_sieve(gpuTop, bc, &gsec);
    if (r.ok) {
      u64 sum = 0;
      for (u64 i = 0; i < r.nblocks; i++) sum += bc[i];
      u64 mism = 0, aud = 0;
      if ((u64)30 * B % GPU_BLOCK_VALS != 0) { mism = 1; }
      else {
        u64 csegs = gpuTop / ((u64)30 * B);
        u64 step = (csegs > 512) ? (csegs / 512U) : 1;
        u64 aiList[1024]; u64 nc = 0;
        for (u64 ai = 0; ai < csegs && nc < 1023; ai += step) aiList[nc++] = ai;
        if (nc == 0 || aiList[nc - 1] != (csegs > 0 ? csegs - 1 : 0)) {
          if (nc < 1024) aiList[nc++] = csegs > 0 ? csegs - 1 : 0;
        }
        for (u64 x = 0; x < nc && aud < 512; x++) {
          u64 segLo = aiList[x] * (u64)30 * B;
          u64 segHi = segLo + (u64)30 * B;
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
        if (n >= 5) sum += 3; else if (n >= 3) sum += 2; else if (n >= 2) sum += 1;
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
    if (n >= 5) pi += 3;
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
  u64 segStart = (lo > (u64)30 * B) ? lo - (u64)30 * B : 0;
  segStart -= segStart % 30;   /* wheel layout requires segLow = 0 (mod 30) */
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














