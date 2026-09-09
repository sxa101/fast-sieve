/* api_test.c - verification of the fastsieve C API on CPU + any GPU backend.
 * Every check is against hard-coded primesieve-derived oracle values.
 *
 * Note: do NOT call setvbuf(_IOLBF) here - the MSVC VCRT fail-fasts (0xC0000409)
 * on it when stdout is redirected. Console tools can live without it. */
#include <stdio.h>
#include <stdlib.h>
#include "fastsieve.h"

static int fails = 0, checks = 0;
static void check(const char* what, long long got, long long want) {
  checks++;
  if (got == want) printf("PASS %-46s %lld\n", what, got);
  else { fails++; printf("FAIL %-46s got %lld want %lld\n", what, got, want); }
}

typedef struct { uint64_t last; int64_t n; int ascending; int stop_after; } gen_state;
static int gen_collect(uint64_t p, void* ud) {
  gen_state* s = (gen_state*)ud;
  if (s->n > 0 && p <= s->last) s->ascending = 0;
  s->last = p; s->n++;
  if (s->stop_after > 0 && s->n >= s->stop_after) return 1;
  return 0;
}

int main(void) {
  fastsieve_init();
  printf("version: %s\n\n", fastsieve_version());

  fastsieve_config gpu = {0};
  gpu.threads = 12; gpu.use_gpu = 1;
  fastsieve_config cpu12 = {0};
  cpu12.threads = 12;

  /* ---- pi: GPU path across the sweep ---- */
  check("pi(100) gpu",            fastsieve_pi(100, &gpu), 25);
  check("pi(1000) gpu",           fastsieve_pi(1000, &gpu), 168);
  check("pi(1e4) gpu",            fastsieve_pi(10000, &gpu), 1229);
  check("pi(1e5) gpu",            fastsieve_pi(100000, &gpu), 9592);
  check("pi(1e6) gpu",            fastsieve_pi(1000000, &gpu), 78498);
  check("pi(1e7) gpu",            fastsieve_pi(10000000, &gpu), 664579);
  check("pi(1e8) gpu",            fastsieve_pi(100000000, &gpu), 5761455);
  check("pi(1e9) gpu",            fastsieve_pi(1000000000, &gpu), 50847534);
  check("pi(1e10) gpu",           fastsieve_pi(10000000000ULL, &gpu), 455052511);
  check("pi(1e11) gpu",           fastsieve_pi(100000000000ULL, &gpu), 4118054813LL);
  check("pi(1e12) gpu",           fastsieve_pi(1000000000000ULL, &gpu), 37607912018LL);

  /* ---- pi: prime upper bounds (the n+1 boundary contract) ---- */
  check("pi(0) gpu",              fastsieve_pi(0, &gpu), 0);
  check("pi(2) gpu",              fastsieve_pi(2, &gpu), 1);
  check("pi(12) gpu",             fastsieve_pi(12, &gpu), 5);
  check("pi(13) gpu",             fastsieve_pi(13, &gpu), 6);
  check("pi(999999999988) gpu",   fastsieve_pi(999999999988ULL, &gpu), 37607912017LL);
  check("pi(999999999989) gpu",   fastsieve_pi(999999999989ULL, &gpu), 37607912018LL);

  /* ---- pi: boundary squares 786431^2 / 786433^2 ---- */
  check("pi(618473717761) gpu",   fastsieve_pi(618473717761ULL, &gpu), 23688293324LL);
  check("pi(618476863489) gpu",   fastsieve_pi(618476863489ULL, &gpu), 23688409284LL);

  /* ---- pi: CPU paths (same oracle) ---- */
  check("pi(1e9) cpu t1",         fastsieve_pi(1000000000, NULL), 50847534);
  check("pi(1e10) cpu 12t",       fastsieve_pi(10000000000ULL, &cpu12), 455052511);
  check("pi(618476863489) 12t",   fastsieve_pi(618476863489ULL, &cpu12), 23688409284LL);

  /* ---- count ---- */
  check("count(2,1e9) == pi gpu", fastsieve_count(2, 1000000000, &gpu), 50847534);
  check("count(1e9,1e9+1e4) gpu", fastsieve_count(1000000000ULL, 1000010000ULL, &gpu), 487);
  check("count(1e9,1e9+1e4) cpu", fastsieve_count(1000000000ULL, 1000010000ULL, NULL), 487);
  check("count(786431,786433)",   fastsieve_count(786431, 786433, &gpu), 2);
  check("count(13,13)",           fastsieve_count(13, 13, &gpu), 1);
  check("count(14,16)",           fastsieve_count(14, 16, &gpu), 0);
  check("count(p^2,p^2) gpu",     fastsieve_count(618473717761ULL, 618473717761ULL, &gpu), 0);
  /* regression windows the full-range pi sweep cannot see:
     * pend migration bug made counts wrong above ~2e12;
     * countLo/cap on a segment edge dropped the phantom candidate
       (repro [9174900,9274900]; 7864320 = 30*262144 is the span). */
  check("count(2e12,2e12+2e8)",   fastsieve_count(2000000000000ULL, 2000200000000ULL, NULL), 7061729);
  check("count(1e13,1e13+1e7)",   fastsieve_count(10000000000000ULL, 10000010000000ULL, NULL), 334312);
  check("count(9174900,9274900)", fastsieve_count(9174900, 9274900, NULL), 6287);
  check("count(7864320,8864320)", fastsieve_count(7864320, 8864320, NULL), 62736);
  check("count(7864320,7864321)", fastsieve_count(7864320, 7864321, NULL), 0);

  /* ---- isprime ---- */
  check("isprime(2)",             fastsieve_isprime(2, NULL), 1);
  check("isprime(3)",             fastsieve_isprime(3, NULL), 1);
  check("isprime(4)",             fastsieve_isprime(4, NULL), 0);
  check("isprime(997)",           fastsieve_isprime(997, NULL), 1);
  check("isprime(786431)",        fastsieve_isprime(786431, NULL), 1);
  check("isprime(786433)",        fastsieve_isprime(786433, NULL), 1);
  check("isprime(491521) comp",   fastsieve_isprime(491521, NULL), 0);   /* 17*28913 */
  check("isprime(4423681)",       fastsieve_isprime(4423681, NULL), 1);
  check("isprime(618473717761)",  fastsieve_isprime(618473717761ULL, &gpu), 0);
  check("isprime(618476863489)",  fastsieve_isprime(618476863489ULL, &gpu), 0);

  /* ---- nth_prime ---- */
  check("nth(1,1)",               fastsieve_nth_prime(1, 1, NULL), 2);
  check("nth(5,1)",               fastsieve_nth_prime(5, 1, NULL), 11);
  check("nth(3,10)",              fastsieve_nth_prime(3, 10, NULL), 17);
  check("nth(1000,1)",            fastsieve_nth_prime(1000, 1, NULL), 7919);
  check("nth(10000,1)",           fastsieve_nth_prime(10000, 1, NULL), 104729);
  check("nth(1000000,1)",         fastsieve_nth_prime(1000000, 1, NULL), 15485863);
  check("nth(1000000,2e6)",       fastsieve_nth_prime(1000000, 2000000, NULL), 17959859);
  /* consistency: p = nth(1, S) must be the first prime >= S. Small S only:
     the nth_prime bracketing does ~40 full pi() calls, each GPU-audited. */
  { int64_t p = fastsieve_nth_prime(1, 1000000000, &gpu);
    check("nth(1,1e9) == 1e9+7", p, 1000000007);
  }
  { int64_t p = fastsieve_nth_prime(1, 491521, NULL);
    check("nth(1,491521) == 491527", p, 491527);
  }

  /* ---- generate ---- */
  { gen_state s = {0, 0, 1, 0};
    int64_t n = fastsieve_generate(10, 30, gen_collect, &s, NULL);
    check("generate(10,30) count", n, 6);
    check("generate(10,30) last",  (long long)s.last, 29);
    check("generate(10,30) asc",   s.ascending, 1);
  }
  { gen_state s = {0, 0, 1, 0};
    int64_t n = fastsieve_generate(2, 30, gen_collect, &s, NULL);
    check("generate(2,30) count", n, 10);
    check("generate(2,30) last", (long long)s.last, 29);
  }
  { gen_state s = {0, 0, 1, 4};   /* early stop after 4 primes */
    int64_t n = fastsieve_generate(2, 1000000, gen_collect, &s, NULL);
    check("generate early-stop", n, 4);
  }
  { gen_state s = {0, 0, 1, 0};
    int64_t n = fastsieve_generate(1000000000ULL, 1000010000ULL, gen_collect, &s, NULL);
    check("generate(1e9,+1e4) count", n, 487);
    check("generate(1e9,+1e4) asc", s.ascending, 1);
  }

  /* ---- error paths ---- */
  check("pi(MAX_N+1) -> ERR_RANGE", fastsieve_pi(FASTSIEVE_MAX_N + 1, NULL), FASTSIEVE_ERR_RANGE);
  check("count(2,MAX_N+1) -> ERR_RANGE", fastsieve_count(2, FASTSIEVE_MAX_N + 1, NULL), FASTSIEVE_ERR_RANGE);
  check("count(5,2) -> ERR_ARGS", fastsieve_count(5, 2, NULL), FASTSIEVE_ERR_ARGS);
  check("nth(0,1) -> ERR_ARGS",   fastsieve_nth_prime(0, 1, NULL), FASTSIEVE_ERR_ARGS);
  check("generate(2,100,NULL) -> ERR_ARGS", fastsieve_generate(2, 100, NULL, NULL, NULL), FASTSIEVE_ERR_ARGS);
  check("generate(5,2) -> ERR_ARGS", fastsieve_generate(5, 2, gen_collect, NULL, NULL), FASTSIEVE_ERR_ARGS);

  printf("\n----------------------------------------\n");
  printf("checks=%d fails=%d -> %s\n", checks, fails, fails ? "FAILED" : "ALL PASS");
  return fails ? 1 : 0;
}
