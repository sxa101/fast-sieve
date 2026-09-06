/* api_demo.c - minimal example of the fastsieve C API.

   Build (Windows, MSVC):  cl /O2 /arch:AVX2 /Oi /openmp fastsieve.c gpu.c examples/api_demo.c
   Build (Linux, no GPU):  gcc -O3 -march=native -fopenmp fastsieve.c examples/api_demo.c
*/
#include <stdio.h>
#include "../fastsieve.h"

static int print_prime(uint64_t p, void* user) {
  (void)user;
  printf("%llu", (unsigned long long)p);
  return 0;   /* return nonzero to stop early */
}

int main(void) {
  fastsieve_init();
  printf("fastsieve version: %s\n", fastsieve_version());
  printf("pi(1e6)                 = %lld\n", (long long)fastsieve_pi(1000000ULL, NULL));

  fastsieve_config cfg = {0};
  cfg.threads = 12;
  cfg.use_gpu = 1;
  printf("pi(1e9)   (12t + GPU)   = %lld\n", (long long)fastsieve_pi(1000000000ULL, &cfg));

  printf("isprime(97)             = %d\n", fastsieve_isprime(97ULL, NULL));
  printf("isprime(98)             = %d\n", fastsieve_isprime(98ULL, NULL));
  printf("count([1e9, 1e9+1e4])   = %lld\n",
         (long long)fastsieve_count(1000000000ULL, 1000010000ULL, NULL));

  long long p5  = fastsieve_nth_prime(5ULL,  1ULL, NULL);
  long long p10 = fastsieve_nth_prime(10ULL, 1ULL, NULL);
  long long p12 = fastsieve_nth_prime(3ULL,  10ULL, NULL);
  printf("5th prime               = %lld\n", p5);
  printf("10th prime              = %lld\n", p10);
  printf("3rd prime >= 10         = %lld\n", p12);

  printf("primes in [10, 30]      = ");
  fastsieve_generate(10ULL, 30ULL, print_prime, NULL, NULL);
  printf("\n");
  return 0;
}