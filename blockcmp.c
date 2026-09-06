/* blockcmp.c - compare /tmp/gpublock.bin (from FASTSIEVE_DUMP_BLOCK) vs naive CPU */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int OFF[8] = {7,11,13,17,19,23,29,31};

int main(int argc, char** argv) {
  unsigned long long block = (argc > 1) ? strtoull(argv[1], NULL, 10) : 8ULL;
  unsigned long long segLow = 30ull * 16384 * block;
  unsigned long long segEnd = segLow + 30ull * 16384;

  /* naive reference bits over candidates [segLow+7, segEnd+1] */
  unsigned long long root = 1;
  while ((root + 1) * (root + 1) <= segEnd + 1) root++;
  char* comp = (char*)calloc((size_t)root + 2, 1);
  unsigned int np = 0;
  unsigned int* primes = (unsigned int*)malloc(sizeof(unsigned int) * (root + 16));
  for (unsigned int i = 2; i <= root; i++) {
    if (!comp[i]) { primes[np++] = i; for (unsigned long long j = (unsigned long long)i * i; j <= root; j += i) comp[j] = 1; }
  }
  free(comp);
  char* isComp = (char*)calloc((size_t)(segEnd + 2), 1);
  for (unsigned int i = 0; i < np; i++) {
    unsigned long long p = primes[i];
    if (p < 7) continue;
    for (unsigned long long v = p * p; v <= segEnd + 1; v += p) isComp[v] = 1;
  }

  FILE* f = fopen("/tmp/gpublock.bin", "rb");
  if (!f) { fprintf(stderr, "no /tmp/gpublock.bin\n"); return 1; }
  unsigned char seg[16384];
  if (fread(seg, 1, 16384, f) != 16384) { fprintf(stderr, "short read\n"); return 1; }
  fclose(f);

  int diffs = 0, cpuCnt = 0, gpuCnt = 0;
  for (unsigned int j = 0; j < 16384; j++) {
    for (int b = 0; b < 8; b++) {
      unsigned long long v = segLow + 30ull * j + (unsigned)OFF[b];
      unsigned char r = (unsigned char)(v % 30);
      int unit = (r==1||r==7||r==11||r==13||r==17||r==19||r==23||r==29);
      if (!unit || v > segEnd + 1) continue;
      int gpuBit = (seg[j] >> b) & 1;
      int cpuBit = isComp[v] ? 0 : 1;
      cpuCnt += cpuBit; gpuCnt += gpuBit;
      if (gpuBit != cpuBit) {
        printf("v=%llu r=%u byte=%u bit=%d gpu=%d cpu=%d %s\n", v, r, j, b,
               gpuBit, cpuBit, cpuBit ? "(prime cleared by GPU)" : "(composite kept by GPU)");
        diffs++;
      }
    }
  }
  printf("block %llu: diffs=%d cpuNaive=%d gpuBits=%d\n", block, diffs, cpuCnt, gpuCnt);
  return 0;
}
