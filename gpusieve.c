/* gpusieve.c - OpenCL segmented-sieve demo for fastsieve (AMD/OpenCL).
   Prototype: block per sub-segment, 256-thread local sieve, popcount count.
   Wired to fastsieve via the --gpu flag (see gpu.c). */
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define CLAPI __stdcall
typedef cl_int (CLAPI *pfnGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CLAPI *pfnGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (CLAPI *pfnGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CLAPI *pfnGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_context (CLAPI *pfnCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void (CLAPI*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (CLAPI *pfnCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_command_queue (CLAPI *pfnCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_queue_properties*, cl_int*);
typedef cl_program (CLAPI *pfnCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CLAPI *pfnBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (CLAPI*)(cl_program, void*), void*);
typedef cl_int (CLAPI *pfnGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_kernel (CLAPI *pfnCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (CLAPI *pfnCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (CLAPI *pfnSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (CLAPI *pfnEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CLAPI *pfnEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CLAPI *pfnEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CLAPI *pfnFinish)(cl_command_queue);
typedef cl_int (CLAPI *pfnReleaseMemObject)(cl_mem);
typedef cl_int (CLAPI *pfnReleaseKernel)(cl_kernel);
typedef cl_int (CLAPI *pfnReleaseProgram)(cl_program);
typedef cl_int (CLAPI *pfnReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CLAPI *pfnReleaseContext)(cl_context);

static pfnGetPlatformIDs GetPlatformIDs; static pfnGetPlatformInfo GetPlatformInfo;
static pfnGetDeviceIDs GetDeviceIDs; static pfnGetDeviceInfo GetDeviceInfo;
static pfnCreateContext CreateContext;
static pfnCreateCommandQueueWithProperties CreateCommandQueueWithProperties;
static pfnCreateCommandQueue CreateCommandQueue;
static pfnCreateProgramWithSource CreateProgramWithSource; static pfnBuildProgram BuildProgram;
static pfnGetProgramBuildInfo GetProgramBuildInfo; static pfnCreateKernel CreateKernel;
static pfnCreateBuffer CreateBuffer; static pfnSetKernelArg SetKernelArg;
static pfnEnqueueNDRangeKernel EnqueueNDRangeKernel; static pfnEnqueueWriteBuffer EnqueueWriteBuffer;
static pfnEnqueueReadBuffer EnqueueReadBuffer; static pfnFinish Finish;
static pfnReleaseMemObject ReleaseMemObject; static pfnReleaseKernel ReleaseKernel;
static pfnReleaseProgram ReleaseProgram; static pfnReleaseCommandQueue ReleaseCommandQueue;
static pfnReleaseContext ReleaseContext;

static void* api(const char* n){ return (void*)GetProcAddress(LoadLibraryA("OpenCL.dll"), n); }
#define LOAD(nm) nm = (pfn##nm) api("cl" #nm)
static int load_opencl(void){
  LOAD(GetPlatformIDs); LOAD(GetPlatformInfo); LOAD(GetDeviceIDs); LOAD(GetDeviceInfo);
  LOAD(CreateContext); LOAD(CreateCommandQueueWithProperties); LOAD(CreateCommandQueue);
  LOAD(CreateProgramWithSource); LOAD(BuildProgram); LOAD(GetProgramBuildInfo);
  LOAD(CreateKernel); LOAD(CreateBuffer); LOAD(SetKernelArg);
  LOAD(EnqueueNDRangeKernel); LOAD(EnqueueWriteBuffer); LOAD(EnqueueReadBuffer); LOAD(Finish);
  LOAD(ReleaseMemObject); LOAD(ReleaseKernel); LOAD(ReleaseProgram);
  LOAD(ReleaseCommandQueue); LOAD(ReleaseContext);
  return (GetPlatformIDs && GetDeviceIDs && CreateContext && CreateProgramWithSource &&
          BuildProgram && CreateKernel && EnqueueNDRangeKernel && EnqueueWriteBuffer &&
          EnqueueReadBuffer && SetKernelArg && CreateBuffer);
}

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
static void host_cross(u8* seg, const u64* pl, u64 np, u64 segLow, u64 segEnd){
  static const u8 mask[30] = {0,0,0,0,0,0,0,0x01,0,0,0,0x02,0,0,0x04,0,0,0x08,0,0x10,0,0,0,0x20,0,0,0,0,0x40,0};
  memset(seg, 0xFF, 16384);
  for (u64 pi = 0; pi < np; pi++){
    u64 p = pl[pi];
    if (p < 7) continue;
    u64 pp = p * p;
    if (pp >= segEnd) break;
    u64 v0 = (pp > segLow + 7) ? pp : segLow + 7;
    u64 q0 = (v0 + p - 1) / p;
    u64 qmax = segEnd / p;
    for (u64 q = q0; q <= qmax; q++){
      u64 v = p * q;
      if (v < segLow + 7 || v >= segEnd) continue;
      u8 r = (u8)(v % 30);
      if (mask[r]){
        u64 idx = (v - segLow) / 30;
        if (idx < 16384) seg[idx] &= ~mask[r];
      }
    }
  }
}
static u64 count_candidates(const u8* seg, u64 segLow, u64 cap){
  u64 rel = cap - segLow;
  if (rel > 30ULL * 16384) rel = 30ULL * 16384;
  u64 nbits = 0;
  if (rel > 1){
    static const int cup[30] = {0,0,0,0,0,0,0,1,0,0,0,2,0,0,3,0,0,4,0,5,0,0,0,6,0,0,0,0,7,0};
    nbits = (rel/30)*8 + (u64)cup[rel%30];
    nbits -= 1;
  }
  if (nbits > 16384ULL*8) nbits = 16384ULL*8;
  u64 nf = nbits/64, nb = nbits&63, c = 0;
  const u64* w = (const u64*)seg;
  for (u64 i = 0; i < nf; i++) c += __popcnt64(w[i]);
  if (nb) c += __popcnt64(w[nf] & ((1ull<<nb)-1));
  return c;
}
static u64* simple_primes(u64 limit, u64* cnt){
  u64 words = (limit/2 + 2 + 63) / 64;
  u64* bits = (u64*)calloc((size_t)words, 8);
  u64* out; u64 c = 0;
  for (u64 i = 1; 2*i+1 <= limit; i++){
    u64 odd = 2*i+1;
    if (!(bits[i>>6] & (1ull<<(i&63))))
      if (odd*odd <= limit)
        for (u64 j = odd*odd; j <= limit; j += 2*odd)
          bits[(j/2)>>6] |= 1ull << ((j/2)&63);
  }
  for (u64 i = 1; 2*i+1 <= limit; i++) if (!(bits[i>>6]&(1ull<<(i&63)))) c++;
  c++;
  out = (u64*)malloc(sizeof(u64)*(size_t)(c+1));
  c = 0; out[c++] = 2;
  for (u64 i = 1; 2*i+1 <= limit; i++) if (!(bits[i>>6]&(1ull<<(i&63)))) out[c++] = 2*i+1;
  free(bits); *cnt = c; return out;
}

static const char* gpu_kernel_program =
"#define SEG_BYTES 16384u\n"
"inline uchar clrbit(uchar r){ if (r==1) return 0x80u; if (r==7) return 0x01u; if (r==11) return 0x02u;"
" if (r==13) return 0x04u; if (r==17) return 0x08u; if (r==19) return 0x10u;"
" if (r==23) return 0x20u; if (r==29) return 0x40u; return 0xFFu; }\n"
"inline int bitidx(uchar r){ if (r==1) return 7; if (r==7) return 0; if (r==11) return 1;"
" if (r==13) return 2; if (r==17) return 3; if (r==19) return 4;"
" if (r==23) return 5; if (r==29) return 6; return 7; }\n"
"inline int isunit(uchar r){ return (r==1||r==7||r==11||r==13||r==17||r==19||r==23||r==29); }\n"
"inline int pop2(ulong v){ v=v-((v>>1)&0x5555555555555555UL);"
" v=(v&0x3333333333333333UL)+((v>>2)&0x3333333333333333UL);"
" return (int)((((v+(v>>4))&0x0F0F0F0F0F0F0F0FUL)*0x0101010101010101UL)>>56); }\n"
"kernel void gsieve(global const uint* prim, uint np, ulong top, global ulong* counters)\n"
"{\n"
"  uint gid = get_group_id(0);\n"
"  uint lid = get_local_id(0);\n"
"  uint L   = get_local_size(0);\n"
"  ulong segLow = (ulong)30 * SEG_BYTES * gid;\n"
"  ulong segEnd = segLow + (ulong)30 * SEG_BYTES;\n"
"  local uchar seg[SEG_BYTES];\n"
"  for (uint i = lid; i < SEG_BYTES; i += L) seg[i] = 0xFF;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for (uint pi = 0; pi < np; pi++){\n"
"    uint p = prim[pi];\n"
"    if (p < 7) continue;\n"
"    ulong pp = (ulong)p * p;\n"
"    if (pp >= segEnd) break;\n"
"    ulong v0 = (pp > segLow + 7) ? pp : segLow + 7;\n"
"    ulong q0 = (v0 + p - 1) / p;\n"
"    ulong qmax = segEnd / p;\n"
"    for (ulong q = q0 + lid; q <= qmax; q += L){\n"
"      ulong v = p * q;\n"
"      if (v >= segLow + 7 && v < segEnd){\n"
"        uchar r = (uchar)(v % 30);\n"
"        if (isunit(r)){\n"
"          ulong idx = (v - segLow) / 30;\n          if (r == 1) idx -= 1;\n"
"          if (idx < SEG_BYTES){\n"
"            uint wi = (uint)idx >> 2;\n"
"            uint by = (uint)idx & 3;\n"
"            uint wordMask = ~(((uint)1u << bitidx(r)) << (8*by));\n"
"            atomic_and(&((__local uint*)seg)[wi], wordMask);\n"
"          }\n"
"        }\n"
"      }\n"
"    }\n"
"  }\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"

"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  ulong cap = (segEnd < top) ? (segEnd + 2) : top;\n"
"  ulong rel = cap - segLow;\n"
"  if (rel > (ulong)30 * SEG_BYTES) rel = (ulong)30 * SEG_BYTES;\n"
"  ulong nbits = 0;\n"
"  if (rel > 1){\n"
"    ulong full = rel / 30; ulong rem = rel % 30;\n"
"    int cup[30] = {0,0,0,0,0,0,0,1,0,0,0,2,0,0,3,0,0,4,0,5,0,0,0,6,0,0,0,0,7,0};\n"
"    nbits = full*8 + (ulong)cup[rem];\n"
"    nbits -= 1;\n"
"  }\n"
"  if (nbits > (ulong)SEG_BYTES*8) nbits = (ulong)SEG_BYTES*8;\n"
"  ulong nf = nbits / 64, nb = nbits & 63;\n"
"  ulong cnt = 0;\n"
"  { __local ulong* w = (__local ulong*)seg;\n"
"    for (ulong i = lid; i < nf; i += L) cnt += pop2(w[i]);\n"
"    if (nb && lid == 0) cnt += pop2(w[nf] & ((1ul<<nb)-1));\n"
"  }\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  local ulong lsum[256];\n"
"  lsum[lid] = cnt; barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for (uint s = L/2; s > 0; s >>= 1){ if (lid < s) lsum[lid] += lsum[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); }\n"
"  if (lid == 0) counters[gid] = lsum[0];\n"
"}\n";

static double now(void){ LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart/f.QuadPart; }

int main(int argc, char** argv){
  u64 top = 100000000000ULL;
  int reps = 1;
  for (int a = 1; a < argc; a++){
    if (!strcmp(argv[a], "-r") && a+1 < argc) reps = atoi(argv[++a]);
    else top = strtoull(argv[a], 0, 10);
  }
  if (!load_opencl()){ fprintf(stderr, "OpenCL runtime unavailable\n"); return 1; }

  cl_uint nplat; cl_platform_id pids[8];
  GetPlatformIDs(8, pids, &nplat);
  cl_uint nd = 0; cl_device_id dev = 0;
  for (cl_uint pi = 0; pi < nplat && !nd; pi++)
    GetDeviceIDs(pids[pi], 0x4, 1, &dev, &nd);
  if (!nd){ fprintf(stderr, "no GPU\n"); return 1; }

  u64 root = 1; while ((root+1) <= top/(root+1)) root++;
  u64 np_ = 0; u64* pl = simple_primes(root, &np_);
  u32* prime32 = (u32*)malloc(sizeof(u32)*(size_t)np_);
  for (u64 i = 0; i < np_; i++) prime32[i] = (u32)pl[i];

  cl_int err;
  cl_context ctx = CreateContext(0, 1, &dev, 0, 0, &err);
  cl_command_queue q = CreateCommandQueueWithProperties
      ? CreateCommandQueueWithProperties(ctx, dev, 0, &err)
      : CreateCommandQueue(ctx, dev, 0, &err);
  const char* src[] = { gpu_kernel_program };
  size_t srclen = strlen(gpu_kernel_program);
  cl_program prog = CreateProgramWithSource(ctx, 1, src, &srclen, &err);
  err = BuildProgram(prog, 1, &dev, "-cl-std=CL1.2", 0, 0);
  if (err != 0){
    char log[16384] = {0}; size_t lsz = 0;
    GetProgramBuildInfo(prog, dev, 0x1183, sizeof(log), log, &lsz);
    fprintf(stderr, "build failed (%d):\n%s\n", err, log);
    return 1;
  }
cl_kernel ker = CreateKernel(prog, "gsieve", &err);
  printf("CreateKernel err=%d\n", err);
  const u64 SEG = 30ULL * 16384ULL;
  u64 nblocks = (top + SEG - 1) / SEG;
  cl_mem bufP = CreateBuffer(ctx, 0x1, np_*4, 0, &err);
  cl_mem bufC = CreateBuffer(ctx, 0x2, nblocks*8, 0, &err);
  EnqueueWriteBuffer(q, bufP, 1, 0, np_*4, prime32, 0, 0, 0);
  u64* counters = (u64*)calloc((size_t)nblocks, 8);
  cl_uint arg = 0;
  SetKernelArg(ker, arg++, sizeof(bufP), &bufP);
  SetKernelArg(ker, arg++, sizeof(cl_uint), &np_);
  SetKernelArg(ker, arg++, sizeof(cl_ulong), &top);
  SetKernelArg(ker, arg++, sizeof(bufC), &bufC);
  size_t gsz = nblocks * 256, lsz = 256;

  for (int rep = 0; rep < reps; rep++){
    double t0 = now();
err = EnqueueNDRangeKernel(q, ker, 1, 0, &gsz, &lsz, 0, 0, 0);
    printf("err enqueue=%d (gsz=%llu)\n", err, (unsigned long long)gsz);
    if (err) fprintf(stderr, "ERNR err=%d\n", err);
    Finish(q);
    double t1 = now();
    EnqueueReadBuffer(q, bufC, 1, 0, nblocks*8, counters, 0, 0, 0);
    Finish(q);
    u64 total = 0;
    for (u64 i = 0; i < nblocks; i++) total += counters[i];
    if (top >= 5) total += 3; else if (top >= 3) total += 2; else if (top >= 2) total += 1;
printf("pi(%llu) = %llu   [%.3f s, %.2f Gcand/s]\n",
           (unsigned long long)top, (unsigned long long)total, t1 - t0, (double)top/1e9/(t1 - t0));
  }
  return 0;
}

