// lanex_async.c -- run lanex_xchg() as a CUDA stream host callback.
//
// The exchange becomes one more step in the stream (D2H -> exchange -> H2D -> add),
// so the Python caller never blocks on the GPU and keeps launching the next layer's
// kernels, the same way it does with NCCL.
//
// Build into the same .so as lanex_core.c (no CUDA headers needed; libcuda is
// dlopen'ed, and it is already loaded in any process that uses torch.cuda):
//   gcc -O2 -shared -fPIC -pthread -o lanex_core.so lanex_core.c lanex_async.c -ldl
#include <dlfcn.h>
#include <stddef.h>

// From lanex_core.c. Declared with void* here, which is ABI-identical.
int lanex_xchg(void *h, void *a, void *b, size_t n);
void lanex_last_stats(void *h, double *wait_us, double *total_us);

typedef int CUresult;
typedef void *CUstream;
typedef void (*CUhostFn)(void *);

// Must match the ctypes Structure _Job in lanex.py field for field.
struct lanex_job {
  void *h;
  void *a;
  void *b;
  size_t n;
  volatile int rc;    // lanex_xchg return code (0 = ok)
  volatile int done;  // 1 once the callback has finished
  double us;          // wall time inside lanex_xchg
  double wait_us;     // part of it spent waiting for the peer's first byte
};

static CUresult (*launch_host_func)(CUstream, CUhostFn, void *);

// Runs on the CUDA driver's callback thread once every earlier op on the stream
// (the D2H into A) has completed. Later ops on the stream (H2D from B, the add)
// wait until it returns. It must not call CUDA APIs; lanex_xchg is pure sockets.
static void lanex_hostfn(void *arg) {
  struct lanex_job *j = arg;
  j->rc = lanex_xchg(j->h, j->a, j->b, j->n);
  lanex_last_stats(j->h, &j->wait_us, &j->us);
  __atomic_store_n(&j->done, 1, __ATOMIC_RELEASE);
}

// stream: torch.cuda.current_stream().cuda_stream (a CUstream / cudaStream_t).
// Returns 0 on success, a CUresult on launch failure, or -1000/-1001 if libcuda
// or cuLaunchHostFunc could not be found.
int lanex_xchg_async(void *stream, struct lanex_job *j) {
  if (!launch_host_func) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) return -1000;
    launch_host_func = (CUresult (*)(CUstream, CUhostFn, void *))dlsym(lib, "cuLaunchHostFunc");
    if (!launch_host_func) return -1001;
  }
  j->rc = 0;
  j->done = 0;
  return launch_host_func((CUstream)stream, lanex_hostfn, j);
}
