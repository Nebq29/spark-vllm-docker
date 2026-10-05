"""lanex.py — 4-lane TCP exchange-and-add all_reduce for TP=2 on Jetson Thor.

Hot path is C (lanex_core.so, one pthread per lane). Flow per all_reduce(x),
all on the caller's current CUDA stream:
  1. D2H: x -> host buffer A
  2. exchange: C threads send A / receive the peer's buffer into B across 4 lanes
  3. H2D: B -> GPU scratch C
  4. x += C on the GPU

LANEX_ASYNC=1 (default): step 2 is queued on the stream as a host callback
(lanex_xchg_async in lanex_async.c, via cuLaunchHostFunc). Python enqueues all
four steps and returns immediately, so it keeps launching the next layer's
kernels while the exchange runs, just as it does with NCCL.
LANEX_ASYNC=0: the original synchronous path (event.synchronize(), then the
exchange on the Python thread). Kept for A/B comparison.

fp add is commutative -> both ranks get bit-identical results.
Messages < LANEX_MIN_BYTES fall through to the original pynccl path, so decode
(70 KB) is protected by construction.

Env:
  LANEX_ENABLE=1
  LANEX_LOCAL_IPS / LANEX_PEER_IPS   4 lane IPs each
  LANEX_PORT=49000                   lane i uses port+i
  LANEX_MIN_BYTES=262144
  LANEX_LIB=/path/to/lanex_core.so   (lanex_core.c v2 + lanex_async.c)
  LANEX_ASYNC=1
  LANEX_HOSTBUF=alloc   alloc    = cudaHostAlloc via torch pin_memory (default)
                        register = mmap + cudaHostRegister (round 2: no faster)
                        managed  = cuMemAllocManaged. Only allowed when the GPU
                                   reports CONCURRENT_MANAGED_ACCESS=1, because
                                   the lane threads touch it while kernels run.
                        hostmem  = ordinary pageable memory accessed by the GPU
                                   directly (needs PAGEABLE_MEMORY_ACCESS=1, which
                                   Thor has). Allocation and page size are set by
                                   LANEX_HOSTMEM_ALLOC=malloc|mmap (default malloc)
                                   and LANEX_HOSTMEM_THP=nohuge|huge|default
                                   (default nohuge). Round 4 used mmap+huge, which
                                   was slow for the CPU send path.
                                   A Triton kernel writes x into A; after the
                                   exchange a second kernel does x += B in place,
                                   so there is no separate H2D copy. The CPU lane
                                   threads get the fast (cached) page class.
  LANEX_ZEROCOPY=0      1 = MSG_ZEROCOPY sends (NIC reads the send buffer by DMA;
                        needs lanex_core.c v3, and RLIMIT_MEMLOCK unlimited or
                        CAP_IPC_LOCK in the container, e.g. --ulimit memlock=-1:-1)
  LANEX_PROF=0          1 = print a timing summary every
  LANEX_PROF_EVERY=89     this many lanex calls (89 ~ one prefill)
"""
import ctypes
import mmap
import os
import time

import torch

LANEX_ENABLE = os.environ.get("LANEX_ENABLE", "0") == "1"
LANEX_LOCAL_IPS = os.environ.get(
    "LANEX_LOCAL_IPS", "10.0.0.1,10.0.1.1,10.0.2.1,10.0.3.1")
LANEX_PEER_IPS = os.environ.get(
    "LANEX_PEER_IPS", "10.0.0.2,10.0.1.2,10.0.2.2,10.0.3.2")
LANEX_PORT = int(os.environ.get("LANEX_PORT", "49000"))
LANEX_MIN_BYTES = int(os.environ.get("LANEX_MIN_BYTES", str(256 * 1024)))
LANEX_LIB = os.environ.get("LANEX_LIB", "/tmp/lanex_core.so")
LANEX_ASYNC = os.environ.get("LANEX_ASYNC", "1") == "1"
LANEX_HOSTBUF = os.environ.get("LANEX_HOSTBUF", "alloc")
LANEX_PROF = os.environ.get("LANEX_PROF", "0") == "1"
LANEX_PROF_EVERY = int(os.environ.get("LANEX_PROF_EVERY", "89"))
NL = 4
NJOBS = 1024  # ring of in-flight exchange descriptors (one per queued all-reduce)

_lib = ctypes.CDLL(LANEX_LIB)
_lib.lanex_create.restype = ctypes.c_void_p
_lib.lanex_create.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.c_int,
]
_lib.lanex_xchg.restype = ctypes.c_int
_lib.lanex_xchg.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
]
_lib.lanex_last_stats.restype = None
_lib.lanex_last_stats.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double)]
_lib.lanex_destroy.restype = None
_lib.lanex_destroy.argtypes = [ctypes.c_void_p]
_lib.lanex_set_zerocopy.restype = ctypes.c_int
_lib.lanex_set_zerocopy.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.lanex_zc_stats.restype = None
_lib.lanex_zc_stats.argtypes = [ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64),
                                ctypes.POINTER(ctypes.c_uint64)]
LANEX_ZEROCOPY = os.environ.get("LANEX_ZEROCOPY", "0") == "1"
# LANEX_COMPRESS=int8: exchange int8 blocks of 128 + fp32 scale (~53% of bf16 bytes).
# hostmem mode, bf16/fp16 only. Not bit-identical to an uncompressed all-reduce, but
# both ranks get bit-identical results. Default off.
LANEX_COMPRESS = os.environ.get("LANEX_COMPRESS", "none")
LANEX_HOSTMEM_ALLOC = os.environ.get("LANEX_HOSTMEM_ALLOC", "malloc")
LANEX_HOSTMEM_THP = os.environ.get("LANEX_HOSTMEM_THP", "nohuge")


def zc_stats(h):
    n, c = ctypes.c_uint64(), ctypes.c_uint64()
    _lib.lanex_zc_stats(h, ctypes.byref(n), ctypes.byref(c))
    return n.value, c.value


class _Job(ctypes.Structure):
    # Must match struct lanex_job in lanex_async.c.
    _fields_ = [
        ("h", ctypes.c_void_p),
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("n", ctypes.c_size_t),
        ("rc", ctypes.c_int),
        ("done", ctypes.c_int),
        ("us", ctypes.c_double),
        ("wait_us", ctypes.c_double),
    ]


if LANEX_ASYNC:
    _lib.lanex_xchg_async.restype = ctypes.c_int
    _lib.lanex_xchg_async.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Job)]


# ---- CUDA driver API (libcuda is already loaded by torch.cuda) -----------------
_cu = ctypes.CDLL("libcuda.so.1")
_cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_cu.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_cu.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint]
_cu.cuMemFree_v2.argtypes = [ctypes.c_uint64]
_cu.cuMemcpyAsync.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]

CU_ATTRS = {  # CUdevice_attribute values from cuda.h
    "INTEGRATED": 18,
    "MANAGED_MEMORY": 83,
    "HOST_NATIVE_ATOMIC_SUPPORTED": 86,
    "PAGEABLE_MEMORY_ACCESS": 88,
    "CONCURRENT_MANAGED_ACCESS": 89,
    "CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM": 91,
    "CAN_USE_STREAM_MEM_OPS": 92,
    "HOST_REGISTER_SUPPORTED": 99,
    "PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES": 100,
    "DIRECT_MANAGED_MEM_ACCESS_FROM_HOST": 101,
}


def cuda_mem_attrs(ordinal=None):
    torch.cuda.init()
    ordinal = torch.cuda.current_device() if ordinal is None else ordinal
    dev = ctypes.c_int()
    _cu.cuDeviceGet(ctypes.byref(dev), ordinal)
    out = {}
    for name, a in CU_ATTRS.items():
        v = ctypes.c_int(-1)
        rc = _cu.cuDeviceGetAttribute(ctypes.byref(v), a, dev.value)
        out[name] = v.value if rc == 0 else f"err{rc}"
    return out


def host_buffer(nbytes, mode=None):
    """Return (cpu_tensor_view, keepalive)."""
    mode = mode or LANEX_HOSTBUF
    if mode == "alloc":
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True), None
    if mode == "register":
        mm = mmap.mmap(-1, nbytes)
        t = torch.frombuffer(mm, dtype=torch.uint8)
        t.fill_(0)  # fault the pages in before pinning
        rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 0)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {rc}")
        return t, ("register", mm)
    if mode == "hostmem":
        return hostmem_buffer(nbytes)
    if mode == "managed":
        p = ctypes.c_uint64()
        rc = _cu.cuMemAllocManaged(ctypes.byref(p), nbytes, 1)  # CU_MEM_ATTACH_GLOBAL
        if rc != 0:
            raise RuntimeError(f"cuMemAllocManaged failed rc={rc}")
        arr = (ctypes.c_uint8 * nbytes).from_address(p.value)
        t = torch.frombuffer(arr, dtype=torch.uint8)
        t.fill_(0)
        return t, ("managed", p.value, arr)
    raise ValueError(f"LANEX_HOSTBUF={mode!r} (use alloc, register, managed or hostmem)")


_libc = ctypes.CDLL(None, use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_MADV = {"huge": 14, "nohuge": 15}  # MADV_HUGEPAGE, MADV_NOHUGEPAGE
_PAGE = 4096


def _madvise(addr, length, thp):
    """Apply THP advice to the page-aligned part of [addr, addr+length)."""
    if thp == "default":
        return
    lo = (addr + _PAGE - 1) // _PAGE * _PAGE
    hi = (addr + length) // _PAGE * _PAGE
    if hi > lo and _libc.madvise(lo, hi - lo, _MADV[thp]) != 0:
        raise OSError(ctypes.get_errno(), f"madvise({thp}) failed")


def hostmem_buffer(nbytes, alloc=None, thp=None):
    """Ordinary pageable host memory for the GPU-direct (hostmem) path.
    alloc: 'malloc'      torch CPU allocator (private anonymous memory)
           'mmap'        private anonymous mmap (MAP_PRIVATE|MAP_ANONYMOUS)
           'mmap_shared' shared anonymous mmap (Python's default; shmem-backed).
                         This is what round 4's hostmem used.
    thp:   'nohuge' (MADV_NOHUGEPAGE, 4 KB pages), 'huge' (MADV_HUGEPAGE),
           or 'default' (no advice; the kernel decides).
    Advice is applied before the pages are first touched."""
    alloc = alloc or LANEX_HOSTMEM_ALLOC
    thp = thp or LANEX_HOSTMEM_THP
    if thp not in ("huge", "nohuge", "default"):
        raise ValueError(f"LANEX_HOSTMEM_THP={thp!r}")
    if alloc in ("mmap", "mmap_shared"):
        size = (nbytes + (2 << 20) - 1) // (2 << 20) * (2 << 20)
        if alloc == "mmap":
            mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        else:
            mm = mmap.mmap(-1, size)
        base = torch.frombuffer(mm, dtype=torch.uint8)
        _madvise(base.data_ptr(), size, thp)
        keep = ("hostmem", mm, base)
    elif alloc == "malloc":
        base = torch.empty(nbytes, dtype=torch.uint8)  # not touched yet
        _madvise(base.data_ptr(), nbytes, thp)
        keep = ("hostmem", base)
    else:
        raise ValueError(f"LANEX_HOSTMEM_ALLOC={alloc!r}")
    t = base[:nbytes]
    t.fill_(0)  # populate the pages before the GPU or the lanes touch them
    return t, keep


def vma_info(addr):
    """Describe the mapping containing addr from /proc/self/smaps:
    (size_kB, rss_kB, anon_huge_kB, vmflags, path)."""
    try:
        with open("/proc/self/smaps") as f:
            cur = None
            for line in f:
                parts = line.split()
                if "-" in parts[0] and len(parts) >= 5 and ":" not in parts[0]:
                    lo, hi = (int(v, 16) for v in parts[0].split("-"))
                    if cur is not None:
                        return cur
                    if lo <= addr < hi:
                        cur = {"size": (hi - lo) // 1024, "rss": 0, "huge": 0,
                               "flags": "", "path": parts[5] if len(parts) > 5 else "[anon]"}
                    continue
                if cur is None:
                    continue
                if parts[0] == "Rss:":
                    cur["rss"] = int(parts[1])
                elif parts[0] == "AnonHugePages:":
                    cur["huge"] = int(parts[1])
                elif parts[0] == "VmFlags:":
                    cur["flags"] = " ".join(parts[1:])
            return cur
    except OSError:
        return None


def free_host_buffer(t, keep):
    if keep is None:
        return
    if keep[0] == "register":
        torch.cuda.cudart().cudaHostUnregister(t.data_ptr())
    elif keep[0] == "managed":
        _cu.cuMemFree_v2(keep[1])


def copy_async(dst_ptr, src_ptr, nbytes, stream):
    """Stream-ordered copy between any two unified addresses (used for managed)."""
    rc = _cu.cuMemcpyAsync(dst_ptr, src_ptr, nbytes, stream.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"cuMemcpyAsync failed rc={rc}")


_tk = None  # Triton kernels for hostmem mode, compiled on first use


def _triton_kernels():
    global _tk
    if _tk is None:
        import triton
        import triton.language as tl

        @triton.jit
        def copy_out(x_ptr, dst_addr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            dst = dst_addr.to(tl.pointer_type(x_ptr.dtype.element_ty))
            tl.store(dst + offs, tl.load(x_ptr + offs, mask=m), mask=m)

        @triton.jit
        def add_in(x_ptr, src_addr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            src = src_addr.to(tl.pointer_type(x_ptr.dtype.element_ty))
            a = tl.load(x_ptr + offs, mask=m)
            b = tl.load(src + offs, mask=m)
            # same as torch's add_ for bf16/fp16: add in fp32, round once
            tl.store(x_ptr + offs, (a.to(tl.float32) + b.to(tl.float32)).to(a.dtype), mask=m)

        @triton.jit
        def q8_out(x_ptr, dst_addr, n, soff, G: tl.constexpr, GP: tl.constexpr):
            # x -> int8 per G-element block + one fp32 scale per block, written to
            # host memory: data at dst_addr[0:n], scales at dst_addr[soff:soff+4*ngroups]
            rows = tl.program_id(0) * GP + tl.arange(0, GP)
            offs = rows[:, None] * G + tl.arange(0, G)[None, :]
            m = offs < n
            x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=1)
            scale = tl.where(amax > 0, amax / 127.0, 1.0)
            y = x / scale[:, None]
            q = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(dst_addr.to(tl.pointer_type(tl.int8)) + offs, q.to(tl.int8), mask=m)
            ng = (n + G - 1) // G
            tl.store((dst_addr + soff).to(tl.pointer_type(tl.float32)) + rows, scale, mask=rows < ng)

        @triton.jit
        def q8_add(x_ptr, first_addr, second_addr, n, soff, G: tl.constexpr, GP: tl.constexpr):
            # x = dequant(first) + dequant(second). Both ranks pass rank 0's data as
            # `first`, so the arithmetic (including any FMA contraction) is identical
            # on both ranks and the results are bit-identical.
            rows = tl.program_id(0) * GP + tl.arange(0, GP)
            offs = rows[:, None] * G + tl.arange(0, G)[None, :]
            m = offs < n
            ng = (n + G - 1) // G
            rm = rows < ng
            q1 = tl.load(first_addr.to(tl.pointer_type(tl.int8)) + offs, mask=m, other=0).to(tl.float32)
            q2 = tl.load(second_addr.to(tl.pointer_type(tl.int8)) + offs, mask=m, other=0).to(tl.float32)
            s1 = tl.load((first_addr + soff).to(tl.pointer_type(tl.float32)) + rows, mask=rm, other=0.0)
            s2 = tl.load((second_addr + soff).to(tl.pointer_type(tl.float32)) + rows, mask=rm, other=0.0)
            out = q1 * s1[:, None] + q2 * s2[:, None]
            tl.store(x_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=m)

        _tk = (triton, copy_out, add_in, q8_out, q8_add)
    return _tk


Q8_G = int(os.environ.get("LANEX_Q8_G", "128"))  # elements per int8 block (one fp32 scale); power of 2: 32/64/128/256
assert Q8_G in (32, 64, 128, 256), "LANEX_Q8_G must be 32, 64, 128 or 256"
Q8_GP = 8    # blocks per Triton program


def q8_layout(n):
    """(scale offset, total bytes exchanged) for n elements in int8 mode."""
    soff = (n + 15) // 16 * 16
    ng = (n + Q8_G - 1) // Q8_G
    return soff, soff + 4 * ng


def gpu_q8_out(x, host_t):
    triton, _, _, q8_out, _ = _triton_kernels()
    n = x.numel()
    soff, _ = q8_layout(n)
    ng = (n + Q8_G - 1) // Q8_G
    q8_out[(triton.cdiv(ng, Q8_GP),)](x, host_t.data_ptr(), n, soff, G=Q8_G, GP=Q8_GP)


def gpu_q8_add(x, first_t, second_t):
    triton, _, _, _, q8_add = _triton_kernels()
    n = x.numel()
    soff, _ = q8_layout(n)
    ng = (n + Q8_G - 1) // Q8_G
    q8_add[(triton.cdiv(ng, Q8_GP),)](x, first_t.data_ptr(), second_t.data_ptr(), n, soff,
                                      G=Q8_G, GP=Q8_GP)


def q8_roundtrip_ref(x):
    """Torch reference of what one rank's contribution becomes after int8 blocks
    (for tests; Triton's fp32 division may differ in the last bit)."""
    f = x.float().reshape(-1)
    n = f.numel()
    pad = (-n) % Q8_G
    g = torch.nn.functional.pad(f, (0, pad)).view(-1, Q8_G)
    amax = g.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    y = g / scale
    q = torch.where(y >= 0, torch.floor(y + 0.5), torch.ceil(y - 0.5)).clamp(-127, 127)
    return (q * scale).reshape(-1)[:n]


_BLOCK = 4096


def gpu_copy_out(x, host_t):
    """x (CUDA, contiguous) -> pageable host memory at host_t, on the current stream."""
    triton, copy_out = _triton_kernels()[:2]
    n = x.numel()
    copy_out[(triton.cdiv(n, _BLOCK),)](x, host_t.data_ptr(), n, BLOCK=_BLOCK)


def gpu_add_in(x, host_t):
    """x += (pageable host memory at host_t viewed as x.dtype), on the current stream."""
    triton, _, add_in = _triton_kernels()[:3]
    n = x.numel()
    add_in[(triton.cdiv(n, _BLOCK),)](x, host_t.data_ptr(), n, BLOCK=_BLOCK)


def _cstr_array(items):
    bs = [s.encode() for s in items]
    arr = (ctypes.c_char_p * len(bs))(*bs)
    return arr, bs  # keep bs alive alongside the pointer array


class LaneX:
    """One instance per process. rank 0 listens, rank 1 connects."""

    def __init__(self, rank):
        self.rank = rank
        if LANEX_HOSTBUF == "managed":
            cma = cuda_mem_attrs()["CONCURRENT_MANAGED_ACCESS"]
            if cma != 1:
                raise RuntimeError(
                    f"LANEX_HOSTBUF=managed needs CONCURRENT_MANAGED_ACCESS=1, got {cma}")
        if LANEX_HOSTBUF == "hostmem":
            pma = cuda_mem_attrs()["PAGEABLE_MEMORY_ACCESS"]
            if pma != 1:
                raise RuntimeError(
                    f"LANEX_HOSTBUF=hostmem needs PAGEABLE_MEMORY_ACCESS=1, got {pma}")
        lip = LANEX_LOCAL_IPS.split(",")
        pip = LANEX_PEER_IPS.split(",")
        assert len(lip) == NL and len(pip) == NL, "need exactly 4 lane IPs"
        self._lip_arr, self._lip_keep = _cstr_array(lip)
        self._pip_arr, self._pip_keep = _cstr_array(pip)
        self._h = _lib.lanex_create(
            NL, self._lip_arr, self._pip_arr, LANEX_PORT, 1 if rank == 0 else 0)
        if not self._h:
            raise RuntimeError("lanex_create failed (check lane IPs / peer up)")
        if LANEX_ZEROCOPY:
            rc = _lib.lanex_set_zerocopy(self._h, 1)
            if rc != 0:
                raise RuntimeError(f"SO_ZEROCOPY failed errno={rc}")
        # Staging buffers per CUDA stream: (A host, B host, C GPU scratch, keepA, keepB).
        # Work on one stream is serialized, so one set per stream is race-free
        # even with many all-reduces queued ahead.
        self._bufs = {}
        self._jobs = (_Job * NJOBS)()
        for j in self._jobs:
            j.done = 1
        self._ji = 0
        self._prof_calls = 0
        self._prof_cpu = 0.0    # seconds Python spent inside all_reduce
        self._prof_block = 0.0  # seconds blocked in event.synchronize (sync mode)
        self._prof_xchg = 0.0   # seconds inside lanex_xchg
        self._prof_wait = 0.0   # part of xchg waiting for the peer's first byte
        self._prof_n = 0        # exchanges accounted in xchg/wait

    def _buffers(self, stream, nbytes):
        key = stream.cuda_stream
        b = self._bufs.get(key)
        if b is None or b[0].numel() < nbytes:
            if b is not None:
                # Earlier queued exchanges may still reference the old buffers.
                stream.synchronize()
                free_host_buffer(b[0], b[3])
                free_host_buffer(b[1], b[4])
            A, ka = host_buffer(nbytes)
            B, kb = host_buffer(nbytes)
            C = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            b = (A, B, C, ka, kb)
            self._bufs[key] = b
        return b[0], b[1], b[2]

    def _next_job(self, stream):
        j = self._jobs[self._ji]
        self._ji = (self._ji + 1) % NJOBS
        if not j.done:
            # NJOBS exchanges queued and not yet run: wait for the oldest.
            stream.synchronize()
        self._harvest(j)
        return j

    def _harvest(self, j):
        if j.n:
            if j.rc != 0:
                raise RuntimeError(f"lanex_xchg failed errno={j.rc}")
            self._prof_xchg += j.us * 1e-6
            self._prof_wait += j.wait_us * 1e-6
            self._prof_n += 1
            j.n = 0

    def all_reduce(self, x):
        assert x.is_cuda
        t0 = time.perf_counter()
        if not x.is_contiguous():
            x = x.contiguous()
        nbytes = x.numel() * x.element_size()
        cur = torch.cuda.current_stream()
        A, B, C = self._buffers(cur, nbytes)
        Cv = C[:nbytes].view(x.dtype)
        managed = LANEX_HOSTBUF == "managed"
        hostmem = LANEX_HOSTBUF == "hostmem"
        xf = x.view(-1)

        # 1. D2H into A
        xbytes = nbytes
        q8 = (hostmem and LANEX_COMPRESS == "int8"
              and x.dtype in (torch.bfloat16, torch.float16)
              and q8_layout(xf.numel())[1] <= nbytes)
        if q8:
            xbytes = q8_layout(xf.numel())[1]
            gpu_q8_out(xf, A)
        elif hostmem:
            gpu_copy_out(xf, A)
        elif managed:
            copy_async(A.data_ptr(), x.data_ptr(), nbytes, cur)
        else:
            A[:nbytes].view(x.dtype).copy_(xf, non_blocking=True)
        # 2. exchange A -> peer, peer -> B
        if LANEX_ASYNC:
            j = self._next_job(cur)
            j.h, j.a, j.b, j.n = self._h, A.data_ptr(), B.data_ptr(), xbytes
            rc = _lib.lanex_xchg_async(cur.cuda_stream, ctypes.byref(j))
            if rc != 0:
                raise RuntimeError(f"cuLaunchHostFunc failed rc={rc}")
        else:
            ev = torch.cuda.Event()
            ev.record(cur)
            tw = time.perf_counter()
            ev.synchronize()
            self._prof_block += time.perf_counter() - tw
            rc = _lib.lanex_xchg(self._h,
                                 ctypes.cast(A.data_ptr(), ctypes.c_char_p),
                                 ctypes.cast(B.data_ptr(), ctypes.c_char_p),
                                 xbytes)
            if rc != 0:
                raise RuntimeError(f"lanex_xchg failed errno={rc}")
            w, t = ctypes.c_double(), ctypes.c_double()
            _lib.lanex_last_stats(self._h, ctypes.byref(w), ctypes.byref(t))
            self._prof_xchg += t.value * 1e-6
            self._prof_wait += w.value * 1e-6
            self._prof_n += 1
        # 3. H2D peer data to GPU scratch, 4. add on GPU (stream-ordered)
        if q8:
            first, second = (A, B) if self.rank == 0 else (B, A)
            gpu_q8_add(xf, first, second)
        elif hostmem:
            gpu_add_in(xf, B)  # reads B directly from host memory; no H2D copy
        else:
            if managed:
                copy_async(C.data_ptr(), B.data_ptr(), nbytes, cur)
            else:
                Cv.copy_(B[:nbytes].view(x.dtype), non_blocking=True)
            xf.add_(Cv)

        if LANEX_PROF:
            self._prof_cpu += time.perf_counter() - t0
            self._prof_calls += 1
            if self._prof_calls % LANEX_PROF_EVERY == 0:
                self._report(cur)
        return x

    def _report(self, stream):
        if LANEX_ASYNC:
            stream.synchronize()  # profiling only: let queued exchanges finish
            for j in self._jobs:
                if j.done:
                    self._harvest(j)
        n = max(self._prof_n, 1)
        print(f"[lanex rank{self.rank}] {LANEX_PROF_EVERY} calls  "
              f"python-in-hook {self._prof_cpu*1e3:.1f} ms  "
              f"blocked-on-gpu {self._prof_block*1e3:.1f} ms  "
              f"xchg {self._prof_xchg*1e3:.1f} ms  "
              f"(peer-wait {self._prof_wait*1e3:.1f} ms, "
              f"transfer {(self._prof_xchg-self._prof_wait)*1e3:.1f} ms, "
              f"{self._prof_n} xchg, {self._prof_xchg/n*1e3:.2f} ms avg)  "
              f"mode={'async' if LANEX_ASYNC else 'sync'} hostbuf={LANEX_HOSTBUF}"
              + (" zerocopy notifs/copied=%d/%d" % zc_stats(self._h) if LANEX_ZEROCOPY else ""),
              flush=True)
        self._prof_cpu = self._prof_block = self._prof_xchg = self._prof_wait = 0.0
        self._prof_n = 0

    def close(self):
        if self._h:
            torch.cuda.synchronize()
            for b in self._bufs.values():
                free_host_buffer(b[0], b[3])
                free_host_buffer(b[1], b[4])
            self._bufs = {}
            _lib.lanex_destroy(self._h)
            self._h = None


# ---- vLLM hook ---------------------------------------------------------------
_INSTALLED = False
_LX = {}


def install():
    """Monkey-patch GroupCoordinator.all_reduce for big TP=2 messages."""
    global _INSTALLED
    if _INSTALLED or not LANEX_ENABLE:
        return
    import vllm.distributed.parallel_state as ps

    orig = ps.GroupCoordinator.all_reduce

    def patched(self, input_, *args, **kwargs):
        if self.world_size != 2:
            return orig(self, input_, *args, **kwargs)
        nbytes = input_.numel() * input_.element_size()
        if nbytes < LANEX_MIN_BYTES:
            return orig(self, input_, *args, **kwargs)
        lx = _LX.get(id(self))
        if lx is None:
            lx = LaneX(self.rank)
            _LX[id(self)] = lx
        return lx.all_reduce(input_)

    ps.GroupCoordinator.all_reduce = patched
    _INSTALLED = True
