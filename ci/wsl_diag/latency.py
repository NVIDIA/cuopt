import ctypes
import glob
import os
import sys
import threading
import time

paths = sorted(
    glob.glob(os.path.join(os.environ.get("CONDA_PREFIX", "/nonexistent"), "lib", "libcudart.so.*"))
    + glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "nvidia", "cu*", "lib", "libcudart.so.*"))
    + glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "nvidia", "cuda_runtime", "lib", "libcudart.so.*"))
)
print("libcudart:", paths[0], flush=True)
rt = ctypes.CDLL(paths[0])
flags = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print("cudaSetDeviceFlags", flags, "->", rt.cudaSetDeviceFlags(flags), flush=True)
p = ctypes.c_void_p()
assert rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(1 << 20)) == 0
h = (ctypes.c_char * 64)()
s = ctypes.c_void_p()
rt.cudaStreamCreate(ctypes.byref(s))
N = 2000


def bench(name, f):
    f()
    t = time.perf_counter()
    for _ in range(N):
        f()
    print(f"  {name}: {(time.perf_counter() - t) / N * 1e6:.1f} us/op", flush=True)


bench("D2H 8B cudaMemcpy", lambda: rt.cudaMemcpy(h, p, ctypes.c_size_t(8), 2))
bench("H2D 8B cudaMemcpy", lambda: rt.cudaMemcpy(p, h, ctypes.c_size_t(8), 1))
bench("D2H 8B cudaMemcpyAsync+StreamSync", lambda: (rt.cudaMemcpyAsync(h, p, ctypes.c_size_t(8), 2, s), rt.cudaStreamSynchronize(s)))
bench("MemsetAsync+StreamSync", lambda: (rt.cudaMemsetAsync(p, 0, ctypes.c_size_t(8), s), rt.cudaStreamSynchronize(s)))
bench("MemsetAsync x10 + StreamSync", lambda: ([rt.cudaMemsetAsync(p, 0, ctypes.c_size_t(8), s) for _ in range(10)], rt.cudaStreamSynchronize(s)))
bench("cudaStreamSynchronize (idle)", lambda: rt.cudaStreamSynchronize(s))
q = ctypes.c_void_p()
bench("cudaMalloc+Free 1MiB", lambda: (rt.cudaMalloc(ctypes.byref(q), ctypes.c_size_t(1 << 20)), rt.cudaFree(q)))

if flags == 0:
    ev = [threading.Event(), threading.Event()]

    def pong():
        for _ in range(N):
            ev[0].wait()
            ev[0].clear()
            ev[1].set()

    th = threading.Thread(target=pong)
    th.start()
    t = time.perf_counter()
    for _ in range(N):
        ev[0].set()
        ev[1].wait()
        ev[1].clear()
    th.join()
    print(f"  CPU thread ping-pong: {(time.perf_counter() - t) / N * 1e6:.1f} us/roundtrip", flush=True)
    t = time.perf_counter()
    x = 0
    for i in range(3_000_000):
        x += i
    print(f"  CPU python loop 3M: {(time.perf_counter() - t) * 1e3:.0f} ms", flush=True)
