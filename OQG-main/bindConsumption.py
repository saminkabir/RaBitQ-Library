import numpy as np
import time
import oqglib

def bench(fn, *args, repeat=200):
    # warmup
    for _ in range(50):
        fn(*args)
    t0 = time.perf_counter_ns()
    for _ in range(repeat):
        fn(*args)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / repeat

n, d = 10000, 128
qs = np.random.rand(n, d).astype(np.float32)
q0 = qs[0]

t_big = bench(oqglib.many_nd, qs, repeat=50)      # batch大一点 repeat小一点
t_small = bench(oqglib.one_d, q0, repeat=2000)   # 单个 repeat大一点

# n次 small 的估计
t_n_small = t_small * n

print("t_big (one call on n*d):", t_big/1e9, "s")
print("t_n_small (n calls on d):", t_n_small/1e9, "s")
print("difference:", (t_n_small - t_big)/1e9, "s  ~ pybind overhead")

