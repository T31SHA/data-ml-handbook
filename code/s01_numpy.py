import numpy as np, time
x = np.random.default_rng(0).normal(size=1_000_000)
t = time.perf_counter(); s1 = sum(v * v for v in x); t_loop = time.perf_counter() - t
t = time.perf_counter(); s2 = x @ x; t_vec = time.perf_counter() - t
print(f"loop {t_loop*1e3:.0f} ms vs vectorised {t_vec*1e3:.2f} ms  -> {t_loop/t_vec:.0f}x faster, same result: {np.isclose(s1, s2)}")
A = np.arange(6).reshape(2, 3); b = np.array([10, 20, 30])
print("broadcast (2,3)+(3,) ->\n", A + b)
print("standardise columns:", ((A - A.mean(0)) / A.std(0)).round(2).tolist())
