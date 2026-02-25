import numpy as np
from astropy.timeseries import LombScargle

# Match your actual data characteristics
rng = np.random.default_rng(42)

# Case 1: Gaia-like — 302k points, t ~ 1700–2700, y ~ 1.0, small amplitude
t1 = np.sort(rng.uniform(1738, 2743, 302099))
y1 = 1.0 + 0.01 * np.sin(2 * np.pi * t1 / 5.0) + 0.015 * rng.normal(size=len(t1))
dy1 = np.full_like(y1, 0.015)
freqs1 = np.linspace(0.02, 20.0, 401388)

# Case 2: ZTF-like — 3.4k points, t ~ 58000–61000, y ~ 1.0, small amplitude
t2 = np.sort(rng.uniform(58198, 60948, 3397))
y2 = 1.0 + 0.01 * np.sin(2 * np.pi * t2 / 5.0) + 0.015 * rng.normal(size=len(t2))
dy2 = np.full_like(y2, 0.013)
freqs2 = np.linspace(0.02, 20.0, 1098835)

for label, t, y, dy, freqs in [("Gaia-like", t1, y1, dy1, freqs1),
                                 ("ZTF-like",  t2, y2, dy2, freqs2)]:
    print(f"\n=== {label}: n={len(t)}, nfreqs={len(freqs)}, t=[{t.min():.0f},{t.max():.0f}] ===")
    
    print("\n=== Gaia-like, LINEAR FREQ GRID ===")
    freqs_lin = np.linspace(0.02, 20.0, 401388)  # already linear — was used above
    freqs_bad = np.flip(1 / np.linspace(0.05, 50.0, 401388))  # 1/linspace like your code

    p1 = LombScargle(t1[:5000], y1[:5000], dy1[:5000]).power(freqs_lin, method="fast")
    p2 = LombScargle(t1[:5000], y1[:5000], dy1[:5000]).power(freqs_bad, method="fast")

    print(f"  linear freqs: max={np.nanmax(p1):.6f}, NaN={np.isnan(p1).sum()}/{len(p1)}")
    print(f"  1/linear (bad): max={np.nanmax(p2):.6f}, NaN={np.isnan(p2).sum()}/{len(p2)}")