"""Typical candidate norm of every candidate distribution (support) the GuacaMol arms draw from.

Result files store no candidate latents or norms (``norm_trace`` is empty except for
adaptive GeoCMA), so each support's typical candidate norm E||x|| is computed here from the
inputs the solvers themselves read, with the solvers' own sampling rule:

* shell (``random_search``, the GP argmax/continuous shell arms, PCA-64, box directions at r):
  every candidate sits at r = mean encoder-mean norm of the run's 5 cold-start molecules
  (``StepByStepSolver.mean_norm``). With the cold-start pool approximated by the full ZINC-250k
  set, E r is EXACTLY the population mean of ``mu_norm.npy`` (the mean of 5 draws without
  replacement is unbiased), and its SD is the finite-population value;
* [p01,p99] and [min,max] boxes: z = lo + u * max(hi - lo, 1e-6), u ~ U[0,1)^D, bounds
  ``per_coord.{p01,p99,min,max}`` of ``results/precomputed/zinc_per_coord_bounds.json``
  (the box GP arms' unit cube and ``random_box_faithful``);
* +-3 sigma cube: u ~ U[-h, h]^D, h = BOX_SIGMA_MULTIPLIER * ZINC_PER_COORD_STD read from
  ``src/analysis/config.py`` (the ``*_box`` arms);
* N(0, I): exact E||z|| = sqrt(2) Gamma((D+1)/2) / Gamma(D/2);
* Gaussian fit (``ellipsoid_rs``): mean + L eps with L = V sqrt(max(lambda, 1e-12)) from
  np.cov of the 3,000-latent cache, exactly as EllipsoidRandomSearchSolver;
* convex hull k (``convex_hull_k*``): k anchors drawn uniformly without replacement from the
  3,000-latent cache, Dirichlet(1) weights, as ConvexHullRandomSearchSolver; k = 1 is exact
  (the mean norm of the 3,000 cached latents).

How each value is obtained (GEN_v2_verify section 9 found one printed Monte Carlo digit wrong
and five established only by luck at 2,000,000 plain draws):

* boxes, the cube and the Gaussian fit have INDEPENDENT coordinates (in the eigenbasis for
  the Gaussian), so E||x|| = E sqrt(S), S = sum_i x_i^2, is a one-dimensional integral of the
  Laplace transform phi(t) = E exp(-t S) = prod_i E exp(-t x_i^2):
      E sqrt(S) = (2 sqrt(pi))^-1  int_0^inf (1 - phi(t)) t^(-3/2) dt
  evaluated by the trapezoid rule in u = log t with analytic tails (``esqrt_from_logphi``);
  the error estimate is the difference between step h and step 2h. A plain Monte Carlo of the
  solver's sampling rule is kept beside it as a check (``mc_check``; a disagreement beyond 5
  standard errors raises);
* convex combinations with k >= 2: Monte Carlo with the control variate S = ||x||^2, whose
  mean is known exactly from the Gram matrix of the cached latents
  (E S = (2 q + (k - 1) c) / (k + 1), q = mean ||z||^2, c = mean z_a.z_b over a != b); draws
  are added in batches from the support's fixed stream until every printed digit is
  established (``digits_established``) or the cap MAX_DRAWS is reached.

Printed digits (``stable_text``): a value is printed with d decimals only if its standard error
is at most 0.2 of the last printed unit AND the value lies at least 4 standard errors away
from the nearest rounding boundary at d decimals; otherwise one decimal fewer is printed.
Exact values and quadrature values (error estimate used as the standard error) pass by
construction. Ratios are taken to r-bar = mean encoder-mean norm of the full ZINC-250k set
(geometry root ``default/geometry.json``).
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import erf, erfc, gammaln

DRAWS = 2_000_000           # first Monte Carlo batch of a convex sampler (and the batch size)
MAX_DRAWS = 128_000_000     # cap of the adaptive convex Monte Carlo
CHECK_DRAWS = 200_000       # plain Monte Carlo kept beside every quadrature value
MC_SEED = 20261012
CHUNK = 50_000
NORM_DECIMALS = 2           # printed decimals of a mean norm
RATIO_DECIMALS = 3          # printed decimals of a norm / r-bar ratio
SE_FRACTION = 0.2           # SE <= 0.2 of the last printed unit
BOUNDARY_SE = 4.0           # value >= 4 SE away from the nearest rounding boundary

# (support id, index of its random stream, definition shown in tables)
SUPPORTS = [
    ("shell", 0, "uniform direction at $r$ = mean norm of the run's 5 cold-start encodings"),
    ("box_p01p99", 1, "uniform in the per-coordinate $[p_{01},p_{99}]$ box"),
    ("box_minmax", 2, "uniform in the per-coordinate $[\\min,\\max]$ box"),
    ("box_3sigma", 3, "uniform in the isotropic cube $[-3\\bar\\sigma, 3\\bar\\sigma]^D$"),
    ("prior", 4, "$N(0, I_D)$ (the VAE prior)"),
    ("gaussian_fit", 5, "$N(\\hat\\mu,\\hat\\Sigma)$ fitted to 3,000 encoded latents"),
    ("convex_k1", 6, "one random encoded latent ($k=1$)"),
    ("convex_k2", 7, "convex combination of $k=2$ encoded latents"),
    ("convex_k3", 8, "convex combination of $k=3$ encoded latents"),
    ("convex_k4", 9, "convex combination of $k=4$ encoded latents"),
    ("convex_k8", 10, "convex combination of $k=8$ encoded latents"),
    ("convex_k16", 11, "convex combination of $k=16$ encoded latents"),
    ("convex_k64", 12, "convex combination of $k=64$ encoded latents"),
    ("pca64", 13, "uniform direction in the top-64 PCA subspace, at $r$"),
    ("angular_box", 14, "box-shaped direction ($[-3\\bar\\sigma,3\\bar\\sigma]^D$) renormalised to $r$"),
]
SUPPORT_DEF = {s: d for s, _, d in SUPPORTS}
CONVEX_K = {"convex_k1": 1, "convex_k2": 2, "convex_k3": 3, "convex_k4": 4, "convex_k8": 8, "convex_k16": 16,
            "convex_k64": 64}
CONVEX_ARM = {"convex_k1": "convex_hull_k1", "convex_k2": "convex_hull_k2", "convex_k3": "convex_hull_rs",
              "convex_k4": "convex_hull_k4", "convex_k8": "convex_hull_k8", "convex_k16": "convex_hull_k16",
              "convex_k64": "convex_hull_k64"}


def _rng(index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([MC_SEED, index]))


# ---- printed digits ------------------------------------------------------------------------------
def digits_established(value: float, se: float, decimals: int) -> bool:
    """True iff ``value`` may be printed with ``decimals`` decimals: SE <= 0.2 of the last printed unit
    and the value at least 4 SE away from the nearest rounding boundary (half-unit point). The second
    condition is the stronger one (a value is never more than half a unit from a boundary, so it implies
    SE <= 0.125 of the unit); the first is kept because it is the stated requirement."""
    unit = 10.0 ** (-decimals)
    if se > SE_FRACTION * unit:
        return False
    frac = (value / unit) % 1.0                 # position inside the unit cell
    dist = abs(frac - 0.5) * unit               # distance to the rounding boundary
    return dist >= BOUNDARY_SE * se


def stable_decimals(value: float, se: float, max_decimals: int) -> int:
    for d in range(max_decimals, -1, -1):
        if digits_established(value, se, d):
            return d
    return 0


def stable_text(value: float, se: float, max_decimals: int) -> tuple[str, int]:
    d = stable_decimals(value, se, max_decimals)
    return f"{value:.{d}f}", d


# ---- exact E sqrt(S) for independent coordinates ---------------------------------------------------
def logphi_uniform(t: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """log prod_i E exp(-t x_i^2) for independent x_i ~ U[lo_i, hi_i]; t is a 1-D array."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    w = hi - lo
    out = np.zeros(t.shape, dtype=float)
    k = np.arange(1, 6)                                           # even moments E x^(2k), k = 1..5
    mom = (hi[None, :] ** (2 * k[:, None] + 1) - lo[None, :] ** (2 * k[:, None] + 1)) / ((2 * k[:, None] + 1) * w[None, :])
    c = np.maximum(lo * lo, hi * hi)                              # sup of x_i^2
    fact = np.array([1.0, 2.0, 6.0, 24.0, 120.0])
    for j, tj in enumerate(t):
        small = tj * c < 1e-2                                     # series: truncation (t x^2)^6/720 < 2e-15
        s = np.zeros(lo.shape)
        if small.any():
            ser = np.zeros(int(small.sum()))
            for i in range(5):
                ser += ((-1) ** i) * (tj ** (i + 1)) * mom[i][small] / fact[i]
            s[small] = np.log1p(-ser)
        big = ~small
        if big.any():
            rt = math.sqrt(tj)
            a, b = rt * lo[big], rt * hi[big]
            pos, neg = a >= 0, b <= 0
            d = np.empty(a.shape)
            d[pos] = erfc(a[pos]) - erfc(b[pos])
            d[neg] = erfc(-b[neg]) - erfc(-a[neg])
            mid = ~(pos | neg)
            d[mid] = erf(b[mid]) - erf(a[mid])
            with np.errstate(divide="ignore"):
                s[big] = np.log(d * math.sqrt(math.pi) / (2.0 * rt * w[big]))
        out[j] = s.sum()
    return out


def logphi_gaussian(t: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log prod_i E exp(-t x_i^2) for independent x_i ~ N(mean_i, var_i)."""
    tt = t[:, None]
    den = 1.0 + 2.0 * tt * var[None, :]
    return (-0.5 * np.log1p(2.0 * tt * var[None, :]) - tt * (mean[None, :] ** 2) / den).sum(1)


def esqrt_from_logphi(logphi, m2: float, h: float = 0.01, u_lo: float = -32.0, u_hi: float = 40.0) -> tuple[float, float]:
    """E sqrt(S) = (2 sqrt(pi))^-1 int_0^inf (1 - phi(t)) t^(-3/2) dt for S >= 0 with E S = m2 and
    Laplace transform phi; trapezoid rule in u = log t on [u_lo, u_hi] with the analytic tails
    2 m2 sqrt(t_lo) (1 - phi ~ t m2 as t -> 0) and 2 / sqrt(t_hi) (phi -> 0). Returns (value, error
    estimate = |value(h) - value(2h)|)."""
    def run(step):
        n = int(round((u_hi - u_lo) / step))
        u = u_lo + step * np.arange(n + 1)
        f = -np.expm1(logphi(np.exp(u))) * np.exp(-0.5 * u)
        core = step * (f.sum() - 0.5 * (f[0] + f[-1]))
        tails = 2.0 * m2 * math.exp(0.5 * u_lo) + 2.0 * math.exp(-0.5 * u_hi)
        return (core + tails) / (2.0 * math.sqrt(math.pi))
    v1, v2 = run(h), run(2.0 * h)
    return float(v1), float(abs(v1 - v2))


def box_exact(lo: np.ndarray, hi: np.ndarray) -> dict:
    """E||x||, SD||x|| for x uniform in the box (widths floored at 1e-6, as the solvers do)."""
    lo = np.asarray(lo, float)
    hi = lo + np.maximum(np.asarray(hi, float) - lo, 1e-6)
    m2 = float(((hi ** 3 - lo ** 3) / (3.0 * (hi - lo))).sum())
    v, err = esqrt_from_logphi(lambda t: logphi_uniform(t, lo, hi), m2)
    return {"mean": v, "sd": math.sqrt(max(m2 - v * v, 0.0)), "se": err, "exact": True, "method": "quadrature",
            "quadrature_error": err, "second_moment": m2, "draws": 0}


def gaussian_exact(mean: np.ndarray, L: np.ndarray) -> dict:
    """E||x|| for x = mean + L eps: in the eigenbasis of L L^T the coordinates are independent normals."""
    cov = L @ L.T
    lam, V = np.linalg.eigh(cov)
    lam = np.clip(lam, 0.0, None)
    mu = V.T @ mean
    m2 = float(lam.sum() + (mu ** 2).sum())
    v, err = esqrt_from_logphi(lambda t: logphi_gaussian(t, mu, lam), m2)
    return {"mean": v, "sd": math.sqrt(max(m2 - v * v, 0.0)), "se": err, "exact": True, "method": "quadrature",
            "quadrature_error": err, "second_moment": m2, "draws": 0}


# ---- plain Monte Carlo of the solvers' sampling rules (checks) -------------------------------------
def _summ(norms_chunks, draws: int) -> dict:
    x = np.concatenate(norms_chunks)
    assert x.size == draws
    sd = float(x.std(ddof=1))
    return {"mean": float(x.mean()), "sd": sd, "se": sd / math.sqrt(draws), "p05": float(np.quantile(x, 0.05)),
            "p95": float(np.quantile(x, 0.95)), "draws": int(draws), "exact": False}


def box_half_width(repo_root: str) -> tuple[float, str]:
    """BOX_SIGMA_MULTIPLIER * ZINC_PER_COORD_STD from src/analysis/config.py (parsed, not imported)."""
    p = os.path.join(repo_root, "src", "analysis", "config.py")
    s = open(p).read()
    std = float(re.search(r"^ZINC_PER_COORD_STD\s*=\s*([0-9.]+)", s, re.M).group(1))
    mult = float(re.search(r"^BOX_SIGMA_MULTIPLIER\s*=\s*([0-9.]+)", s, re.M).group(1))
    return mult * std, p


def shell_radius_mc(mu_norm: np.ndarray, draws: int, rng) -> dict:
    """r = mean of 5 norms drawn without replacement (rejection of repeated indices)."""
    out = []
    n = mu_norm.size
    left = draws
    while left:
        m = min(CHUNK * 4, left)
        idx = _distinct_rows(rng, n, 5, m)
        out.append(mu_norm[idx].mean(1))
        left -= m
    return _summ(out, draws)


def shell_radius_exact(mu_norm: np.ndarray, k: int = 5) -> dict:
    """Mean and SD of the mean of k draws without replacement from the population ``mu_norm``."""
    x = np.asarray(mu_norm, dtype=float)
    n = x.size
    var = x.var(ddof=0) / k * (n - k) / (n - 1)
    return {"mean": float(x.mean()), "sd": float(math.sqrt(var)), "se": 0.0, "exact": True, "method": "exact",
            "draws": 0}


def _distinct_rows(rng, n: int, k: int, m: int) -> np.ndarray:
    """m rows of k distinct indices in [0, n), uniform over ordered k-tuples of distinct indices
    (i.i.d. draws, rows with a repeat are redrawn: exact sampling without replacement)."""
    idx = rng.integers(0, n, size=(m, k))
    if k == 1:
        return idx
    while True:
        s = np.sort(idx, axis=1)
        bad = np.nonzero((s[:, 1:] == s[:, :-1]).any(1))[0]
        if bad.size == 0:
            return idx
        idx[bad] = rng.integers(0, n, size=(bad.size, k))


def box_mc(lo: np.ndarray, hi: np.ndarray, draws: int, rng) -> dict:
    w = np.maximum(hi - lo, 1e-6)
    out, left = [], draws
    while left:
        m = min(CHUNK, left)
        u = rng.random((m, lo.size))
        out.append(np.linalg.norm(lo + u * w, axis=1))
        left -= m
    return _summ(out, draws)


def gaussian_mc(mean: np.ndarray, L: np.ndarray, draws: int, rng) -> dict:
    out, left = [], draws
    while left:
        m = min(CHUNK, left)
        e = rng.standard_normal((m, mean.size))
        out.append(np.linalg.norm(mean + e @ L.T, axis=1))
        left -= m
    return _summ(out, draws)


def convex_mc(Z: np.ndarray, k: int, draws: int, rng) -> dict:
    """Plain Monte Carlo of ||sum_j w_j z_{a_j}|| (reference implementation, used by the tests)."""
    Z32 = Z.astype(np.float32)
    out, left = [], draws
    while left:
        m = min(CHUNK, left)
        idx = _distinct_rows(rng, Z.shape[0], k, m)
        g = rng.standard_exponential((m, k))
        w = (g / g.sum(1, keepdims=True)).astype(np.float32)      # Dirichlet(1, ..., 1)
        acc = np.zeros((m, Z.shape[1]), dtype=np.float64)
        for j in range(k):
            acc += w[:, j:j + 1] * Z32[idx[:, j]]
        out.append(np.linalg.norm(acc, axis=1))
        left -= m
    return _summ(out, draws)


# ---- convex combinations: control-variate Monte Carlo on the Gram matrix ----------------------------
def convex_second_moment(G: np.ndarray, k: int) -> float:
    """Exact E||x||^2 of x = sum_j w_j z_{a_j}: k distinct anchors, Dirichlet(1) weights.
    E w_j^2 = 2/(k(k+1)), E w_j w_l = 1/(k(k+1)); q = mean_a G_aa, c = mean_{a != b} G_ab."""
    n = G.shape[0]
    q = float(np.trace(G)) / n
    c = (float(G.sum()) - float(np.trace(G))) / (n * (n - 1))
    return (2.0 * q + (k - 1) * c) / (k + 1)


class _CVAccumulator:
    """Running sums for the control-variate estimator mean(y) - beta (mean(s) - E s)."""

    def __init__(self, es: float):
        self.es = es
        self.n = 0
        self.sy = self.ss = self.syy = self.sss = self.sys = 0.0
        self.q = []         # a thinned sample of y for the quantiles

    def add(self, y: np.ndarray, s: np.ndarray) -> None:
        self.n += y.size
        self.sy += float(y.sum())
        self.ss += float(s.sum())
        self.syy += float((y * y).sum())
        self.sss += float((s * s).sum())
        self.sys += float((y * s).sum())
        self.q.append(y[::16].copy())

    def result(self) -> dict:
        n = self.n
        my, ms = self.sy / n, self.ss / n
        vy = (self.syy - n * my * my) / (n - 1)
        vs = (self.sss - n * ms * ms) / (n - 1)
        cys = (self.sys - n * my * ms) / (n - 1)
        beta = cys / vs if vs > 0 else 0.0
        resid_var = max(vy - beta * cys, 0.0)
        q = np.concatenate(self.q)
        return {"mean": my - beta * (ms - self.es), "sd": math.sqrt(vy), "se": math.sqrt(resid_var / n),
                "plain_mean": my, "plain_se": math.sqrt(vy / n), "beta": beta, "second_moment": self.es,
                "second_moment_mc": ms, "second_moment_mc_se": math.sqrt(vs / n),
                "p05": float(np.quantile(q, 0.05)), "p95": float(np.quantile(q, 0.95)),
                "draws": int(n), "exact": False, "method": "control_variate_mc"}


GRAM_MAX_K = 8      # k <= 8: ||x||^2 from k(k+1)/2 Gram entries per draw; larger k: the weighted sum itself


def convex_cv_batch(Z: np.ndarray, G: np.ndarray, k: int, draws: int, rng, acc: _CVAccumulator) -> None:
    """Add ``draws`` draws of (||x||, ||x||^2) to ``acc``. For k <= GRAM_MAX_K,
    ||x||^2 = sum_{j,l} w_j w_l G[a_j, a_l] (G = Z Z^T); for larger k the combination is formed in float64."""
    left = draws
    n = Z.shape[0]
    chunk = CHUNK * 4 if k <= GRAM_MAX_K else CHUNK
    while left:
        m = min(chunk, left)
        idx = _distinct_rows(rng, n, k, m)
        g = rng.standard_exponential((m, k))
        w = g / g.sum(1, keepdims=True)                           # Dirichlet(1, ..., 1)
        if k <= GRAM_MAX_K:
            s = np.zeros(m)
            for j in range(k):
                s += w[:, j] ** 2 * G[idx[:, j], idx[:, j]]
                for l_ in range(j + 1, k):
                    s += 2.0 * w[:, j] * w[:, l_] * G[idx[:, j], idx[:, l_]]
            np.maximum(s, 0.0, out=s)
        else:
            # x = W Z with W the (m x n) weight matrix that has k non-zeros per row (one sparse product)
            W = csr_matrix((w.ravel(), idx.ravel(), np.arange(0, m * k + 1, k)), shape=(m, n))
            x = W @ Z
            s = np.einsum("ij,ij->i", x, x)
        acc.add(np.sqrt(s), s)
        left -= m


def convex_cv(Z: np.ndarray, G: np.ndarray, k: int, rng, rbar: float, draws: int = DRAWS,
              max_draws: int = MAX_DRAWS, log=None) -> dict:
    """Control-variate Monte Carlo of E||x|| for the convex sampler, adding batches of ``draws`` draws from
    the support's fixed stream until the norm (NORM_DECIMALS) and the ratio to r-bar (RATIO_DECIMALS) have
    every printed digit established, or ``max_draws`` is reached."""
    acc = _CVAccumulator(convex_second_moment(G, k))
    while True:
        convex_cv_batch(Z, G, k, draws, rng, acc)
        r = acc.result()
        ok = (digits_established(r["mean"], r["se"], NORM_DECIMALS)
              and digits_established(r["mean"] / rbar, r["se"] / rbar, RATIO_DECIMALS))
        if ok or acc.n + draws > max_draws:
            r["digits_established"] = bool(ok)
            return r
        if log and (acc.n // draws) % 8 == 0:
            log(f"      convex k={k}: {acc.n:,} draws, mean {r['mean']:.6f} (SE {r['se']:.1e}) -- adding batches")


def prior_exact(D: int) -> float:
    return float(math.sqrt(2.0) * math.exp(gammaln((D + 1) / 2.0) - gammaln(D / 2.0)))


def _check(name: str, exact: dict, mc: dict) -> dict:
    """Attach the plain Monte Carlo check of a quadrature/exact value; a gap beyond 5 SE is an error."""
    z = (mc["mean"] - exact["mean"]) / mc["se"] if mc["se"] > 0 else 0.0
    if abs(z) > 5.0:
        raise RuntimeError(f"support {name}: exact value {exact['mean']:.6f} and its Monte Carlo check "
                           f"{mc['mean']:.6f} +- {mc['se']:.1e} disagree (z = {z:.1f})")
    if exact.get("method") == "quadrature" and exact.get("quadrature_error", 0.0) > 1e-8:
        raise RuntimeError(f"support {name}: quadrature error estimate {exact['quadrature_error']:.1e} > 1e-8")
    exact["mc_check"] = {"mean": mc["mean"], "se": mc["se"], "draws": mc["draws"], "z": z}
    exact.setdefault("p05", mc["p05"])
    exact.setdefault("p95", mc["p95"])
    return exact


def compute_supports(latent_npz: str, bounds_json: str, geometry_default: str, mu_norm_npy: str | None,
                     repo_root: str, draws: int = DRAWS, log=print, check_draws: int = CHECK_DRAWS,
                     max_draws: int = MAX_DRAWS) -> dict:
    """{support id: {mean, sd, se, p05, p95, draws, exact, method, seed_index, ratio, ratio_se, text_norm,
    text_ratio, inputs, definition}}, plus '_meta'. Every input path is listed so the dossier can hash it."""
    z = np.load(latent_npz)
    if "Z" not in z.files:
        raise RuntimeError(f"{latent_npz}: no 'Z' array (the solvers read ['Z'])")
    Z = np.asarray(z["Z"], dtype=np.float64)
    D = Z.shape[1]
    B = json.load(open(bounds_json))["per_coord"]
    g = json.load(open(geometry_default))
    rbar = float(g["mu_cloud"]["norm"]["mean"])
    h, cfg_path = box_half_width(repo_root)
    out: dict = {}
    seed_of = {s: i for s, i, _ in SUPPORTS}

    def put(sid, res, inputs):
        res.update(seed_index=seed_of[sid], definition=SUPPORT_DEF[sid], inputs=sorted(set(inputs)),
                   ratio=res["mean"] / rbar, ratio_se=res["se"] / rbar, ratio_sd=res["sd"] / rbar)
        res["text_norm"], res["dec_norm"] = stable_text(res["mean"], res["se"], NORM_DECIMALS)
        res["text_ratio"], res["dec_ratio"] = stable_text(res["ratio"], res["ratio_se"], RATIO_DECIMALS)
        out[sid] = res
        log(f"    support {sid:14s} mean {res['mean']:.6f} ({res.get('method', 'mc')}, SE/err {res['se']:.1e}, "
            f"{res['draws']:,} draws) prints {res['text_norm']} ratio {res['text_ratio']}")

    if mu_norm_npy and os.path.exists(mu_norm_npy):
        mn = np.asarray(np.load(mu_norm_npy), dtype=float)
        sh = _check("shell", shell_radius_exact(mn), shell_radius_mc(mn, check_draws, _rng(seed_of["shell"])))
        put("shell", sh, [mu_norm_npy, geometry_default])
        for sid in ("pca64", "angular_box"):          # same radius rule: every candidate sits at r
            r = dict(out["shell"])
            r["same_as"] = "shell"
            put(sid, r, [mu_norm_npy, geometry_default])
    else:
        log("    support shell: mu_norm.npy absent -- shell, PCA-64 and box-direction rows get no value")
    for sid, lo_k, hi_k in (("box_p01p99", "p01", "p99"), ("box_minmax", "min", "max")):
        lo, hi = np.asarray(B[lo_k], float), np.asarray(B[hi_k], float)
        put(sid, _check(sid, box_exact(lo, hi), box_mc(lo, hi, check_draws, _rng(seed_of[sid]))),
            [bounds_json, geometry_default])
    lo, hi = np.full(D, -h), np.full(D, h)
    put("box_3sigma", _check("box_3sigma", box_exact(lo, hi), box_mc(lo, hi, check_draws, _rng(seed_of["box_3sigma"]))),
        [cfg_path, geometry_default])
    pr_mc = gaussian_mc(np.zeros(D), np.eye(D), check_draws, _rng(seed_of["prior"]))
    pr = {"mean": prior_exact(D), "sd": math.sqrt(max(D - prior_exact(D) ** 2, 0.0)), "se": 0.0, "exact": True,
          "method": "exact", "draws": 0}
    pr = _check("prior", pr, pr_mc)
    pr["mc_mean"], pr["mc_se"] = pr_mc["mean"], pr_mc["se"]
    put("prior", pr, [geometry_default])
    mean = Z.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(Z, rowvar=False))
    L = evecs * np.sqrt(np.clip(evals, 1e-12, None))
    put("gaussian_fit", _check("gaussian_fit", gaussian_exact(mean, L),
                               gaussian_mc(mean, L, check_draws, _rng(seed_of["gaussian_fit"]))),
        [latent_npz, geometry_default])
    nz = np.linalg.norm(Z, axis=1)
    put("convex_k1", {"mean": float(nz.mean()), "sd": float(nz.std(ddof=1)), "se": 0.0,
                      "p05": float(np.quantile(nz, 0.05)), "p95": float(np.quantile(nz, 0.95)),
                      "draws": 0, "exact": True, "method": "exact"}, [latent_npz, geometry_default])
    G = Z @ Z.T
    for sid, k in CONVEX_K.items():
        if k == 1:
            continue
        put(sid, convex_cv(Z, G, k, _rng(seed_of[sid]), rbar, draws=draws, max_draws=max_draws, log=log),
            [latent_npz, geometry_default])
    out["_meta"] = {"draws": draws, "max_draws": max_draws, "check_draws": check_draws, "seed": MC_SEED,
                    "rbar": rbar, "rbar_source": geometry_default,
                    "n_latents": int(Z.shape[0]), "latent_npz": latent_npz, "bounds": bounds_json,
                    "box_half_width": h, "config_py": cfg_path, "D": D,
                    "digit_rule": f"SE <= {SE_FRACTION} of the last printed unit and >= {BOUNDARY_SE:g} SE from the "
                                  f"nearest rounding boundary, else one decimal fewer",
                    "unestablished": sorted(s for s, v in out.items()
                                            if not s.startswith("_") and v.get("digits_established") is False)}
    return out
