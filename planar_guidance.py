"""Adapted from the supplied lowthrust.py; original source is bundled with the transfer films."""
import numpy as np
from numba import njit

MU_E, MU_S = 3.986004418e14, 1.32712440018e20
R_E, AU, G0 = 6378.137e3, 1.495978707e11, 9.80665

@njit(cache=True)
def _elements(x, y, vx, vy, mu):
    r = np.hypot(x, y); v2 = vx * vx + vy * vy
    a = 1.0 / (2.0 / r - v2 / mu)
    h = x * vy - y * vx
    ex = vy * h / mu - x / r; ey = -vx * h / mu - y / r
    e = np.hypot(ex, ey)
    nu = np.arctan2(y, x) - np.arctan2(ey, ex) if e > 1e-9 else np.arctan2(y, x)
    return r, a, e, nu, h

@njit(cache=True)
def _grad(a, e, nu, aT, W, mu):
    """dQ/dt per unit acceleration in (radial, tangential) directions."""
    p = a * (1.0 - e * e); h = np.sqrt(mu * p); r = p / (1.0 + e * np.cos(nu))
    dQa = 2.0 * (a - aT) / (aT * aT); dQe = 2.0 * W * e
    da_r = 2.0 * a * a / h * e * np.sin(nu); da_t = 2.0 * a * a / h * p / r
    de_r = p * np.sin(nu) / h; de_t = ((p + r) * np.cos(nu) + r * e) / h
    return dQa * da_r + dQe * de_r, dQa * da_t + dQe * de_t

@njit(cache=True)
def _accel(s, mu, aT, W, eta, T0, mdot0, rref, k):
    x, y, vx, vy, m = s
    r, a, e, nu, h = _elements(x, y, vx, vy, mu)
    gx = -mu * x / r ** 3; gy = -mu * y / r ** 3
    Ar, At = _grad(a, e, nu, aT, W, mu)
    mag = np.hypot(Ar, At)
    best = 0.0
    for j in range(36):                                     # best achievable rate on this orbit
        ar_, at_ = _grad(a, e, 2.0 * np.pi * j / 36.0, aT, W, mu)
        best = max(best, np.hypot(ar_, at_))
    fire = mag >= eta * best and mag > 0.0
    scale = min(1.0, (rref / r) ** k)
    if not fire:
        return np.array([vx, vy, gx, gy, 0.0]), False
    ur, ut = -Ar / mag, -At / mag
    rx, ry = x / r, y / r; tx, ty = -ry, rx                   # radial and local horizontal (prograde for h > 0)
    if h < 0: tx, ty = -tx, -ty
    T = T0 * scale
    ax = T / m * (ur * rx + ut * tx); ay = T / m * (ur * ry + ut * ty)
    return np.array([vx, vy, gx + ax, gy + ay, -mdot0 * scale]), True

@njit(cache=True)
def _run(s0, mu, aT, W, eta, T0, mdot0, rref, k, t_max, m_dry_min, samples_per_rev, max_out, min_radius):
    s = s0.copy(); t = 0.0
    out = np.zeros((max_out + 1, 7)); n = 0; next_sample = 0.0
    status = 0
    steps = 0
    while True:
        x, y, vx, vy, m = s
        r, a, e, nu, h = _elements(x, y, vx, vy, mu)
        if not np.isfinite(a) or a <= 0 or e >= 1 or r <= min_radius:
            status = 4; break
        Torb = 2.0 * np.pi * np.sqrt(abs(a) ** 3 / mu)
        if t >= next_sample and n < max_out:
            out[n, 0] = t; out[n, 1:6] = s; out[n, 6] = a; n += 1
            next_sample = t + max(t_max / (max_out-1), Torb / samples_per_rev)
        if abs(a - aT) / aT < 1e-2 and e < 1e-2:
            status = 1; break
        if m <= m_dry_min + 1e-9:
            status = 2; break
        if t >= t_max:
            status = 3; break
        if steps >= 1500000:
            status = 5; break
        steps += 1
        v = np.hypot(vx, vy)
        dt = min(Torb / 240.0, 2e-3 * v / max(T0 / m, 1e-12))
        dt = min(dt, t_max-t, .01*m/mdot0)
        if dt < 1e-8:
            status = 2 if m-m_dry_min < 1e-8 else 3
            break
        k1, f = _accel(s, mu, aT, W, eta, T0, mdot0, rref, k)
        k2, _ = _accel(s + 0.5 * dt * k1, mu, aT, W, eta, T0, mdot0, rref, k)
        k3, _ = _accel(s + 0.5 * dt * k2, mu, aT, W, eta, T0, mdot0, rref, k)
        k4, _ = _accel(s + dt * k3, mu, aT, W, eta, T0, mdot0, rref, k)
        increment = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if m + increment[4] < m_dry_min:
            fraction = (m-m_dry_min)/(-increment[4])
            s += increment * fraction
            s[4] = m_dry_min
            t += dt * fraction
        else:
            s += increment
            t += dt
    out[n, 0] = t; out[n, 1:6] = s; out[n, 6] = a; n += 1
    return out[:n], status

