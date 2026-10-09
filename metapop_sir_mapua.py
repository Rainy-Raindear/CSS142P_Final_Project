#!/usr/bin/env python3
"""
Spatial Metapopulation SIR Model of Seasonal Influenza Spread Across
High-Contact Architectural Nodes (Floors 1-3), Mapua University - Makati
---------------------------------------------------------------------------
Modeling and Simulation Theory | Engr. Joel C. De Goma

Nodes (blueprints A-103/A-104/A-105):
  1F = turnstile lobby, student services, canteen
  2F = lecture rooms, Chem/Micro/Physics labs, SOIT & Math-Physics offices
  3F = library, group study rooms, computer laboratories

State y = [S1,S2,S3, I1,I2,I3, R1,R2,R3]  (9 ODEs)
    dS_k/dt = - sum_j beta_kj(t) S_k I_j / N_k
    dI_k/dt = + sum_j beta_kj(t) S_k I_j / N_k - gamma I_k
    dR_k/dt = + gamma I_k
    beta_kj(t) = p_trans * c_kj * phi_k(t) * o_j(t)
    phi_k(t)   = (rho_k * o_k(t) / ACH_k) * tau_k(t)          (normalised)

NO TIMETABLES OR SURVEYS ARE AVAILABLE, so contact timing is generated from
a schedule-based occupancy assumption (all marked [ASSUMED]):
  * Campus opens 06:00; college classes run 07:30-21:00; building empties by 22:00.
  * Students arrive from 06:00 and wait in the 1F lobby before 07:30 classes.
  * SHS classes assumed to end 17:00; after that SHS rooms are used by
    college students, so 2F/3F room utilisation stays high in the evening.
  * Operating days Mon-Sat; Sunday closed. t = 0 is Monday 00:00.
  * Transmission occurs only while the building is open; at night people are
    off-campus (no transmission in this model) but still recover at rate gamma.
o_k(t) = fraction of floor k's peak occupancy present at hour t.

Run:  python metapop_sir_mapua.py     (numpy, scipy, matplotlib)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------ 1. BASELINE
FLOORS = ["1F Lobby/Canteen", "2F Lecture & Science Labs", "3F Library & Computer Labs"]
N = np.array([1200.0, 450.0, 600.0])      # [ASSUMED] peak occupants (also the patch population)
C = np.array([[10.0, 2.5, 1.5],           # [ASSUMED] peak contact rates (per day-equivalent);
              [ 2.0, 6.0, 2.0],           # off-diagonals = class-changeover flows via stairs/elevators
              [ 1.0, 2.0, 12.0]])
RHO = np.array([1.00, 0.40, 0.90])        # [ASSUMED] peak persons / m^2
ACH = np.array([3.0, 6.0, 4.0])           # [ASSUMED] effective air changes / h
TAU = np.array([0.8, 2.0, 2.5])           # [ASSUMED] dwell time per visit (h)
GAMMA = 1.0 / 4.0                         # D = 4 days
R0_TARGET = 1.8
T_END = 60                                # days
OPERATING_DAYS = {0, 1, 2, 3, 4, 5}       # Mon..Sat ; Sunday (6) closed
SHS_ROOMS_REUSED = True                   # SHS rooms used by college after 17:00
MEALS = [11, 12, 13, 17, 18]              # canteen meal windows (hour-start)
LOBBY_WAIT = {6: 1.6, 7: 1.3}             # [ASSUMED] longer lobby dwell while waiting for 07:30 classes

HOURS = np.arange(6, 22)                  # hour-start bins 06:00 ... 21:00-22:00

def baseline_profiles(reused=SHS_ROOMS_REUSED):
    """o_k(h): fraction of peak occupancy present, hours 06..21 [ASSUMED shapes]."""
    #            6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21
    O1 = [.15, .55, .40, .35, .45, .80, 1.0, .90, .50, .40, .45, .65, .70, .45, .50, .65]
    O2d = [.02, .30, .85, .90, .90, .85, .65, .85, .90, .90, .85]            # 06-16
    O3d = [.00, .35, .80, .90, .90, .85, .70, .85, .90, .90, .85]
    O2e = np.array([1.0, 1.0, .95, .80, .30])                                 # 17-21 shape
    O3e = np.array([.95, .90, .75, .55, .25])                                 # library shuts 20:00
    u2, u3 = (0.85, 0.85) if reused else (0.45, 0.50)                         # evening room utilisation
    return np.array([O1, O2d + list(u2 * O2e), O3d + list(u3 * O3e)])

def shave(O, s):
    """Staggered schedule: flatten peaks about the open-hours mean (person-hours kept)."""
    out = O.copy()
    for k in range(3):
        op = O[k] > 0.05
        m = O[k][op].mean()
        out[k][op] = m + (1 - s) * (O[k][op] - m)
    return out

RAW_REF = ((RHO / ACH) * TAU).mean()      # fixed normalisation of phi across scenarios

def hourly_beta(spec):
    """Return beta_kj for each open hour (unit p_trans) under a scenario spec."""
    O = baseline_profiles(spec.get("reused", SHS_ROOMS_REUSED))
    if spec.get("stagger"):
        O = shave(O, 0.40)
    rho_t = RHO[:, None] * O
    tau_t = np.repeat(TAU[:, None], len(HOURS), axis=1).astype(float)
    for h, m in LOBBY_WAIT.items():
        tau_t[0, h - 6] *= m
    if spec.get("canteen_cap"):                       # grab-and-go cap in meal windows
        for h in MEALS:
            rho_t[0, h - 6] *= 0.60
            tau_t[0, h - 6] *= 0.60
    ach = ACH.copy()
    if spec.get("hepa"):
        ach[2] = 10.0                                  # effective ACH incl. HEPA CADR
    phi_t = (rho_t / ach[:, None]) * tau_t / RAW_REF
    Cm = C * (0.90 if spec.get("stagger") else 1.0)    # staggered changeovers: fewer crowded stair flows
    return np.array([Cm * phi_t[:, i][:, None] * O[:, i][None, :]
                     for i in range(len(HOURS))]), O

def weekly_mean_beta(beta_h):
    """Time-averaged beta (per day) over a Mon-Sat-open week; used for R0 (period << 1/gamma)."""
    return beta_h.sum(axis=0) / 24.0 * len(OPERATING_DAYS) / 7.0

def r0_of(B):
    return np.max(np.abs(np.linalg.eigvals(B / GAMMA)))

SCENARIOS = {"SC-0": ("Baseline", {}),
             "SC-1": ("1F canteen cap", {"canteen_cap": True}),
             "SC-2": ("3F lab/library HEPA", {"hepa": True}),
             "SC-3": ("Staggered schedule", {"stagger": True}),
             "SC-4": ("Combined NPI", {"canteen_cap": True, "hepa": True, "stagger": True})}
SEEDS = {"1F cafeteria": 0, "3F lab": 2}

# p_trans calibrated ONCE on the baseline so weekly-mean R0 = R0_TARGET
P_TRANS = R0_TARGET / r0_of(weekly_mean_beta(hourly_beta({})[0]))

# ------------------------------------------------------------------ 2. ODEs
def sir_rhs(t, y, beta, N, gamma):
    S, I, R = y[0:3], y[3:6], y[6:9]
    new_inf = S * (beta @ I) / N
    return np.concatenate([-new_inf, new_inf - gamma * I, gamma * I])

def initial_state(seed_patch, n_seed=1.0):
    I0 = np.zeros(3); I0[seed_patch] = n_seed
    return np.concatenate([N - I0, I0, np.zeros(3)])

def simulate(beta_h, seed_patch, suspend=None):
    """Integrate with RK45 segment by segment: beta(t) is piecewise-constant per
    hour (06-22 on operating days), zero overnight/Sunday."""
    zero = np.zeros((3, 3)); y = initial_state(seed_patch)
    ts, ys = [0.0], [y]; max_dt = 0.0; max_drift = 0.0
    def seg(t0, t1, beta):
        nonlocal y, max_dt, max_drift
        s = solve_ivp(sir_rhs, (t0, t1), y, method="RK45", t_eval=[t1],
                      args=(beta, N, GAMMA), rtol=1e-8, atol=1e-10)
        y = s.y[:, -1]
        max_dt = max(max_dt, abs(sir_rhs(t1, y, beta, N, GAMMA).sum()))        # d/dt sum(S+I+R)
        max_drift = max(max_drift, np.abs(y[0:3] + y[3:6] + y[6:9] - N).max())
        ts.append(t1); ys.append(y)
    for d in range(T_END):
        if d % 7 not in OPERATING_DAYS or (suspend and suspend[0] <= d < suspend[1]):
            seg(d, d + 1, zero); continue
        seg(d, d + 6 / 24, zero)                                              # 00:00-06:00 closed
        for i in range(len(HOURS)):
            seg(d + HOURS[i] / 24, d + (HOURS[i] + 1) / 24, P_TRANS * beta_h[i])
        seg(d + 22 / 24, d + 1, zero)                                         # 22:00-24:00 closed
    return np.array(ts), np.array(ys).T, max_dt, max_drift

# ------------------------------------------------------------------ 3. RUN
def main():
    print(f"gamma={GAMMA:.2f}/day  p_trans={P_TRANS:.4f}  SHS rooms reused after 17:00: {SHS_ROOMS_REUSED}")
    res, profiles = {}, {}
    for sc, (label, spec) in SCENARIOS.items():
        beta_h, O = hourly_beta(spec); profiles[sc] = O
        print(f"{sc} ({label}): weekly-mean R0 = {r0_of(P_TRANS * weekly_mean_beta(beta_h)):.3f}")
        for sname, sp in SEEDS.items():
            t, y, dmax, drift = simulate(beta_h, sp)
            I = y[3:6]; tot = I.sum(0)
            res[(sc, sname)] = dict(t=t, I=I, peak=tot.max(), t_peak=t[tot.argmax()],
                                    cases=N.sum() - y[0:3, -1].sum(), cases_k=N - y[0:3, -1])
            r = res[(sc, sname)]
            print(f"   seed {sname:13s}| max|d/dt sum|={dmax:.1e} drift={drift:.1e} | "
                  f"peak I={r['peak']:.1f} (day {r['t_peak']:.1f}) | cumulative cases={r['cases']:.0f} "
                  f"(1F/2F/3F = {r['cases_k'][0]:.0f}/{r['cases_k'][1]:.0f}/{r['cases_k'][2]:.0f})")
            assert dmax < 1e-9 and drift < 1e-6, "mass conservation violated"

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3, "font.family": "serif"})
    col = {"SC-0": "k", "SC-1": "tab:blue", "SC-2": "tab:red", "SC-3": "tab:green", "SC-4": "tab:purple"}

    # Fig 1: I_k(t) per floor
    for sname in SEEDS:
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
        for k, ax in enumerate(axs):
            for sc, (label, _) in SCENARIOS.items():
                r = res[(sc, sname)]
                ax.plot(r["t"], r["I"][k], color=col[sc], lw=1.6,
                        ls="-" if sc in ("SC-0", "SC-1", "SC-2") else "--", label=f"{sc} {label}")
            ax.set_title(FLOORS[k]); ax.set_xlabel("Time (days)")
        axs[0].set_ylabel("Active infections $I_k(t)$"); axs[0].legend(fontsize=7)
        fig.suptitle(f"Index case seeded in {sname}", y=1.02); fig.tight_layout()
        fig.savefig(f"fig1_timeseries_seed_{sname.split()[0]}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Fig 2: peak comparison
    fig, ax = plt.subplots(figsize=(8, 4.2)); w = .38; xs = np.arange(len(SCENARIOS))
    for i, sname in enumerate(SEEDS):
        b = ax.bar(xs + (i - .5) * w, [res[(sc, sname)]["peak"] for sc in SCENARIOS], w,
                   label=f"Seed: {sname}", edgecolor="k", alpha=.85)
        ax.bar_label(b, fmt="%.0f", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{sc}\n{v[0]}" for sc, v in SCENARIOS.items()], fontsize=8)
    ax.set_ylabel("Peak simultaneous infections (1F+2F+3F)")
    ax.set_title("Comparative peak burden by intervention scenario"); ax.legend()
    fig.tight_layout(); fig.savefig("fig2_peak_comparison.png", dpi=300); plt.close(fig)

    # Fig 3: assumed daily occupancy profiles (baseline vs staggered)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for k, ax in enumerate(axs):
        ax.step(HOURS, profiles["SC-0"][k], where="post", color="k", label="Baseline")
        ax.step(HOURS, profiles["SC-3"][k], where="post", color="tab:green", ls="--", label="Staggered (SC-3)")
        for x, lab in [(7.5, "classes 07:30"), (17, "SHS ends 17:00"), (21, "classes end 21:00")]:
            ax.axvline(x, color="grey", lw=.8, ls=":"); ax.text(x + .1, 1.02, lab, rotation=90, fontsize=7, va="bottom")
        ax.set_title(FLOORS[k], pad=34); ax.set_xlabel("Hour of day"); ax.set_xlim(6, 22)
    axs[0].set_ylabel("Fraction of peak occupancy $o_k$"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("fig3_occupancy_profiles.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    extra_analyses()
    print("Saved fig1_*, fig2_*, fig3_*, fig4_* PNG files")

# ------------------------------------------------------------------ 4. EXTRA ANALYSES
def extra_analyses():
    """(a) SHS-room-reuse switch, (b) blanket 14-day suspension benchmark,
    (c) one-at-a-time +/-30% sensitivity of R0 (tornado)."""
    global C, RHO, ACH, TAU
    base_beta, _ = hourly_beta({})
    R0b = r0_of(P_TRANS * weekly_mean_beta(base_beta))
    R0_nr = r0_of(P_TRANS * weekly_mean_beta(hourly_beta({"reused": False})[0]))
    print(f"\n[a] SHS rooms NOT reused after 17:00: R0 = {R0_nr:.3f} (baseline {R0b:.3f})")

    print("[b] Blanket suspension days 21-35 (Mon-Sat classes lost = 12):")
    for sname, sp in SEEDS.items():
        t, y, _, _ = simulate(base_beta, sp, suspend=(21, 35))
        tot = y[3:6].sum(0)
        print(f"    seed {sname:13s}| peak I={tot.max():.1f} (day {t[tot.argmax()]:.1f}) "
              f"| cumulative cases={N.sum() - y[0:3, -1].sum():.0f}")

    C0, R0_, A0, T0 = C.copy(), RHO.copy(), ACH.copy(), TAU.copy()
    def r0_mod(Cm=None, rho=None, ach=None, tau=None):
        global C, RHO, ACH, TAU
        C = C0.copy() if Cm is None else Cm; RHO = R0_.copy() if rho is None else rho
        ACH = A0.copy() if ach is None else ach; TAU = T0.copy() if tau is None else tau
        r = r0_of(P_TRANS * weekly_mean_beta(hourly_beta({})[0]))
        C, RHO, ACH, TAU = C0.copy(), R0_.copy(), A0.copy(), T0.copy()
        return r
    def scaled(v, k, f): w = v.copy(); w[k] *= f; return w
    items = []
    for k in range(3):
        lo = C0.copy(); lo[k, k] *= .7
        hi = C0.copy(); hi[k, k] *= 1.3
        items.append((f"c{k+1}{k+1} within-floor contact", r0_mod(Cm=lo), r0_mod(Cm=hi)))
    off = np.eye(3) == 0
    items.append(("off-diagonal c (inter-floor)", r0_mod(Cm=np.where(off, C0 * .7, C0)),
                  r0_mod(Cm=np.where(off, C0 * 1.3, C0))))
    for k in range(3):
        items.append((f"rho {k+1}F density", r0_mod(rho=scaled(R0_, k, .7)), r0_mod(rho=scaled(R0_, k, 1.3))))
        items.append((f"ACH {k+1}F", r0_mod(ach=scaled(A0, k, 1.3)), r0_mod(ach=scaled(A0, k, .7))))
        items.append((f"tau {k+1}F dwell time", r0_mod(tau=scaled(T0, k, .7)), r0_mod(tau=scaled(T0, k, 1.3))))
    items.sort(key=lambda x: abs(x[2] - x[1]))
    print("[c] R0 sensitivity (+/-30% one at a time, p_trans fixed):")
    for lab, lo, hi in items[::-1]:
        print(f"    {lab:32s} {lo:.2f} .. {hi:.2f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (lab, lo, hi) in enumerate(items):
        ax.barh(i, hi - R0b, left=R0b, color="tab:red", alpha=.8)
        ax.barh(i, lo - R0b, left=R0b, color="tab:blue", alpha=.8)
    ax.axvline(R0b, color="k", lw=1); ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=8)
    ax.set_xlabel("Weekly-mean $R_0$ (dashed line: $R_0=1$)")
    ax.set_title("One-at-a-time sensitivity of $R_0$ to $\\pm30\\%$ parameter changes")
    fig.tight_layout(); fig.savefig("fig4_r0_sensitivity.png", dpi=300); plt.close(fig)

if __name__ == "__main__":
    main()
