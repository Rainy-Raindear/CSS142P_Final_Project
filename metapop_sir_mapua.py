#!/usr/bin/env python3
"""
Spatial Metapopulation SIR Model of Seasonal Influenza Spread Across
High-Contact Architectural Nodes (Floors 1-3), Mapua University - Makati
---------------------------------------------------------------------------
Modeling and Simulation Theory | Engr. Joel C. De Goma

Scope reduced from GF-6F to GF-3F (blueprints A-103/A-104/A-105). The ground
floor is labelled "1F" below.
  1F = turnstile lobby, student services, canteen
  2F = lecture rooms, Chem/Micro/Physics labs, SOIT & Math-Physics offices, SHS rooms
  3F = library, group study rooms, computer laboratories

State y = [S1,S2,S3, I1,I2,I3, R1,R2,R3]  (9 ODEs)
    dS_k/dt = - S_k * sum_j beta_kj(t) I_j / N_j
    dI_k/dt = + S_k * sum_j beta_kj(t) I_j / N_j - gamma I_k
    dR_k/dt = + gamma I_k
    beta_kj(t) = p_trans * c_kj * phi_k(t) * o_j(t)
    phi_k(t)   = rho_k (o_k(t) + H_k(t)/N_k) * tau_k(t) * eta_k(t) / ACH_k   (normalised)
c_kj = peak contacts per day a floor-k student has with floor-j students, so the
force of infection uses the prevalence I_j/N_j of the source floor.

NO TIMETABLES OR SURVEYS ARE AVAILABLE, so contact timing is generated from
a schedule-based occupancy assumption (all marked [ASSUMED]):
  * Campus opens 06:00; college classes run 07:30-21:00; building empties by 22:00.
  * Students arrive from 06:00 and wait in the 1F lobby before 07:30 classes.
  * SHS classes run 07:30-17:00 in 2F SHS rooms. After 17:00 those rooms are
    EMPTY (not used by college), so 2F evening occupancy drops.
  * SHS students are outside the study population (never S/I/R) but add
    crowding H_k(t) to 1F (arrival, lunch, dismissal) and 2F (classes).
  * Operating days Mon-Sat; Sunday closed. t = 0 is Monday 00:00.
  * Transmission occurs only while the building is open; at night people are
    off-campus (no transmission in this model) but still recover at rate gamma.
o_k(t) = fraction of floor k's peak college occupancy present at hour t.

Run:  python metapop_sir_mapua.py     (numpy, scipy, matplotlib)
"""
import csv
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ------------------------------------------------------------------ 1. BASELINE
FLOORS = ["1F Lobby/Canteen", "2F Lecture & Science Labs", "3F Library & Computer Labs"]
N = np.array([1200.0, 450.0, 600.0])      # [ASSUMED] peak college occupants (also the patch population)
C = np.array([[12.0, 2.0, 2.0],           # [ASSUMED] peak contact rates (per day-equivalent);
              [ 2.5, 7.0, 1.0],           # off-diagonals = lobby/canteen passage and class-changeover
              [ 2.5, 1.0, 6.0]])          # flows via stairs/elevators; 3F library is low-talk
RHO = np.array([0.80, 0.60, 0.40])        # [ASSUMED] peak persons / m^2 (canteen, classroom, library/lab)
ACH = np.array([4.0, 3.5, 2.0])           # [ASSUMED] effective air changes / h (2F lab hoods, 3F sealed)
TAU = np.array([0.75, 1.5, 2.0])          # [ASSUMED] dwell time per visit (h): meal, class block, lab/study
ETA_MEAL = 2.0                            # [ASSUMED] emission/proximity factor, unmasked eating in canteen
SHS_N = 300.0                             # [ASSUMED] SHS headcount present on campus (not modelled as S/I/R)
GAMMA = 1.0 / 4.0                         # infectious period D = 4 days (Carrat et al., 2008)
R0_TARGET = 1.8                           # dense-indoor planning value; 1.3 and 2.2 tested in [d]
R0_RANGE = [1.3, 1.8, 2.2]
T_END = 200                               # days (long enough for every run to burn out)
OPERATING_DAYS = {0, 1, 2, 3, 4, 5}       # Mon..Sat ; Sunday (6) closed
SHS_ROOMS_REUSED = False                  # SHS rooms are empty after 17:00 (switch tested in [a])
MEALS = [11, 12, 13, 17, 18]              # canteen meal windows (hour-start)
LOBBY_WAIT = {6: 1.6, 7: 1.3}             # [ASSUMED] longer lobby dwell while waiting for 07:30 classes
SUSPEND = (21, 35)                        # blanket 14-day campus suspension (benchmark policy)

HOURS = np.arange(6, 22)                  # hour-start bins 06:00 ... 21:00-22:00

def baseline_profiles(reused=SHS_ROOMS_REUSED):
    """o_k(h): fraction of peak college occupancy present, hours 06..21 [ASSUMED shapes]."""
    #            6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21
    O1 = [.15, .55, .40, .35, .45, .80, 1.0, .90, .50, .40, .45, .65, .70, .45, .50, .65]
    O2d = [.02, .30, .85, .90, .90, .85, .65, .85, .90, .90, .85]            # 06-16
    O3d = [.00, .35, .80, .90, .90, .85, .70, .85, .90, .90, .85]
    O2e = np.array([1.0, 1.0, .95, .80, .30])                                 # 17-21 shape
    O3e = np.array([.95, .90, .75, .55, .25])                                 # library shuts 20:00
    u2 = 0.85 if reused else 0.45                                             # 2F evening: SHS rooms empty
    u3 = 0.85                                                                 # 3F has no SHS rooms
    return np.array([O1, O2d + list(u2 * O2e), O3d + list(u3 * O3e)])

def shs_headcount():
    """H_k(h): SHS students present on floor k (07:30-17:00 classes on 2F) [ASSUMED]."""
    #               6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21
    s1 = np.array([.30, .40, .05, .05, .05, .30, .40, .05, .05, .05, .10, .30, .00, .00, .00, .00])
    s2 = np.array([.00, .50, .90, .90, .90, .60, .50, .90, .90, .90, .85, .00, .00, .00, .00, .00])
    return SHS_N * np.array([s1, s2, np.zeros(len(HOURS))])

def shave(O, s):
    """Staggered schedule: flatten peaks about the open-hours mean (person-hours kept)."""
    out = O.copy()
    for k in range(3):
        op = O[k] > 0.05
        m = O[k][op].mean()
        out[k][op] = m + (1 - s) * (O[k][op] - m)
    return out

def raw_ref():
    """Fixed normalisation of phi (baseline parameter means), so scenarios share one scale."""
    return ((RHO / ACH) * TAU).mean()

RAW_REF = raw_ref()

def hourly_beta(spec):
    """Return beta_kj for each open hour (unit p_trans) under a scenario spec."""
    O = baseline_profiles(spec.get("reused", SHS_ROOMS_REUSED))
    if spec.get("stagger"):
        O = shave(O, 0.40)
    H = shs_headcount()
    rho_t = RHO[:, None] * (O + H / N[:, None])           # crowding incl. SHS students
    tau_t = np.repeat(TAU[:, None], len(HOURS), axis=1).astype(float)
    eta_t = np.ones_like(tau_t)
    for h, m in LOBBY_WAIT.items():
        tau_t[0, h - 6] *= m
    for h in MEALS:
        eta_t[0, h - 6] = ETA_MEAL                        # unmasked eating/talking
    if spec.get("canteen_cap"):                           # grab-and-go cap in meal windows
        for h in MEALS:
            rho_t[0, h - 6] *= 0.60
            tau_t[0, h - 6] *= 0.60
    ach = ACH.copy()
    if spec.get("hepa"):
        ach[2] += 4.0                                      # portable HEPA CADR adds ~4 eACH
    phi_t = (rho_t / ach[:, None]) * tau_t * eta_t / RAW_REF
    Cm = C * (0.90 if spec.get("stagger") else 1.0)    # staggered changeovers: fewer crowded stair flows
    return np.array([Cm * phi_t[:, i][:, None] * O[:, i][None, :]
                     for i in range(len(HOURS))]), O

def weekly_mean_beta(beta_h):
    """Time-averaged beta (per day) over a Mon-Sat-open week. Used for R0 as an
    averaging approximation: the week (7 d) is not short relative to 1/gamma (4 d),
    so this is an indicator, not an exact periodic R0; outcomes come from simulate()."""
    return beta_h.sum(axis=0) / 24.0 * len(OPERATING_DAYS) / 7.0

def next_gen(B):
    """K_kj = expected floor-k infections caused by one floor-j case (fully susceptible)."""
    return B * N[:, None] / N[None, :] / GAMMA

def r0_of(B):
    return np.max(np.abs(np.linalg.eigvals(next_gen(B))))

SCENARIOS = {"SC-0": ("Baseline", {}),
             "SC-1": ("1F canteen cap", {"canteen_cap": True}),
             "SC-2": ("3F lab/library HEPA", {"hepa": True}),
             "SC-3": ("Staggered schedule", {"stagger": True}),
             "SC-4": ("Combined NPI", {"canteen_cap": True, "hepa": True, "stagger": True}),
             "SC-5": ("Blanket 14-day suspension", {"suspend": SUSPEND})}
SEEDS = {"1F cafeteria": 0, "3F lab": 2}

def calibrate(r0_target):
    """p_trans such that the baseline weekly-mean R0 equals r0_target."""
    return r0_target / r0_of(weekly_mean_beta(hourly_beta({})[0]))

# p_trans calibrated ONCE on the baseline so weekly-mean R0 = R0_TARGET
P_TRANS = calibrate(R0_TARGET)

# ------------------------------------------------------------------ 2. ODEs
def sir_rhs(t, y, beta, N, gamma):
    S, I, R = y[0:3], y[3:6], y[6:9]
    new_inf = S * (beta @ (I / N))
    return np.concatenate([-new_inf, new_inf - gamma * I, gamma * I])

def initial_state(seed_patch, n_seed=1.0):
    I0 = np.zeros(3); I0[seed_patch] = n_seed
    return np.concatenate([N - I0, I0, np.zeros(3)])

def simulate(beta_h, seed_patch, suspend=None, p=None):
    """Integrate with RK45 segment by segment: beta(t) is piecewise-constant per
    hour (06-22 on operating days), zero overnight/Sunday. Also returns inc[k, h],
    infections acquired on floor k during hour bin h."""
    p = P_TRANS if p is None else p
    zero = np.zeros((3, 3)); y = initial_state(seed_patch)
    ts, ys = [0.0], [y]; max_dt = 0.0; max_drift = 0.0
    inc = np.zeros((3, len(HOURS)))
    def seg(t0, t1, beta):
        nonlocal y, max_dt, max_drift
        s = solve_ivp(sir_rhs, (t0, t1), y, method="RK45", t_eval=[t1],
                      args=(beta, N, GAMMA), rtol=1e-8, atol=1e-10)
        drop = y[0:3] - s.y[0:3, -1]
        y = s.y[:, -1]
        max_dt = max(max_dt, abs(sir_rhs(t1, y, beta, N, GAMMA).sum()))        # d/dt sum(S+I+R)
        max_drift = max(max_drift, np.abs(y[0:3] + y[3:6] + y[6:9] - N).max())
        ts.append(t1); ys.append(y)
        return drop
    for d in range(T_END):
        if d % 7 not in OPERATING_DAYS or (suspend and suspend[0] <= d < suspend[1]):
            seg(d, d + 1, zero); continue
        seg(d, d + 6 / 24, zero)                                              # 00:00-06:00 closed
        for i in range(len(HOURS)):
            inc[:, i] += seg(d + HOURS[i] / 24, d + (HOURS[i] + 1) / 24, p * beta_h[i])
        seg(d + 22 / 24, d + 1, zero)                                         # 22:00-24:00 closed
    return np.array(ts), np.array(ys).T, max_dt, max_drift, inc

def homogeneous_sir(r0, n_seed=1.0):
    """Classical Kermack-McKendrick SIR, one well-mixed population of sum(N),
    constant beta = r0*gamma (no floors, no schedule): the comparison model for RQ1."""
    Nt = N.sum(); beta = r0 * GAMMA
    rhs = lambda t, y: [-beta * y[0] * y[1] / Nt, beta * y[0] * y[1] / Nt - GAMMA * y[1], GAMMA * y[1]]
    t = np.linspace(0, T_END, T_END * 24 + 1)
    s = solve_ivp(rhs, (0, T_END), [Nt - n_seed, n_seed, 0.0], t_eval=t, rtol=1e-8, atol=1e-10)
    return s.t, s.y

def final_size(r0):
    """Analytic final attack rate z = 1 - exp(-r0 z) (verification of homogeneous run)."""
    return brentq(lambda z: z - 1 + np.exp(-r0 * z), 1e-9, 1.0) if r0 > 1 else 0.0

def summarise(t, y, n_seed=1.0):
    I = y[3:6]; tot = I.sum(0)
    outbreak = tot.max() > n_seed * 1.001                 # I never exceeds the index case -> no outbreak
    return dict(t=t, y=y, I=I, peak=tot.max(), t_peak=t[tot.argmax()], outbreak=outbreak,
                cases=N.sum() - y[0:3, -1].sum(), cases_k=N - y[0:3, -1],
                active_end=tot[-1])

def tick_labels():
    return [f"{sc}\n" + textwrap.fill(v[0], 14) for sc, v in SCENARIOS.items()]

def fmt_peak(r):
    return f"peak I={r['peak']:.1f} (day {r['t_peak']:.1f})" if r["outbreak"] else "no outbreak (I never exceeds index case)"

# ------------------------------------------------------------------ 3. RUN
def main():
    print(f"gamma={GAMMA:.2f}/day  p_trans={P_TRANS:.4f}  T_END={T_END} d  "
          f"SHS rooms reused after 17:00: {SHS_ROOMS_REUSED}  SHS headcount={SHS_N:.0f}")
    res, profiles, r0s, incs = {}, {}, {}, {}
    for sc, (label, spec) in SCENARIOS.items():
        beta_h, O = hourly_beta(spec); profiles[sc] = O
        r0s[sc] = r0_of(P_TRANS * weekly_mean_beta(beta_h))
        r0_txt = f"weekly-mean R0 = {r0s[sc]:.3f}" if "suspend" not in spec else \
                 f"R0 = {r0s[sc]:.3f} outside suspension, 0 during days {SUSPEND[0]}-{SUSPEND[1]}"
        print(f"{sc} ({label}): {r0_txt}")
        for sname, sp in SEEDS.items():
            t, y, dmax, drift, inc = simulate(beta_h, sp, suspend=spec.get("suspend"))
            r = res[(sc, sname)] = summarise(t, y); incs[(sc, sname)] = inc
            print(f"   seed {sname:13s}| max|d/dt sum|={dmax:.1e} drift={drift:.1e} | "
                  f"{fmt_peak(r)} | cumulative cases={r['cases']:.0f} "
                  f"(1F/2F/3F = {r['cases_k'][0]:.0f}/{r['cases_k'][1]:.0f}/{r['cases_k'][2]:.0f})"
                  f" | active at day {T_END}: {r['active_end']:.2f}")
            assert dmax < 1e-9 and drift < 1e-6, "mass conservation violated"

    # RQ2: node-level secondary infection yields (next-generation matrix of the baseline)
    K = next_gen(P_TRANS * weekly_mean_beta(hourly_beta({})[0]))
    w, v = np.linalg.eig(K); share = np.abs(np.real(v[:, np.argmax(np.abs(w))])); share /= share.sum()
    print("\n[RQ2] Baseline next-generation matrix K_kj (rows: floor where infection occurs, cols: floor of the case)")
    for k in range(3):
        print(f"    {FLOORS[k]:28s} " + "  ".join(f"{K[k, j]:.3f}" for j in range(3)))
    print("    secondary infections per case (column sums): " +
          ", ".join(f"{FLOORS[j].split()[0]}={K[:, j].sum():.2f}" for j in range(3)))
    print("    early-epidemic share of new infections by floor: " +
          ", ".join(f"{FLOORS[k].split()[0]}={share[k]:.0%}" for k in range(3)))

    # RQ1: homogeneous comparison model at the same R0
    th, yh = homogeneous_sir(R0_TARGET)
    hom = dict(peak=yh[1].max(), t_peak=th[yh[1].argmax()], cases=N.sum() - yh[0, -1])
    print(f"\n[RQ1] Homogeneous SIR (N={N.sum():.0f}, R0={R0_TARGET}): peak I={hom['peak']:.1f} (day {hom['t_peak']:.1f}) "
          f"| cumulative cases={hom['cases']:.0f} ({hom['cases'] / N.sum():.1%}); "
          f"analytic final size {final_size(R0_TARGET):.1%}")
    for sname in SEEDS:
        r = res[("SC-0", sname)]
        print(f"      metapopulation baseline, seed {sname:13s}: {fmt_peak(r)} | cases={r['cases']:.0f} "
              f"({r['cases'] / N.sum():.1%})")

    write_csv(res, r0s, hom, K)

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3, "font.family": "serif"})
    col = {"SC-0": "k", "SC-1": "tab:blue", "SC-2": "tab:red", "SC-3": "tab:green", "SC-4": "tab:purple",
           "SC-5": "tab:orange"}

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

    # Fig 2: peak and cumulative comparison
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4)); w_ = .38; xs = np.arange(len(SCENARIOS))
    for ax, key, ylab in [(axs[0], "peak", "Peak simultaneous infections (1F+2F+3F)"),
                          (axs[1], "cases", f"Cumulative cases by day {T_END}")]:
        for i, sname in enumerate(SEEDS):
            b = ax.bar(xs + (i - .5) * w_, [res[(sc, sname)][key] for sc in SCENARIOS], w_,
                       label=f"Seed: {sname}", edgecolor="k", alpha=.85)
            ax.bar_label(b, fmt="%.0f", fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(tick_labels(), fontsize=7)
        ax.set_ylabel(ylab)
    axs[0].legend(); fig.suptitle("Comparative outbreak burden by intervention scenario")
    fig.tight_layout(); fig.savefig("fig2_peak_comparison.png", dpi=300); plt.close(fig)

    # Fig 3: assumed daily occupancy profiles (baseline vs staggered) + SHS crowding
    H = shs_headcount() / N[:, None]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for k, ax in enumerate(axs):
        hx = np.append(HOURS, 22); ext = lambda v: np.append(v, v[-1])       # draw the 21:00-22:00 bin
        ax.step(hx, ext(profiles["SC-0"][k]), where="post", color="k", label="College, baseline")
        ax.step(hx, ext(profiles["SC-3"][k]), where="post", color="tab:green", ls="--", label="College, staggered (SC-3)")
        if H[k].any():
            ax.step(hx, ext(H[k]), where="post", color="tab:orange", ls=":", label="SHS crowding $H_k/N_k$")
        for x, lab in [(7.5, "classes 07:30"), (17, "SHS ends 17:00"), (21, "classes end 21:00")]:
            ax.axvline(x, color="grey", lw=.8, ls=":"); ax.text(x + .1, 1.02, lab, rotation=90, fontsize=7, va="bottom")
        ax.set_title(FLOORS[k], pad=34); ax.set_xlabel("Hour of day"); ax.set_xlim(6, 22)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Fraction of peak occupancy $o_k$")
    fig.tight_layout(); fig.savefig("fig3_occupancy_profiles.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Fig 5: aggregate S, I, R -- metapopulation baseline vs homogeneous SIR
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for c_, lab in enumerate(["Susceptible $S$", "Infected $I$", "Recovered $R$"]):
        ax = axs[c_]
        ax.plot(th, yh[c_], color="grey", lw=2, label=f"Homogeneous SIR ($R_0$={R0_TARGET})")
        for sname, ls in zip(SEEDS, ["-", "--"]):
            r = res[("SC-0", sname)]
            ax.plot(r["t"], r["y"][3 * c_:3 * c_ + 3].sum(0), color="k", ls=ls, lw=1.4,
                    label=f"Metapopulation, seed {sname}")
        ax.set_title(lab); ax.set_xlabel("Time (days)")
    axs[0].set_ylabel("Students (all floors)"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("fig5_sir_vs_homogeneous.png", dpi=300); plt.close(fig)

    # Fig 6: where and when infections are acquired (baseline)
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for ax, sname in zip(axs, SEEDS):
        inc = incs[("SC-0", sname)]; bottom = np.zeros(len(HOURS))
        for k, c_ in enumerate(["tab:orange", "tab:blue", "tab:green"]):
            ax.bar(HOURS + .5, inc[k], 1.0, bottom=bottom, color=c_, edgecolor="k", lw=.4, label=FLOORS[k])
            bottom += inc[k]
        ax.set_title(f"Baseline, seed {sname}"); ax.set_xlabel("Hour of day"); ax.set_xlim(6, 22)
    axs[0].set_ylabel("Infections acquired (whole run)"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("fig6_infections_by_hour_floor.png", dpi=300); plt.close(fig)

    extra_analyses()
    print("Saved fig1-fig7 PNG files, sim_timeseries.csv and summary_results.csv")

def write_csv(res, r0s, hom, K):
    with open("sim_timeseries.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scenario", "seed", "t_days"] + [f"{c}{k}" for c in "SIR" for k in (1, 2, 3)])
        for (sc, sname), r in res.items():
            for i, t in enumerate(r["t"]):
                wr.writerow([sc, sname, f"{t:.5f}"] + [f"{v:.4f}" for v in r["y"][:, i]])
    with open("summary_results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scenario", "label", "seed", "weekly_mean_R0", "outbreak", "peak_I", "peak_day",
                     "cumulative_cases", "attack_rate", "cases_1F", "cases_2F", "cases_3F"])
        for (sc, sname), r in res.items():
            wr.writerow([sc, SCENARIOS[sc][0], sname, f"{r0s[sc]:.3f}", r["outbreak"], f"{r['peak']:.1f}",
                         f"{r['t_peak']:.1f}", f"{r['cases']:.0f}", f"{r['cases'] / N.sum():.3f}"] +
                        [f"{c:.0f}" for c in r["cases_k"]])
        wr.writerow(["HOM", "Homogeneous SIR", "-", f"{R0_TARGET:.3f}", True, f"{hom['peak']:.1f}",
                     f"{hom['t_peak']:.1f}", f"{hom['cases']:.0f}", f"{hom['cases'] / N.sum():.3f}", "", "", ""])

# ------------------------------------------------------------------ 4. EXTRA ANALYSES
def extra_analyses():
    """(a) SHS-room-reuse switch, (b) R0 target robustness, (c) one-at-a-time
    +/-30% sensitivity of R0 (tornado)."""
    global C, RHO, ACH, TAU, ETA_MEAL, SHS_N
    base_beta, _ = hourly_beta({})
    R0b = r0_of(P_TRANS * weekly_mean_beta(base_beta))
    R0_r = r0_of(P_TRANS * weekly_mean_beta(hourly_beta({"reused": True})[0]))
    print(f"\n[a] If SHS rooms WERE reused by college after 17:00: R0 = {R0_r:.3f} (baseline {R0b:.3f})")

    print("[b] Robustness to the R0 target (p_trans recalibrated each time, seed 1F cafeteria):")
    rob = {}
    for r0t in R0_RANGE:
        p = calibrate(r0t)
        for sc, (label, spec) in SCENARIOS.items():
            t, y, _, _, _ = simulate(hourly_beta(spec)[0], SEEDS["1F cafeteria"], suspend=spec.get("suspend"), p=p)
            r = summarise(t, y); rob[(r0t, sc)] = r["cases"] / N.sum()
        print(f"    R0={r0t}: attack rate " + ", ".join(f"{sc}={rob[(r0t, sc)]:.1%}" for sc in SCENARIOS))
    fig, ax = plt.subplots(figsize=(9, 4)); w_ = .8 / len(R0_RANGE); xs = np.arange(len(SCENARIOS))
    for i, r0t in enumerate(R0_RANGE):
        b = ax.bar(xs + (i - 1) * w_, [100 * rob[(r0t, sc)] for sc in SCENARIOS], w_, edgecolor="k",
                   label=f"Baseline $R_0$ = {r0t}")
        ax.bar_label(b, fmt="%.0f", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(tick_labels(), fontsize=7)
    ax.set_ylabel(f"Attack rate by day {T_END} (%)"); ax.legend(fontsize=8)
    ax.set_title("Scenario ranking under different $R_0$ assumptions (seed: 1F cafeteria)")
    fig.tight_layout(); fig.savefig("fig7_r0_target_robustness.png", dpi=300); plt.close(fig)

    C0, R0_, A0, T0, E0, S0 = C.copy(), RHO.copy(), ACH.copy(), TAU.copy(), ETA_MEAL, SHS_N
    def r0_mod(Cm=None, rho=None, ach=None, tau=None, eta=None, shs=None):
        global C, RHO, ACH, TAU, ETA_MEAL, SHS_N
        C = C0.copy() if Cm is None else Cm; RHO = R0_.copy() if rho is None else rho
        ACH = A0.copy() if ach is None else ach; TAU = T0.copy() if tau is None else tau
        ETA_MEAL = E0 if eta is None else eta; SHS_N = S0 if shs is None else shs
        r = r0_of(P_TRANS * weekly_mean_beta(hourly_beta({})[0]))
        C, RHO, ACH, TAU, ETA_MEAL, SHS_N = C0.copy(), R0_.copy(), A0.copy(), T0.copy(), E0, S0
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
    items.append(("canteen eating factor eta", r0_mod(eta=E0 * .7), r0_mod(eta=E0 * 1.3)))
    items.append(("SHS headcount", r0_mod(shs=S0 * .7), r0_mod(shs=S0 * 1.3)))
    items.sort(key=lambda x: abs(x[2] - x[1]))
    print("[c] R0 sensitivity (+/-30% one at a time, p_trans fixed):")
    for lab, lo, hi in items[::-1]:
        print(f"    {lab:32s} {lo:.2f} .. {hi:.2f}")
    fig, ax = plt.subplots(figsize=(8, 5.5))
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
