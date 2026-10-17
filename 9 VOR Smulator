"""
VOR / CVOR Simulators Hub — Streamlit Single-File App
=====================================================
مجموعه‌ی ۹ شبیه‌ساز در یک برنامه. کاربر از نوار کناری شبیه‌ساز را انتخاب می‌کند.

شبیه‌سازها:
  1) CVOR ایده‌آل (استخراج زاویه)
  2) اثر توپوگرافی کوه
  3) جنگل
  4) یک درخت
  5) یک ساختمان
  6) چند ساختمان
  7) دکل انتقال نیرو
  8) نویز کرونا/عایق دکل فشار قوی
  9) توربین بادی / مزرعه بادی

اجرا:
    streamlit run vor_simulators_hub.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, hilbert


# ############################################################
#  بخش ۱: هسته‌های محاسباتی (بدون تغییر نسبت به فایل‌های اصلی)
# ############################################################

# ------------------------------------------------------------
# 1) CVOR ایده‌آل
# ------------------------------------------------------------
def simulate_cvor(true_bearing_deg=250.0,
                  fs=200e3, T=1.0,
                  f_ref=30.0, f_sub=9960.0,
                  dev=480.0,
                  m_ref=0.3, m_var=0.3):
    """شبیه‌سازی ایده‌آل CVOR و تخمین زاویه."""
    t = np.arange(0, T, 1/fs)
    beta = dev / f_ref
    theta = np.deg2rad(true_bearing_deg)

    var_sig = m_var * np.cos(2*np.pi*f_ref*t - theta)
    ref_fm = m_ref * np.cos(2*np.pi*f_sub*t + beta*np.sin(2*np.pi*f_ref*t))
    m_t = var_sig + ref_fm
    rx = m_t.copy()

    bv, av = butter(4, 100/(fs/2), btype='low')
    var_rx = filtfilt(bv, av, rx)

    I_var = 2*np.mean(var_rx * np.cos(2*np.pi*f_ref*t))
    Q_var = -2*np.mean(var_rx * np.sin(2*np.pi*f_ref*t))
    phase_var = np.arctan2(Q_var, I_var)

    br, ar = butter(4, [(f_sub-1500)/(fs/2), (f_sub+1500)/(fs/2)], btype='bandpass')
    sub_rx = filtfilt(br, ar, rx)

    analytic_sub = hilbert(sub_rx)
    inst_phase = np.unwrap(np.angle(analytic_sub))
    inst_freq = np.concatenate(([0.0], np.diff(inst_phase))) * fs / (2*np.pi)
    ref_demod = inst_freq - f_sub

    valid = (t > 0.05) & (t < (T - 0.05))
    tv = t[valid]
    ref_valid = ref_demod[valid]

    I_ref = 2*np.mean(ref_valid * np.cos(2*np.pi*f_ref*tv))
    Q_ref = -2*np.mean(ref_valid * np.sin(2*np.pi*f_ref*tv))
    phase_ref = np.arctan2(Q_ref, I_ref)

    measured_bearing_deg = np.mod(np.rad2deg(phase_ref - phase_var), 360)
    bearing_error_deg = np.mod(measured_bearing_deg - true_bearing_deg + 180, 360) - 180

    N = len(m_t)
    M = np.abs(np.fft.fft(m_t)) / N
    f_axis = np.arange(N) * (fs / N)
    half = N // 2

    test_bearings = np.arange(0, 360, 10)
    errors = np.zeros_like(test_bearings, dtype=float)
    for k, tb in enumerate(test_bearings):
        th_k = np.deg2rad(tb)
        var_k = m_var * np.cos(2*np.pi*f_ref*t - th_k)
        ref_k = m_ref * np.cos(2*np.pi*f_sub*t + beta*np.sin(2*np.pi*f_ref*t))
        rx_k = var_k + ref_k

        var_k_rx = filtfilt(bv, av, rx_k)
        I_v = 2*np.mean(var_k_rx * np.cos(2*np.pi*f_ref*t))
        Q_v = -2*np.mean(var_k_rx * np.sin(2*np.pi*f_ref*t))
        ph_v = np.arctan2(Q_v, I_v)

        sub_k = filtfilt(br, ar, rx_k)
        ph_k = np.unwrap(np.angle(hilbert(sub_k)))
        fi_k = np.concatenate(([0.0], np.diff(ph_k))) * fs / (2*np.pi)
        rd_k = fi_k - f_sub
        rd_valid = rd_k[valid]

        I_r = 2*np.mean(rd_valid * np.cos(2*np.pi*f_ref*tv))
        Q_r = -2*np.mean(rd_valid * np.sin(2*np.pi*f_ref*tv))
        ph_r = np.arctan2(Q_r, I_r)

        bearing_k = np.mod(np.rad2deg(ph_r - ph_v), 360)
        errors[k] = np.mod(bearing_k - tb + 180, 360) - 180

    return {
        "t": t, "m_t": m_t, "var_rx": var_rx, "sub_rx": sub_rx,
        "tv": tv, "ref_valid": ref_valid, "valid": valid,
        "f_axis": f_axis[:half], "spectrum": 2 * M[:half],
        "fs": fs,
        "measured_bearing_deg": float(measured_bearing_deg),
        "bearing_error_deg": float(bearing_error_deg),
        "test_bearings": test_bearings,
        "errors": errors,
        "params": dict(true_bearing_deg=true_bearing_deg, fs=fs, T=T,
                       f_ref=f_ref, f_sub=f_sub, dev=dev,
                       m_ref=m_ref, m_var=m_var),
    }


# ------------------------------------------------------------
# 2) کوه
# ------------------------------------------------------------
def simulate_mountain_vor(
        bearing_ac_deg=60.0,
        f_c=113e6,
        D_m=12000.0, H_m=900.0,
        rho_diff=0.35, rho0_refl=0.15,
        h_ac=2500.0,
        R_max=45000.0, R_min=1500.0,
        N=1200,
        tol=3.0):
    """شبیه‌سازی اثر کوه بر خطای زاویه VOR."""
    c = 299792458.0
    lam = c / f_c

    R = np.linspace(R_max, R_min, N)

    h_ob = np.zeros(N)
    nu = np.zeros(N)
    Jnu_dB = np.zeros(N)
    rho = np.zeros(N)
    mode = [""] * N
    d_direct = np.zeros(N)
    d_scatter = np.zeros(N)
    dPhi_deg = np.zeros(N)

    for k in range(N):
        Rk = R[k]
        d_direct[k] = np.hypot(Rk, h_ac)

        if Rk > D_m:
            los_h = h_ac * (D_m / Rk)
            h_ob[k] = H_m - los_h
        else:
            h_ob[k] = -np.inf

        d1 = np.hypot(D_m, H_m)
        d2 = np.hypot(Rk - D_m, h_ac - H_m)
        d_scatter[k] = d1 + d2

        if h_ob[k] > 0:
            mode[k] = "diffraction"
            nu[k] = h_ob[k] * np.sqrt(2*(d1 + d2) / (lam * d1 * d2))
            if nu[k] <= -0.78:
                Jnu_dB[k] = 0.0
            else:
                Jnu_dB[k] = 6.9 + 20*np.log10(
                    np.sqrt((nu[k] - 0.1)**2 + 1) + nu[k] - 0.1
                )
            A_diff = 10 ** (-Jnu_dB[k] / 20.0)
            rho[k] = rho_diff * A_diff
        else:
            mode[k] = "reflection"
            Jnu_dB[k] = 0.0
            spread_factor = d_direct[k] / d_scatter[k]
            rho[k] = rho0_refl * spread_factor

        dPhi_deg[k] = np.mod(360.0 * (d_scatter[k] - d_direct[k]) / lam, 360.0)

    phi_rad = np.deg2rad(dPhi_deg)
    err_rad = np.arctan2(rho * np.sin(phi_rad), 1 + rho * np.cos(phi_rad))
    bearing_error_deg = np.rad2deg(err_rad)

    exceed = np.abs(bearing_error_deg) > tol
    max_err_idx = int(np.argmax(np.abs(bearing_error_deg)))
    max_err = float(np.abs(bearing_error_deg)[max_err_idx])
    trans_idx = next((i for i, m in enumerate(mode) if m == "reflection"), None)

    return {
        "R": R, "h_ac": h_ac,
        "D_m": D_m, "H_m": H_m,
        "rho": rho, "d_direct": d_direct, "d_scatter": d_scatter,
        "dDelta": d_scatter - d_direct,
        "bearing_error_deg": bearing_error_deg,
        "exceed": exceed, "mode": mode,
        "tol": tol,
        "max_err": max_err, "max_err_idx": max_err_idx,
        "frac_exceed": float(np.mean(exceed)),
        "trans_idx": trans_idx,
        "R_max": R_max,
        "bearing_ac_deg": bearing_ac_deg,
        "f_c": f_c,
    }


# ------------------------------------------------------------
# 3) جنگل
# ------------------------------------------------------------
def cvor_demod(env, e30, ephase, mask, pa_k):
    """دمدولاسیون CVOR: VAR از پاکت AM و REF از FM زیرحامل."""
    X = np.sum(env * e30)
    phi_var = -np.angle(X)

    z = np.fft.ifft(np.fft.fft(env) * mask)
    psi = np.unwrap(np.angle(z * ephase))
    delta = np.angle(np.sum(psi * e30)) + np.pi / 2

    err_deg = np.rad2deg(np.angle(np.exp(1j * (pa_k - (phi_var + delta)))))
    dref_deg = np.rad2deg(np.angle(np.exp(1j * delta)))
    return float(err_deg), float(dref_deg)


def generate_forest(D_c, D_in, az_forest_deg, density,
                    sigma_mean, sigma_sd, h_ant, seed=1):
    """تولید موقعیت تصادفی درختان و پارامترهای پراکندگی."""
    rng = np.random.default_rng(seed)
    R_f = D_c - D_in
    area = np.pi * R_f ** 2
    Nt = int(round(density * area / 1e4))
    rr = R_f * np.sqrt(rng.random(Nt))
    ang = 2 * np.pi * rng.random(Nt)
    xc = D_c * np.sin(np.deg2rad(az_forest_deg))
    yc = D_c * np.cos(np.deg2rad(az_forest_deg))
    xt = xc + rr * np.cos(ang)
    yt = yc + rr * np.sin(ang)
    zt = 8 + 12 * rng.random(Nt)
    sigma = sigma_mean * 10 ** (sigma_sd * rng.standard_normal(Nt) / 10)

    azt = np.arctan2(xt, yt)
    d1 = np.sqrt(xt ** 2 + yt ** 2 + (zt - h_ant) ** 2)
    amp0 = np.sqrt(sigma / (4 * np.pi))

    return dict(Nt=Nt, R_f=R_f, xt=xt, yt=yt, zt=zt,
                azt=azt, d1=d1, amp0=amp0,
                xc=xc, yc=yc)


def run_forest_simulation(
        f0=112e6, fLF=30.0, fsc=9960.0,
        mf=16.0, ma=0.30,
        R_air=10000.0, h_air=1000.0, h_ant=5.0,
        az_forest=45.0, D_in=300.0, D_c=1500.0,
        density=5.0, sigma_mean=10.0, sigma_sd=3.0,
        phi_ref=np.pi,
        tol=1.0, az_step=0.5,
        az_zoom0=60.0, az_zoom_len=0.5, az_zoom_step=0.0005,
        fs=24000.0, Tw=0.2,
        seed=1, progress_cb=None):
    """اجرای کامل شبیه‌سازی جنگل + سه حالت مولتی‌پات."""
    c = 299792458.0
    lam = c / f0

    forest = generate_forest(D_c, D_in, az_forest, density,
                             sigma_mean, sigma_sd, h_ant, seed)
    Nt = forest["Nt"]; R_f = forest["R_f"]
    xt = forest["xt"]; yt = forest["yt"]; zt = forest["zt"]
    azt = forest["azt"]; d1 = forest["d1"]; amp0 = forest["amp0"]

    az = np.arange(0.0, 360.0, az_step)
    pa = np.deg2rad(az)
    N = len(az)
    Ax = R_air * np.sin(pa)
    Ay = R_air * np.cos(pa)

    t = np.arange(0, Tw, 1 / fs)
    Ns = len(t)
    w = 2 * np.pi * fLF
    df = fs / Ns

    bw = 1500.0
    kidx = np.arange(int(round((fsc - bw) / df)),
                     int(round((fsc + bw) / df)) + 1)
    mask = np.zeros(Ns, dtype=complex)
    mask[np.clip(kidx + 1, 0, Ns - 1)] = 2.0

    psi_dir = 2 * np.pi * fsc * t + mf * np.sin(w * t)
    ejpsi = np.exp(1j * psi_dir)
    ejwt = np.exp(1j * w * t)
    cwt = np.cos(w * t)
    e30 = np.exp(-1j * w * t)
    ephase = np.exp(-1j * 2 * np.pi * fsc * t)

    cases = np.array([[1, 0], [0, 1], [1, 1]])
    names = ["A: VAR only", "B: REF only", "C: VAR + REF"]

    eps_t = np.zeros((3, N))
    dREF = np.zeros((3, N))
    eps_f = np.zeros(N)

    for k in range(N):
        if progress_cb is not None and (k % 50 == 0 or k == N - 1):
            progress_cb(k / N)

        dd = np.sqrt(Ax[k] ** 2 + Ay[k] ** 2 + (h_air - h_ant) ** 2)
        d2 = np.sqrt((Ax[k] - xt) ** 2 + (Ay[k] - yt) ** 2
                     + (h_air - zt) ** 2)
        a = amp0 * dd / (d1 * d2)
        dpath = d1 + d2 - dd
        theta = -2 * np.pi * dpath / lam + phi_ref
        tau = dpath / c

        phin = azt - pa[k]
        ac = a * np.cos(theta)
        eps_f[k] = np.rad2deg(np.arctan2(-np.sum(ac * np.sin(phin)),
                                          1 + np.sum(ac * np.cos(phin))))

        g = a * np.exp(1j * theta)
        M0 = np.sum(g)
        Cc = np.sum(g * np.exp(-1j * (w * tau + azt)))
        Cp = np.sum(g * np.exp(1j * (w * tau + azt)))
        ep = np.exp(-1j * 2 * np.pi * fsc * tau)
        P0 = np.sum(g * ep)
        P1 = np.sum(g * tau * ep)
        Q0 = np.sum(g * np.conj(ep))
        Q1 = np.sum(g * tau * np.conj(ep))

        VARterm = 0.5 * (Cc * ejwt + Cp * np.conj(ejwt))
        REFterm = (0.5 * ejpsi * (P0 - 1j * mf * w * cwt * P1)
                   + 0.5 * np.conj(ejpsi) * (Q0 + 1j * mf * w * cwt * Q1))
        s_dir = 1 + ma * np.cos(psi_dir) + ma * np.cos(w * t - pa[k])

        for ci in range(3):
            s = (s_dir + M0
                 + cases[ci, 1] * ma * REFterm
                 + cases[ci, 0] * ma * VARterm)
            err, dref = cvor_demod(np.abs(s), e30, ephase, mask, pa[k])
            eps_t[ci, k] = err
            dREF[ci, k] = dref

    k0 = int(round(N / 12))
    s = 1 + ma * np.cos(psi_dir) + ma * np.cos(w * t - pa[k0])
    base_err, _ = cvor_demod(np.abs(s), e30, ephase, mask, pa[k0])

    az_z = az_zoom0 + np.arange(0, az_zoom_len + az_zoom_step / 2, az_zoom_step)
    paz = np.deg2rad(az_z)
    Nz = len(az_z)
    eps_z = np.zeros(Nz)
    for k in range(Nz):
        Axk = R_air * np.sin(paz[k])
        Ayk = R_air * np.cos(paz[k])
        dd = np.sqrt(Axk ** 2 + Ayk ** 2 + (h_air - h_ant) ** 2)
        d2 = np.sqrt((Axk - xt) ** 2 + (Ayk - yt) ** 2
                     + (h_air - zt) ** 2)
        a = amp0 * dd / (d1 * d2)
        theta = -2 * np.pi * (d1 + d2 - dd) / lam + phi_ref
        ac = a * np.cos(theta)
        phin = azt - paz[k]
        eps_z[k] = np.rad2deg(np.arctan2(-np.sum(ac * np.sin(phin)),
                                          1 + np.sum(ac * np.cos(phin))))

    def p95(x):
        x_abs = np.sort(np.abs(x))
        idx = int(np.ceil(0.95 * len(x_abs))) - 1
        return float(x_abs[max(idx, 0)])

    stats = {
        "base_err": base_err,
        "lambda": lam,
        "Nt": Nt,
        "R_f": R_f,
        "names": names,
        "stats": [
            dict(name=names[ci],
                 max_err=float(np.max(np.abs(eps_t[ci]))),
                 rms=float(np.sqrt(np.mean(eps_t[ci] ** 2))),
                 p95=p95(eps_t[ci]),
                 max_dref=float(np.max(np.abs(dREF[ci]))))
            for ci in range(3)
        ],
        "eps_f_stats": dict(
            max_err=float(np.max(np.abs(eps_f))),
            rms=float(np.sqrt(np.mean(eps_f ** 2))),
            p95=p95(eps_f)
        ),
        "max_A_vs_eq": float(np.max(np.abs(eps_t[0] - eps_f))),
        "max_C_vs_A": float(np.max(np.abs(eps_t[2] - eps_t[0]))),
        "frac_outside_tol": float(np.mean(np.abs(eps_t[2]) > tol)),
    }

    return {
        "az": az, "eps_t": eps_t, "dREF": dREF, "eps_f": eps_f,
        "az_z": az_z, "eps_z": eps_z,
        "forest": forest, "stats": stats, "tol": tol, "names": names,
        "D_in": D_in, "D_c": D_c,
    }


# ------------------------------------------------------------
# 4) یک درخت
# ------------------------------------------------------------
def run_single_tree_simulation(
        f0=112e6, fLF=30.0, fsc=9960.0,
        mf=16.0, ma=0.30,
        R_air=10000.0, h_air=1000.0, h_ant=5.0,
        d_tree=100.0, az_tree=45.0, h_tree=10.0,
        rho=0.10, phi_ref=np.pi,
        tol=1.0, az_step=0.25,
        fs=24000.0, Tw=0.2,
        progress_cb=None):
    """اجرای شبیه‌سازی تک‌درخت + سه حالت مولتی‌پات."""
    c = 299792458.0
    lam = c / f0

    az = np.arange(0.0, 360.0, az_step)
    pa = np.deg2rad(az)
    pt = np.deg2rad(az_tree)
    N = len(az)

    S = np.array([0.0, 0.0, h_ant])
    T = np.array([d_tree * np.sin(pt),
                  d_tree * np.cos(pt),
                  h_tree])
    Ax = R_air * np.sin(pa)
    Ay = R_air * np.cos(pa)
    Az = h_air * np.ones(N)

    d_direct = np.sqrt(Ax**2 + Ay**2 + (Az - S[2])**2)
    d1 = float(np.linalg.norm(T - S))
    d2 = np.sqrt((Ax - T[0])**2 + (Ay - T[1])**2 + (Az - T[2])**2)

    a = rho * d_direct / (d1 + d2)
    dpath = (d1 + d2) - d_direct
    theta = -2 * np.pi * dpath / lam + phi_ref
    tau = dpath / c

    dtheta = np.abs(np.diff(np.unwrap(theta)))
    max_dtheta = float(np.max(dtheta)) if len(dtheta) > 0 else 0.0

    phin = pt - pa
    eps_f = np.rad2deg(
        np.arctan2(-a * np.cos(theta) * np.sin(phin),
                   1 + a * np.cos(theta) * np.cos(phin))
    )

    t = np.arange(0, Tw, 1 / fs)
    Ns = len(t)
    w = 2 * np.pi * fLF
    df = fs / Ns

    bw = 1500.0
    kidx = np.arange(int(round((fsc - bw) / df)),
                     int(round((fsc + bw) / df)) + 1)
    mask = np.zeros(Ns, dtype=complex)
    mask[np.clip(kidx + 1, 0, Ns - 1)] = 2.0

    psi_dir = 2 * np.pi * fsc * t + mf * np.sin(w * t)
    e30 = np.exp(-1j * w * t)
    ephase = np.exp(-1j * 2 * np.pi * fsc * t)

    cases = np.array([[1, 0], [0, 1], [1, 1]])
    names = ["A: VAR only", "B: REF only", "C: VAR + REF"]
    eps_t = np.zeros((3, N))
    dREF = np.zeros((3, N))

    total_steps = 3 * N
    step_counter = 0

    for ci in range(3):
        fV = cases[ci, 0]
        fR = cases[ci, 1]
        for k in range(N):
            s_dir = 1 + ma * np.cos(psi_dir) + ma * np.cos(w * t - pa[k])
            psi_mp = (2 * np.pi * fsc * (t - tau[k])
                      + mf * np.sin(w * (t - tau[k])))
            s_mp = (1
                    + fR * ma * np.cos(psi_mp)
                    + fV * ma * np.cos(w * (t - tau[k]) - pt))
            s = s_dir + a[k] * np.exp(1j * theta[k]) * s_mp
            env = np.abs(s)

            X = np.sum(env * e30)
            phi_var = -np.angle(X)

            z = np.fft.ifft(np.fft.fft(env) * mask)
            psi = np.unwrap(np.angle(z * ephase))
            delta = np.angle(np.sum(psi * e30)) + np.pi / 2

            eps_t[ci, k] = np.rad2deg(
                np.angle(np.exp(1j * (pa[k] - (phi_var + delta))))
            )
            dREF[ci, k] = np.rad2deg(np.angle(np.exp(1j * delta)))

            step_counter += 1
            if progress_cb is not None and step_counter % 100 == 0:
                progress_cb(step_counter / total_steps)
        if progress_cb is not None:
            progress_cb(step_counter / total_steps)

    k0 = int(round(N / 12))
    s = 1 + ma * np.cos(psi_dir) + ma * np.cos(w * t - pa[k0])
    env = np.abs(s)
    phi_var = -np.angle(np.sum(env * e30))
    z = np.fft.ifft(np.fft.fft(env) * mask)
    psi = np.unwrap(np.angle(z * ephase))
    delta = np.angle(np.sum(psi * e30)) + np.pi / 2
    base_err = float(np.rad2deg(
        np.angle(np.exp(1j * (pa[k0] - (phi_var + delta))))
    ))

    ip = int(np.argmax(np.abs(eps_t[2])))
    s = 1 + ma * np.cos(psi_dir) + ma * np.cos(w * t - pa[ip])
    psi_mp = (2 * np.pi * fsc * (t - tau[ip])
              + mf * np.sin(w * (t - tau[ip])))
    s = s + a[ip] * np.exp(1j * theta[ip]) * (
        1 + ma * np.cos(psi_mp)
        + ma * np.cos(w * (t - tau[ip]) - pt)
    )
    E = np.abs(np.fft.fft(np.abs(s))) / Ns
    f_axis = np.arange(Ns) * df

    stats = []
    for ci in range(3):
        stats.append(dict(
            name=names[ci],
            max_err=float(np.max(np.abs(eps_t[ci]))),
            rms=float(np.sqrt(np.mean(eps_t[ci] ** 2))),
            max_dref=float(np.max(np.abs(dREF[ci]))),
        ))

    return {
        "az": az, "eps_t": eps_t, "eps_f": eps_f, "dREF": dREF,
        "f_axis": f_axis, "E": E, "fsc": fsc, "bw": bw,
        "max_dtheta": max_dtheta,
        "base_err": base_err,
        "lambda": lam,
        "max_tau_us": float(1e6 * np.max(tau)),
        "stats": stats,
        "max_A_vs_eq": float(np.max(np.abs(eps_t[0] - eps_f))),
        "max_C_vs_A": float(np.max(np.abs(eps_t[2] - eps_t[0]))),
        "eq_max": float(np.max(np.abs(eps_f))),
        "names": names, "tol": tol,
        "d_tree": d_tree, "az_tree": az_tree, "az_step": az_step,
    }


# ------------------------------------------------------------
# 5) یک ساختمان
# ------------------------------------------------------------
def simulate_building_vor(
        f_c=113e6,
        D_b=220.0, H_b=15.0, W_b=20.0,
        base_rho=0.12, area_ref=200.0, rho_diff=0.25,
        h_ac=300.0,
        R_max=15000.0, R_min=300.0,
        N=1200,
        tol=1.0):
    """شبیه‌سازی اثر ساختمان تنها بر خطای زاویه VOR."""
    c = 299792458.0
    lam = c / f_c

    R = np.linspace(R_max, R_min, int(N))
    Nn = int(N)

    h_ob = np.zeros(Nn)
    nu = np.zeros(Nn)
    Jnu_dB = np.zeros(Nn)
    rho = np.zeros(Nn)
    mode = [""] * Nn
    d_direct = np.zeros(Nn)
    d_scatter = np.zeros(Nn)
    dPhi_deg = np.zeros(Nn)

    area_factor = np.sqrt((H_b * W_b) / area_ref)

    for k in range(Nn):
        Rk = R[k]
        d_direct[k] = np.hypot(Rk, h_ac)

        if Rk > D_b:
            los_h = h_ac * (D_b / Rk)
            h_ob[k] = H_b - los_h
        else:
            h_ob[k] = -np.inf

        d1 = np.hypot(D_b, H_b)
        d2 = np.hypot(Rk - D_b, h_ac - H_b)
        d_scatter[k] = d1 + d2

        if h_ob[k] > 0:
            mode[k] = "diffraction"
            nu[k] = h_ob[k] * np.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
            if nu[k] <= -0.78:
                Jnu_dB[k] = 0.0
            else:
                Jnu_dB[k] = 6.9 + 20 * np.log10(
                    np.sqrt((nu[k] - 0.1) ** 2 + 1) + nu[k] - 0.1
                )
            A_diff = 10 ** (-Jnu_dB[k] / 20.0)
            rho[k] = rho_diff * A_diff
        else:
            mode[k] = "reflection"
            Jnu_dB[k] = 0.0
            spread_factor = d_direct[k] / d_scatter[k]
            rho[k] = base_rho * area_factor * spread_factor

        dPhi_deg[k] = np.mod(360.0 * (d_scatter[k] - d_direct[k]) / lam, 360.0)

    phi_rad = np.deg2rad(dPhi_deg)
    err_rad = np.arctan2(rho * np.sin(phi_rad), 1 + rho * np.cos(phi_rad))
    bearing_error_deg = np.rad2deg(err_rad)

    exceed = np.abs(bearing_error_deg) > tol
    max_err_idx = int(np.argmax(np.abs(bearing_error_deg)))
    max_err = float(np.abs(bearing_error_deg)[max_err_idx])
    trans_idx = next((i for i, m in enumerate(mode) if m == "reflection"), None)

    return {
        "R": R, "h_ac": h_ac,
        "D_b": D_b, "H_b": H_b, "W_b": W_b,
        "rho": rho, "d_direct": d_direct, "d_scatter": d_scatter,
        "dDelta": d_scatter - d_direct,
        "bearing_error_deg": bearing_error_deg,
        "exceed": exceed, "mode": mode, "tol": tol,
        "max_err": max_err, "max_err_idx": max_err_idx,
        "frac_exceed": float(np.mean(exceed)),
        "trans_idx": trans_idx,
        "R_max": R_max, "R_min": R_min,
        "f_c": f_c,
        "area_factor": area_factor,
        "lambda": lam,
    }


# ------------------------------------------------------------
# 6) چند ساختمان
# ------------------------------------------------------------
def simulate_multi_building(
        buildings,
        f_c=113e6,
        base_rho=0.12, area_ref=200.0, rho_diff=0.25,
        h_ac=300.0,
        R_max=15000.0, R_min=300.0,
        N=1200,
        tol=1.0):
    """شبیه‌سازی چند ساختمان با جمع فازوری روی کانال VAR."""
    c = 299792458.0
    lam = c / f_c

    R = np.linspace(R_max, R_min, int(N))
    Nn = int(N)
    Nb = len(buildings)

    rho = np.zeros((Nb, Nn))
    dPhi_deg = np.zeros((Nb, Nn))
    mode = [[("", "")] * Nn for _ in range(Nb)]

    d_direct = np.hypot(R, h_ac)

    D_arr = np.array([b["D"] for b in buildings], dtype=float)
    H_arr = np.array([b["H"] for b in buildings], dtype=float)
    W_arr = np.array([b["W"] for b in buildings], dtype=float)
    names = [b["name"] for b in buildings]

    area_factor = np.sqrt((H_arr * W_arr) / area_ref)

    for i in range(Nb):
        d1 = np.hypot(D_arr[i], H_arr[i])
        for k in range(Nn):
            Rk = R[k]
            if Rk > D_arr[i]:
                los_h = h_ac * (D_arr[i] / Rk)
                h_ob = H_arr[i] - los_h
            else:
                h_ob = -np.inf

            d2 = np.hypot(Rk - D_arr[i], h_ac - H_arr[i])
            d_scatter = d1 + d2

            if h_ob > 0:
                mode[i][k] = "diffraction"
                nu = h_ob * np.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
                if nu <= -0.78:
                    Jnu_dB = 0.0
                else:
                    Jnu_dB = 6.9 + 20 * np.log10(
                        np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1
                    )
                A_diff = 10 ** (-Jnu_dB / 20.0)
                rho[i, k] = rho_diff * A_diff
            else:
                mode[i][k] = "reflection"
                spread_factor = d_direct[k] / d_scatter
                rho[i, k] = base_rho * area_factor[i] * spread_factor

            dPhi_deg[i, k] = np.mod(
                360.0 * (d_scatter - d_direct[k]) / lam, 360.0
            )

    resultant = np.ones(Nn, dtype=complex)
    for i in range(Nb):
        resultant = resultant + rho[i] * np.exp(1j * np.deg2rad(dPhi_deg[i]))

    bearing_error_deg = np.rad2deg(np.angle(resultant))
    exceed = np.abs(bearing_error_deg) > tol

    max_err_idx = int(np.argmax(np.abs(bearing_error_deg)))
    max_err = float(np.abs(bearing_error_deg)[max_err_idx])

    return {
        "R": R, "h_ac": h_ac,
        "rho": rho, "dPhi_deg": dPhi_deg, "mode": mode,
        "d_direct": d_direct,
        "bearing_error_deg": bearing_error_deg,
        "resultant": resultant,
        "exceed": exceed, "tol": tol,
        "max_err": max_err, "max_err_idx": max_err_idx,
        "frac_exceed": float(np.mean(exceed)),
        "R_max": R_max, "R_min": R_min,
        "f_c": f_c, "lambda": lam,
        "area_factor": area_factor,
        "D_arr": D_arr, "H_arr": H_arr, "W_arr": W_arr,
        "names": names, "Nb": Nb,
        "buildings": buildings,
    }


# ------------------------------------------------------------
# 7) دکل انتقال نیرو
# ------------------------------------------------------------
def simulate_power_tower(
        elements,
        D_t=250.0,
        f_c=113e6,
        base_rho=0.20, area_ref=60.0, rho_diff=0.30,
        arm_depth=1.5,
        h_ac=300.0,
        R_max=15000.0, R_min=300.0,
        N=1200,
        tol=1.0):
    """شبیه‌سازی اثر دکل انتقال نیرو بر خطای زاویه VOR."""
    c = 299792458.0
    lam = c / f_c

    R = np.linspace(R_max, R_min, int(N))
    Nn = int(N)
    Ne = len(elements)

    rho = np.zeros((Ne, Nn))
    dPhi_deg = np.zeros((Ne, Nn))
    mode = [[""] * Nn for _ in range(Ne)]

    d_direct = np.hypot(R, h_ac)

    h_e = np.array([e["h"] for e in elements], dtype=float)
    w_e = np.array([e["w"] for e in elements], dtype=float)
    names = [e["name"] for e in elements]

    area_factor = np.sqrt((w_e * arm_depth) / area_ref)

    for i in range(Ne):
        d1 = np.hypot(D_t, h_e[i])
        for k in range(Nn):
            Rk = R[k]
            if Rk > D_t:
                los_h = h_ac * (D_t / Rk)
                h_ob = h_e[i] - los_h
            else:
                h_ob = -np.inf

            d2 = np.hypot(Rk - D_t, h_ac - h_e[i])
            d_scatter = d1 + d2

            if h_ob > 0:
                mode[i][k] = "diffraction"
                nu = h_ob * np.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
                if nu <= -0.78:
                    Jnu_dB = 0.0
                else:
                    Jnu_dB = 6.9 + 20 * np.log10(
                        np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1
                    )
                A_diff = 10 ** (-Jnu_dB / 20.0)
                rho[i, k] = rho_diff * A_diff
            else:
                mode[i][k] = "reflection"
                spread_factor = d_direct[k] / d_scatter
                rho[i, k] = base_rho * area_factor[i] * spread_factor

            dPhi_deg[i, k] = np.mod(
                360.0 * (d_scatter - d_direct[k]) / lam, 360.0
            )

    resultant = np.ones(Nn, dtype=complex)
    for i in range(Ne):
        resultant = resultant + rho[i] * np.exp(1j * np.deg2rad(dPhi_deg[i]))

    bearing_error_deg = np.rad2deg(np.angle(resultant))
    exceed = np.abs(bearing_error_deg) > tol

    max_err_idx = int(np.argmax(np.abs(bearing_error_deg)))
    max_err = float(np.abs(bearing_error_deg)[max_err_idx])

    return {
        "R": R, "h_ac": h_ac,
        "rho": rho, "dPhi_deg": dPhi_deg, "mode": mode,
        "d_direct": d_direct,
        "bearing_error_deg": bearing_error_deg,
        "resultant": resultant,
        "exceed": exceed, "tol": tol,
        "max_err": max_err, "max_err_idx": max_err_idx,
        "frac_exceed": float(np.mean(exceed)),
        "R_max": R_max, "R_min": R_min,
        "D_t": D_t, "f_c": f_c, "lambda": lam,
        "area_factor": area_factor,
        "arm_depth": arm_depth,
        "h_e": h_e, "w_e": w_e,
        "names": names, "Ne": Ne,
        "elements": elements,
    }


# ------------------------------------------------------------
# 8) نویز کرونا / عایق
# ------------------------------------------------------------
def analytic_signal_fft(x):
    """سیگنال تحلیلی بدون toolbox — معادل analyticSignalFFT در MATLAB."""
    x = x.flatten()
    N = len(x)
    X = np.fft.fft(x)

    H = np.zeros(N, dtype=complex)
    if N % 2 == 0:
        H[0] = 1.0
        H[N // 2] = 1.0
        H[1:N // 2] = 2.0
    else:
        H[0] = 1.0
        H[1:(N + 1) // 2] = 2.0

    return np.fft.ifft(X * H)


def movmean(x, window):
    """میانگین متحرک (centered) — معادل movmean MATLAB."""
    window = int(max(1, window))
    if window == 1:
        return x.copy()
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")[:len(x)]


def movmean_complex(x, window):
    """میانگین متحرک برای سیگنال مختلط."""
    return (movmean(np.real(x), window)
            + 1j * movmean(np.imag(x), window))


def run_pylon_corona_simulation(
        fVOR=113.8e6, f30=30.0, fSC=9960.0,
        bearing_deg=40.0,
        m_ref=0.30, m_fm=16.0,
        fs=200e3, T=4.0,
        distance_m=800.0, HV_kV=400.0,
        faultSeverity=0.85, weatherFactor=1.5,
        sigma_bg=0.0025, sigma_corona=0.010,
        seed=7,
        progress_cb=None):
    """اجرای کامل شبیه‌سازی نویز کرونا/عایق دکل بر روی VOR."""
    rng = np.random.default_rng(seed)
    c = 3e8
    lam = c / fVOR
    bearing_rad = np.deg2rad(bearing_deg)

    t = np.arange(0, T, 1/fs)
    N = len(t)

    if progress_cb:
        progress_cb(0.05)

    ref30 = m_ref * np.cos(2 * np.pi * f30 * t)
    fm_phase = 2 * np.pi * fSC * t + m_fm * np.sin(2 * np.pi * f30 * t + bearing_rad)
    fm_component = np.cos(fm_phase)
    vor_clean = 1 + ref30 + 0.33 * fm_component

    if progress_cb:
        progress_cb(0.15)

    pulseRate = 150 + 850 * faultSeverity * weatherFactor
    pulseWidth_us = 8 - 3 * faultSeverity
    pulseWidth = pulseWidth_us * 1e-6
    distanceFactor = 1 / (1 + distance_m / 500)
    voltageFactor = max(0.2, (HV_kV / 400) ** 2)
    noiseScale = (distanceFactor * voltageFactor
                  * weatherFactor * (0.4 + 1.2 * faultSeverity))

    n_bg = rng.standard_normal(N)
    env = 0.65 + 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    corona_cont = sigma_corona * noiseScale * env * n_bg

    if progress_cb:
        progress_cb(0.35)

    nPulses = int(round(pulseRate * T))
    pulseNoise = np.zeros(N)
    pulseTimes = rng.random(nPulses) * T
    pulseAmp = ((0.4 + 1.2 * rng.random(nPulses))
                * (1 + 1.5 * faultSeverity) * noiseScale)

    for k in range(nPulses):
        idx0 = max(0, int(round(pulseTimes[k] * fs)))
        L = max(2, int(round(pulseWidth * fs)))
        idx1 = min(N - 1, idx0 + L - 1)
        Lactual = idx1 - idx0 + 1
        if Lactual < 2:
            continue
        q = np.linspace(-2.5, 2.5, Lactual)
        p = np.exp(-q ** 2)
        polarity = np.sign(rng.standard_normal())
        pulseNoise[idx0:idx1 + 1] += pulseAmp[k] * polarity * p

    if progress_cb:
        progress_cb(0.55)

    coronaNoise = corona_cont + pulseNoise
    rxNoise = sigma_bg * rng.standard_normal(N)
    totalNoise = coronaNoise + rxNoise
    vor_received = vor_clean + totalNoise

    if progress_cb:
        progress_cb(0.65)

    z_ref = analytic_signal_fft(vor_received - np.mean(vor_received))
    ref_mix = z_ref * np.exp(-1j * 2 * np.pi * f30 * t)
    ref_lp = movmean_complex(ref_mix, round(fs / 60))
    ref_phase = np.unwrap(np.angle(ref_lp))

    if progress_cb:
        progress_cb(0.75)

    z_all = analytic_signal_fft(vor_received - np.mean(vor_received))
    z_sc = z_all * np.exp(-1j * 2 * np.pi * fSC * t)
    sc_lp = movmean_complex(z_sc, round(fs / 300))
    phi_inst = np.unwrap(np.angle(sc_lp))

    instFreq = np.zeros(N)
    instFreq[:-1] = np.diff(phi_inst) * fs / (2 * np.pi)
    fm30 = instFreq - fSC
    fm30 = movmean(fm30, round(fs / 100))

    if progress_cb:
        progress_cb(0.85)

    z_fm30 = analytic_signal_fft(fm30 - np.mean(fm30))
    fm30_mix = z_fm30 * np.exp(-1j * 2 * np.pi * f30 * t)
    fm30_lp = movmean_complex(fm30_mix, round(fs / 60))
    var_phase = np.unwrap(np.angle(fm30_lp))

    raw_bearing = np.rad2deg(var_phase - ref_phase)
    raw_bearing = np.mod(raw_bearing + 180, 360) - 180

    clean_error_est = raw_bearing - bearing_deg
    i1 = int(round(1 * fs))
    i2 = int(round(2 * fs))
    i2 = min(i2, N)
    if i2 > i1:
        bias = np.median(clean_error_est[i1:i2])
    else:
        bias = 0.0
    bearing_est = raw_bearing - bias
    bearing_error = np.mod(bearing_est - bearing_deg + 180, 360) - 180

    if progress_cb:
        progress_cb(0.95)

    nFFT = 2 ** int(np.floor(np.log2(min(N, round(2 * fs)))))
    x1 = vor_clean[:nFFT]
    x2 = vor_received[:nFFT]
    X1 = np.fft.fftshift(np.fft.fft(x1))
    X2 = np.fft.fftshift(np.fft.fft(x2))
    f_axis = np.arange(-nFFT // 2, nFFT // 2) * (fs / nFFT)
    P1 = 20 * np.log10(np.abs(X1) / (np.max(np.abs(X1)) + 1e-12) + 1e-12)
    P2 = 20 * np.log10(np.abs(X2) / (np.max(np.abs(X1)) + 1e-12) + 1e-12)

    valid = t > 0.5
    err = bearing_error[valid]
    stats = {
        "mean_err": float(np.mean(err)),
        "rms_err": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_err": float(np.max(np.abs(err))),
        "frac_outside_1deg": float(np.mean(np.abs(err) > 1.0)),
        "frac_outside_3deg": float(np.mean(np.abs(err) > 3.0)),
        "pulseRate": float(pulseRate),
        "pulseWidth_us": float(pulseWidth_us),
        "noiseScale": float(noiseScale),
        "bias_deg": float(bias),
        "lambda_m": float(lam),
        "nPulses": int(nPulses),
        "bearing_deg": float(bearing_deg),
        "fVOR_MHz": float(fVOR / 1e6),
        "HV_kV": float(HV_kV),
        "distance_m": float(distance_m),
        "faultSeverity": float(faultSeverity),
        "weatherFactor": float(weatherFactor),
    }

    return {
        "t": t, "N": N,
        "vor_clean": vor_clean,
        "vor_received": vor_received,
        "coronaNoise": coronaNoise,
        "corona_cont": corona_cont,
        "pulseNoise": pulseNoise,
        "rxNoise": rxNoise,
        "totalNoise": totalNoise,
        "f_axis_kHz": f_axis / 1e3,
        "P1": P1, "P2": P2,
        "bearing_est": bearing_est,
        "bearing_error": bearing_error,
        "valid": valid,
        "stats": stats,
        "fs": fs,
    }


# ------------------------------------------------------------
# 9) توربین بادی
# ------------------------------------------------------------
def calculate_complex_vor(xRx, yRx, R0, xS, yS, K, k):
    """اغتشاش مختلط VOR از یک پراکنده نقطه‌ای."""
    phiS = np.arctan2(yS, xS)
    R1 = np.sqrt(xS ** 2 + yS ** 2)
    R2 = np.sqrt((xRx - xS) ** 2 + (yRx - yS) ** 2)
    Ri = R1 + R2
    Rdirect = R0
    deltaMu = k * (Ri - Rdirect)
    phiAircraft = np.arctan2(yRx, xRx)
    dPhi = phiS - phiAircraft
    dPhi = np.arctan2(np.sin(dPhi), np.cos(dPhi))
    A = K * (Rdirect / Ri)
    E = A * np.cos(dPhi) * np.exp(1j * deltaMu)
    return E


def calculate_vor_error(xRx, yRx, R0, xS, yS, K, k):
    """خطای زاویه از یک پراکنده (بدون جمع فازوری)."""
    E = calculate_complex_vor(xRx, yRx, R0, xS, yS, K, k)
    return np.rad2deg(np.arctan2(np.imag(E), 1 + np.real(E)))


def simulate_wind_turbine(
        fVOR=110e6,
        Rrx=20e3,
        Rfarm=5250.0,
        Ntur=9,
        spacing=240.0,
        K_mast=0.0045,
        K_blade=0.0008,
        phi_step=0.25,
        clamp=3.0,
        blade_radius=15.0,
        blade_orientation_offset=35.0,
        single_blade_radius=30.0,
        progress_cb=None):
    """اجرای کامل شبیه‌سازی توربین بادی + مزرعه بادی."""
    c = 3e8
    lam = c / fVOR
    k = 2 * np.pi / lam

    if progress_cb:
        progress_cb(0.05)

    phi0 = np.arange(0.0, 360.0, phi_step)
    xRx = Rrx * np.cos(np.deg2rad(phi0))
    yRx = Rrx * np.sin(np.deg2rad(phi0))

    farmX = (np.arange(Ntur) - (Ntur - 1) / 2) * spacing
    farmY = Rfarm * np.ones(Ntur)

    if progress_cb:
        progress_cb(0.15)

    errorIdeal = np.zeros_like(phi0)

    xT, yT = Rfarm, 0.0
    errorMast = calculate_vor_error(xRx, yRx, Rrx, xT, yT, K_mast, k)

    if progress_cb:
        progress_cb(0.30)

    bladeAngles = [0.0, 120.0, 240.0]

    errorFull_complex = calculate_complex_vor(
        xRx, yRx, Rrx, xT, yT, K_mast, k)

    for b in range(3):
        xb = xT + single_blade_radius * np.cos(np.deg2rad(bladeAngles[b])) * 0.5
        yb = yT + single_blade_radius * np.sin(np.deg2rad(bladeAngles[b])) * 0.5
        errorFull_complex = errorFull_complex + calculate_complex_vor(
            xRx, yRx, Rrx, xb, yb, K_blade, k)

    errorFull = np.rad2deg(np.arctan2(np.imag(errorFull_complex),
                                       1 + np.real(errorFull_complex)))

    if progress_cb:
        progress_cb(0.50)

    errorFarmMast_complex = np.zeros_like(phi0, dtype=complex)
    for n in range(Ntur):
        errorFarmMast_complex = errorFarmMast_complex + calculate_complex_vor(
            xRx, yRx, Rrx, farmX[n], farmY[n], K_mast, k)

    errorFarmMast = np.rad2deg(np.arctan2(np.imag(errorFarmMast_complex),
                                           1 + np.real(errorFarmMast_complex)))

    if progress_cb:
        progress_cb(0.65)

    errorFarm_complex = np.zeros_like(phi0, dtype=complex)
    for n in range(Ntur):
        errorFarm_complex = errorFarm_complex + calculate_complex_vor(
            xRx, yRx, Rrx, farmX[n], farmY[n], K_mast, k)

        rotorAngle = np.mod(blade_orientation_offset * (n + 1), 360.0)
        for b in range(3):
            angleBlade = rotorAngle + b * 120.0
            xb = farmX[n] + blade_radius * np.cos(np.deg2rad(angleBlade))
            yb = farmY[n] + blade_radius * np.sin(np.deg2rad(angleBlade))
            errorFarm_complex = errorFarm_complex + calculate_complex_vor(
                xRx, yRx, Rrx, xb, yb, K_blade, k)

    errorFarm = np.rad2deg(np.arctan2(np.imag(errorFarm_complex),
                                       1 + np.real(errorFarm_complex)))

    if progress_cb:
        progress_cb(0.90)

    errorMast = np.clip(errorMast, -clamp, clamp)
    errorFull = np.clip(errorFull, -clamp, clamp)
    errorFarmMast = np.clip(errorFarmMast, -clamp, clamp)
    errorFarm = np.clip(errorFarm, -clamp, clamp)

    def rms(x):
        return float(np.sqrt(np.mean(x ** 2)))

    stats = {
        "mast_max": float(np.max(np.abs(errorMast))),
        "mast_rms": rms(errorMast),
        "full_max": float(np.max(np.abs(errorFull))),
        "full_rms": rms(errorFull),
        "farm_mast_max": float(np.max(np.abs(errorFarmMast))),
        "farm_mast_rms": rms(errorFarmMast),
        "farm_max": float(np.max(np.abs(errorFarm))),
        "farm_rms": rms(errorFarm),
        "lambda_m": float(lam),
        "Rfarm_m": float(Rfarm),
        "Ntur": int(Ntur),
        "spacing_m": float(spacing),
        "K_mast": float(K_mast),
        "K_blade": float(K_blade),
        "fVOR_MHz": float(fVOR / 1e6),
        "Rrx_km": float(Rrx / 1000),
    }

    return {
        "phi0": phi0,
        "errorIdeal": errorIdeal,
        "errorMast": errorMast,
        "errorFull": errorFull,
        "errorFarmMast": errorFarmMast,
        "errorFarm": errorFarm,
        "farmX": farmX, "farmY": farmY,
        "Rrx": Rrx, "Rfarm": Rfarm,
        "stats": stats,
        "clamp": clamp,
    }


# ############################################################
#  بخش ۲: توابع کمکی رابط کاربری (مشترک بین صفحات)
# ############################################################
def metric_card(label, value, color=None, sub=None):
    """HTML یک کارت نتیجه."""
    style = f' style="color:{color}"' if color else ""
    sub_html = (f'<div style="color:#9aa7c7;font-size:12px;margin-top:6px">{sub}</div>'
                if sub else "")
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value"{style}>{value}</div>{sub_html}</div>')


def show_card(col, *args, **kwargs):
    with col:
        st.markdown(metric_card(*args, **kwargs), unsafe_allow_html=True)


def frac_color(frac):
    return "#4caf50" if frac < 10 else "#ff9800" if frac < 40 else "#f44336"


def make_progress():
    """نوار پیشرفت + تابع callback."""
    bar = st.progress(0, text="در حال شبیه‌سازی...")

    def cb(frac):
        bar.progress(min(max(frac, 0.0), 1.0),
                     text=f"در حال شبیه‌سازی... {frac*100:.0f}%")
    return bar, cb


def plot_error_vs_range(R, err, exceed, tol, yl, trans_idx=None, trans_text=None):
    """نمودار اصلی خطای زاویه بر حسب فاصله (مشترک بین ۴ صفحه)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([R, R[::-1]]),
        y=np.concatenate([np.full_like(R, tol), np.full_like(R, -tol)]),
        fill="toself", fillcolor="rgba(120,200,120,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"باند ±{tol:.1f}°"))
    fig.add_trace(go.Scatter(
        x=R, y=err, mode="lines", name="خطای زاویه",
        line=dict(color="#4fc3f7", width=1.7)))
    fig.add_trace(go.Scatter(
        x=R[exceed], y=err[exceed], mode="markers",
        name="نقض تلورانس",
        marker=dict(color="#ef5350", size=5)))
    fig.add_hline(y=tol, line=dict(color="white", dash="dash", width=1))
    fig.add_hline(y=-tol, line=dict(color="white", dash="dash", width=1))
    fig.add_hline(y=0, line=dict(color="white", dash="dot", width=1))
    if trans_idx is not None and trans_text:
        fig.add_vline(x=float(R[trans_idx]),
                      line=dict(color="magenta", dash="dashdot"),
                      annotation_text=trans_text,
                      annotation_position="top")
    fig.update_layout(
        template="plotly_dark", height=520,
        xaxis_title="فاصله تا VOR — R [m]  (هواپیما در حال نزدیک شدن ←)",
        yaxis_title="خطای زاویه [deg]",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=70, b=40))
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(range=[-yl, yl])
    st.plotly_chart(fig, use_container_width=True)


def plot_approach_geometry(R, h_ac, objects_traces, height=440, legend_y=-0.15):
    """هندسه صفحه عمودی: اشیاء + مسیر هواپیما + VOR."""
    fig = go.Figure()
    for tr in objects_traces:
        fig.add_trace(tr)
    fig.add_trace(go.Scatter(
        x=R, y=np.full_like(R, h_ac), mode="lines",
        name="مسیر هواپیما", line=dict(color="#ef5350", width=2)))
    fig.add_trace(go.Scatter(
        x=[R[0], R[-1]], y=[h_ac, h_ac], mode="markers+text",
        marker=dict(symbol=["triangle-up", "triangle-down"],
                    size=12, color="red"),
        text=["شروع رویکرد", "نزدیک ایستگاه"],
        textposition=["top center", "top center"],
        name="نقاط کلیدی"))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(symbol="square", size=14, color="black"),
        text=["VOR"], textposition="bottom center", name="VOR"))
    fig.update_layout(
        template="plotly_dark", height=height,
        xaxis_title="فاصله در امتداد رادیال از VOR [m]",
        yaxis_title="ارتفاع AGL [m]",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=legend_y))
    st.plotly_chart(fig, use_container_width=True)


def plot_rho_and_path(R, rho, dDelta):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("|ρ(R)| — ضریب مؤثر پراکندگی",
                        "Δd = d_scatter − d_direct [m]"))
    fig.add_trace(go.Scatter(x=R, y=rho, mode="lines",
                             line=dict(color="#4fc3f7", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=R, y=dDelta, mode="lines",
                             line=dict(color="#ba68c8", width=1.5)), row=2, col=1)
    fig.update_xaxes(title_text="فاصله تا VOR — R [m]", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="فاصله تا VOR — R [m]", autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="|ρ|", row=1, col=1)
    fig.update_yaxes(title_text="Δd [m]", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=520,
                      showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


def plot_rho_multi(R, rho, names, colors, ylabel="|ρᵢ(R)|"):
    fig = go.Figure()
    for i in range(len(names)):
        fig.add_trace(go.Scatter(
            x=R, y=rho[i], mode="lines", name=names[i],
            line=dict(color=colors[i % len(colors)], width=1.6)))
    fig.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="فاصله تا VOR — R [m]",
        yaxis_title=ylabel,
        xaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


PALETTE = ["#4fc3f7", "#66bb6a", "#ffb74d", "#ba68c8",
           "#ef5350", "#26c6da", "#d4e157", "#f06292"]


# ############################################################
#  بخش ۳: صفحات رابط کاربری — هر شبیه‌ساز یک تابع
# ############################################################

# ------------------------------------------------------------
# صفحه ۱: CVOR ایده‌آل
# ------------------------------------------------------------
def page_cvor_ideal():
    st.title("📡 شبیه‌ساز ایده‌آل CVOR")
    st.caption("استخراج زاویه از سیگنال مرجع FM و متغیر AM — تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("⚙️ پارامترها")
        true_bearing = st.slider("زاویه واقعی (درجه)", 0.0, 359.9, 250.0, 0.1)
        st.subheader("سیستم")
        fs = st.number_input("فرکانس نمونه‌برداری fs (Hz)", value=200e3, step=1e3, format="%.0f")
        T = st.number_input("مدت شبیه‌سازی T (s)", value=1.0, step=0.1, min_value=0.2)
        st.subheader("مدولاسیون")
        f_ref = st.number_input("فرکانس مرجع f_ref (Hz)", value=30.0, step=1.0)
        f_sub = st.number_input("زیرحامل FM f_sub (Hz)", value=9960.0, step=10.0)
        dev = st.number_input("انحراف FM (Hz)", value=480.0, step=10.0)
        m_ref = st.slider("دامنه مرجع", 0.0, 1.0, 0.3, 0.01)
        m_var = st.slider("دامنه متغیر", 0.0, 1.0, 0.3, 0.01)
        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if not run:
        st.info("👈 پارامترها را در نوار کناری تنظیم کنید و دکمه «اجرای شبیه‌سازی» را بزنید.")
        return

    with st.spinner("در حال شبیه‌سازی..."):
        res = simulate_cvor(true_bearing, fs, T, f_ref, f_sub, dev, m_ref, m_var)

    c1, c2, c3 = st.columns(3)
    show_card(c1, "زاویه واقعی", f"{true_bearing:.2f}°")
    show_card(c2, "زاویه اندازه‌گیری‌شده", f'{res["measured_bearing_deg"]:.2f}°')
    err = res["bearing_error_deg"]
    color = "#4caf50" if abs(err) < 1 else "#ff9800" if abs(err) < 5 else "#f44336"
    show_card(c3, "خطای زاویه", f"{err:.4f}°", color=color)

    st.divider()

    st.subheader("📶 سیگنال پایه CVOR و طیف آن")
    nshow = int(round(0.05 * fs))
    t, m_t, f_axis, spec = res["t"], res["m_t"], res["f_axis"], res["spectrum"]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("سیگنال پایه ترکیبی CVOR", "طیف سیگنال CVOR"))
    fig.add_trace(go.Scatter(x=t[:nshow], y=m_t[:nshow], mode="lines",
                             line=dict(color="#4fc3f7")), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_axis, y=spec, mode="lines",
                             line=dict(color="#ba68c8")), row=2, col=1)
    fig.update_xaxes(title_text="زمان (s)", row=1, col=1)
    fig.update_xaxes(title_text="فرکانس (Hz)", range=[0, 12000], row=2, col=1)
    fig.update_yaxes(title_text="دامنه", row=1, col=1)
    fig.update_yaxes(title_text="دامنه", row=2, col=1)
    fig.update_layout(height=650, template="plotly_dark",
                      showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔄 بازیابی مؤلفه‌های 30Hz (REF از FM و VAR از AM)")
    tv = res["tv"]; ref_v = res["ref_valid"]; var_rx = res["var_rx"]; valid = res["valid"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=tv, y=ref_v/np.max(np.abs(ref_v)),
                              mode="lines", name="REF (FM)",
                              line=dict(color="#4fc3f7")))
    fig2.add_trace(go.Scatter(x=tv, y=var_rx[valid]/np.max(np.abs(var_rx[valid])),
                              mode="lines", name="VAR (AM)",
                              line=dict(color="#ffb74d")))
    fig2.update_layout(template="plotly_dark", height=400,
                       xaxis_title="زمان (s)", yaxis_title="دامنه نرمال‌شده",
                       margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 خطای زاویه در کل بازه 0 تا 350 درجه")
    tb, errs = res["test_bearings"], res["errors"]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=tb, y=errs, marker_color="#66bb6a",
                          name="خطای زاویه (درجه)"))
    fig3.update_layout(template="plotly_dark", height=400,
                       xaxis_title="زاویه واقعی (درجه)",
                       yaxis_title="خطای زاویه (درجه)",
                       margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 خلاصه پارامترها و خروجی"):
        st.json({**res["params"],
                 "measured_bearing_deg": res["measured_bearing_deg"],
                 "bearing_error_deg": res["bearing_error_deg"]})


# ------------------------------------------------------------
# صفحه ۲: کوه
# ------------------------------------------------------------
def page_mountain():
    st.title("🏔️ شبیه‌ساز اثر توپوگرافی کوه بر خطای زاویه VOR")
    st.caption("مدل پراش تیغه‌چاقو (ITU-R P.526) + پراکندگی نقطه‌ای — تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("⚙️ پارامترهای هندسی")
        D_m = st.number_input("فاصله کوه از VOR — D_m [m]", value=12000.0, step=500.0, min_value=100.0)
        H_m = st.number_input("ارتفاع قله کوه — H_m [m]", value=900.0, step=50.0, min_value=10.0)

        st.subheader("هواپیما")
        h_ac = st.number_input("ارتفاع هواپیما AGL — h_ac [m]", value=2500.0, step=100.0, min_value=100.0)
        R_max = st.number_input("شروع رویکرد — R_max [m]", value=45000.0, step=1000.0, min_value=2000.0)
        R_min = st.number_input("پایان رویکرد — R_min [m]", value=1500.0, step=100.0, min_value=100.0)
        bearing_ac = st.slider("رادیال پرواز [deg]", 0.0, 359.9, 60.0, 0.1)

        st.subheader("ضرایب پراکندگی")
        rho_diff = st.slider("ρ_diff (پراش)", 0.0, 1.0, 0.35, 0.01)
        rho0_refl = st.slider("ρ0_refl (بازتاب)", 0.0, 1.0, 0.15, 0.01)

        st.subheader("سیستم")
        f_c = st.number_input("فرکانس حامل [Hz]", value=113e6, step=1e6, format="%.0f")
        tol = st.number_input("تلورانس ICAO [deg]", value=3.0, step=0.5, min_value=0.1)
        N = st.number_input("تعداد نقاط شبیه‌سازی", value=1200, step=100, min_value=200)

        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if R_min >= R_max:
        st.error("R_min باید کوچکتر از R_max باشد.")
        st.stop()

    if not run:
        st.info("👈 پارامترها را در نوار کناری تنظیم کنید و دکمه «اجرای شبیه‌سازی» را بزنید.")
        return

    with st.spinner("در حال شبیه‌سازی..."):
        res = simulate_mountain_vor(
            bearing_ac_deg=bearing_ac, f_c=f_c,
            D_m=D_m, H_m=H_m,
            rho_diff=rho_diff, rho0_refl=rho0_refl,
            h_ac=h_ac, R_max=R_max, R_min=R_min,
            N=int(N), tol=tol,
        )

    c1, c2, c3 = st.columns(3)
    show_card(c1, "بیشینه |خطا|", f'{res["max_err"]:.2f}°')
    show_card(c2, "در R بیشینه خطا", f'{res["R"][res["max_err_idx"]]/1000:.2f} km')
    frac = 100 * res["frac_exceed"]
    show_card(c3, f"درصد نقض ±{tol:.1f}°", f"{frac:.1f}%", color=frac_color(frac))

    st.info(f"بیشینه خطا در حالت **{res['mode'][res['max_err_idx']]}** رخ می‌دهد.")
    st.divider()

    R = res["R"]
    st.subheader("📐 شکل ۱ — هندسه صفحه عمودی (VOR — کوه — هواپیما)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=[0, res["R_max"]], y=[0, h_ac], mode="lines",
        name="LOS اولیه (نمایشی)",
        line=dict(color="#4fc3f7", dash="dash", width=1)))
    fig1.add_trace(go.Scatter(
        x=[D_m - 1500, D_m, D_m + 1500], y=[0, H_m, 0],
        mode="lines", fill="toself",
        fillcolor="rgba(140,100,75,0.7)",
        line=dict(color="black"), name="کوه"))
    fig1.add_trace(go.Scatter(
        x=R, y=np.full_like(R, h_ac), mode="lines",
        name="مسیر هواپیما", line=dict(color="#ef5350", width=2)))
    fig1.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(symbol="square", size=12, color="black"),
        text=["VOR"], textposition="bottom center", name="VOR"))
    fig1.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="فاصله در امتداد رادیال از VOR [m]",
        yaxis_title="ارتفاع AGL [m]",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 شکل ۲ — ضریب پراکندگی و اختلاف طول مسیر")
    plot_rho_and_path(R, res["rho"], res["dDelta"])

    st.subheader("📉 شکل ۳ — خطای زاویه رمزگشایی‌شده vs فاصله (نتیجه اصلی)")
    err = res["bearing_error_deg"]
    yl = max(6.0, np.ceil(np.max(np.abs(err))) + 1.0)
    plot_error_vs_range(R, err, res["exceed"], tol, yl,
                        res["trans_idx"], "LOS از کوه رد می‌شود")

    with st.expander("📋 خلاصه پارامترها و خروجی"):
        st.json({
            "parameters": {
                "bearing_ac_deg": bearing_ac,
                "f_c_Hz": f_c,
                "D_m_m": D_m, "H_m_m": H_m,
                "h_ac_m": h_ac,
                "R_max_m": R_max, "R_min_m": R_min,
                "rho_diff": rho_diff, "rho0_refl": rho0_refl,
                "tol_deg": tol, "N": int(N),
            },
            "results": {
                "max_abs_error_deg": res["max_err"],
                "R_at_max_error_m": float(R[res["max_err_idx"]]),
                "mode_at_max_error": res["mode"][res["max_err_idx"]],
                "fraction_exceeding_tol_percent": 100 * res["frac_exceed"],
            }
        })


# ------------------------------------------------------------
# صفحه ۳: جنگل
# ------------------------------------------------------------
def page_forest():
    st.title("🌲 شبیه‌ساز خطای زاویه CVOR ناشی از جنگل")
    st.caption("مدل پراکنده نقطه‌ای تک‌باره (Born) + زنجیره کامل VAR/REF FM — تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("🌳 جنگل")
        az_forest = st.slider("آزیموت مرکز جنگل [deg]", 0.0, 359.9, 45.0, 0.5)
        D_in = st.number_input("فاصله لبه نزدیک از VOR [m]", value=300.0, step=50.0, min_value=50.0)
        D_c = st.number_input("فاصله مرکز از VOR [m]", value=1500.0, step=100.0, min_value=200.0)
        density = st.number_input("تراکم درختان [درخت/هکتار]", value=5.0, step=1.0, min_value=0.5)
        sigma_mean = st.number_input("میانگین مقطع پراکندگی هر درخت [m²]", value=10.0, step=1.0, min_value=0.1)
        sigma_sd = st.number_input("انحراف معیار log-normal [dB]", value=3.0, step=0.5, min_value=0.0)
        seed = st.number_input("بذر تصادفی", value=1, step=1, min_value=0)

        st.header("✈️ هواپیما")
        R_air = st.number_input("شعاع دایره پرواز [m]", value=10000.0, step=500.0, min_value=500.0)
        h_air = st.number_input("ارتفاع هواپیما [m]", value=1000.0, step=50.0, min_value=50.0)
        h_ant = st.number_input("ارتفاع آنتن VOR [m]", value=5.0, step=1.0, min_value=1.0)

        st.header("📡 سیستم")
        f0 = st.number_input("فرکانس حامل f0 [Hz]", value=112e6, step=1e6, format="%.0f")
        fLF = st.number_input("فرکانس LF [Hz]", value=30.0, step=1.0)
        fsc = st.number_input("زیرحامل FM [Hz]", value=9960.0, step=10.0)
        mf = st.number_input("شاخص مدولاسیون FM", value=16.0, step=1.0)
        ma = st.slider("عمق مدولاسیون AM", 0.0, 1.0, 0.30, 0.01)

        st.header("🎯 تحلیل")
        tol = st.number_input("تلورانس [deg]", value=1.0, step=0.5, min_value=0.1)
        az_step = st.number_input("گام آزیموت [deg]", value=0.5, step=0.1, min_value=0.05)
        az_zoom0 = st.number_input("شروع اسکن دقیق [deg]", value=60.0, step=1.0)
        az_zoom_len = st.number_input("طول اسکن دقیق [deg]", value=0.5, step=0.1, min_value=0.05)

        st.caption("⚠️ با گام آزیموت کوچک، اجرای شبیه‌سازی طولانی می‌شود.")
        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if D_c <= D_in:
        st.error("D_c باید بزرگتر از D_in باشد (شعاع جنگل > 0).")
        st.stop()

    if not run:
        st.info(
            "👈 پارامترهای جنگل، هواپیما و سیستم را در نوار کناری تنظیم کنید "
            "و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته:** برای اجرای سریع، گام آزیموت را روی 2° یا بیشتر بگذارید.")
        return

    progress_bar, update_progress = make_progress()
    t0 = time.time()
    with st.spinner("در حال اجرا..."):
        res = run_forest_simulation(
            f0=f0, fLF=fLF, fsc=fsc, mf=mf, ma=ma,
            R_air=R_air, h_air=h_air, h_ant=h_ant,
            az_forest=az_forest, D_in=D_in, D_c=D_c,
            density=density, sigma_mean=sigma_mean, sigma_sd=sigma_sd,
            tol=tol, az_step=az_step,
            az_zoom0=az_zoom0, az_zoom_len=az_zoom_len,
            seed=int(seed), progress_cb=update_progress,
        )
    progress_bar.empty()
    elapsed = time.time() - t0

    st.success(f"✅ شبیه‌سازی در {elapsed:.1f} ثانیه انجام شد. "
               f"تعداد درختان: {res['stats']['Nt']}")

    st.markdown("### 📊 خلاصه آماری")
    cols = st.columns(3)
    for ci, s in enumerate(res["stats"]["stats"]):
        show_card(cols[ci], s["name"], f'{s["max_err"]:.2f}°',
                  sub=f'RMS={s["rms"]:.2f}° | 95%={s["p95"]:.2f}°')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("خط مبنای بدون جنگل", f"{res['stats']['base_err']:.5f}°")
    with c2:
        st.metric("حداکثر |A − Eq.(3-14)|", f"{res['stats']['max_A_vs_eq']:.4f}°")
    with c3:
        st.metric("حداکثر |C − A|", f"{res['stats']['max_C_vs_A']:.4f}°")

    st.info(f"درصد آزیموت‌های خارج از تلورانس ±{tol:.1f}° (حالت C): "
            f"**{100 * res['stats']['frac_outside_tol']:.1f}%**")
    st.divider()

    st.subheader("🗺️ شکل ۱ — هندسه جنگل در صفحه افقی")
    forest = res["forest"]
    th = np.linspace(0, 2 * np.pi, 400)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=forest["xt"], y=forest["yt"], mode="markers", name="درختان",
        marker=dict(color="rgba(0,140,0,0.7)", size=3)))
    fig1.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(symbol="triangle-up", size=14, color="black"),
        text=["VOR"], textposition="top center", name="VOR"))
    fig1.add_trace(go.Scatter(
        x=forest["xc"] + forest["R_f"] * np.cos(th),
        y=forest["yc"] + forest["R_f"] * np.sin(th),
        mode="lines", line=dict(color="green", width=1.5),
        name=f"مرز جنگل (r={forest['R_f']:.0f} m)"))
    fig1.add_trace(go.Scatter(
        x=D_in * np.sin(th), y=D_in * np.cos(th),
        mode="lines", line=dict(color="black", dash="dot", width=1),
        name=f"لبه نزدیک ({D_in:.0f} m)"))
    fig1.add_trace(go.Scatter(
        x=[0, forest["xc"]], y=[0, forest["yc"]], mode="lines",
        line=dict(color="black", dash="dash", width=1),
        name="خط VOR → مرکز جنگل"))
    fig1.update_layout(
        template="plotly_dark", height=600,
        xaxis_title="شرق [m]", yaxis_title="شمال [m]",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📈 شکل ۲ — خطای زاویه CVOR در طول آزیموت")
    fig2 = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "خطای زاویه سه حالت (A: VAR، B: REF، C: هر دو) + معادله (3-14)",
            "اسکن دقیق (zoom) — موج‌های سریع معادله (3-14)",
            "خطای فاز مرجع 30Hz (REF) — حالت‌های B و C",
            "توزیع خطای زاویه — حالت C"),
        vertical_spacing=0.07,
        row_heights=[0.28, 0.24, 0.24, 0.24])

    az = res["az"]; eps_t = res["eps_t"]; eps_f = res["eps_f"]; dREF = res["dREF"]

    colors = ["#ef5350", "#66bb6a", "#42a5f5"]
    for ci in range(3):
        fig2.add_trace(go.Scatter(
            x=az, y=eps_t[ci], mode="lines", name=res["names"][ci],
            line=dict(color=colors[ci], width=1.0)), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=az, y=eps_f, mode="lines", name="Eq. (3-14)",
        line=dict(color="white", dash="dash", width=0.8)), row=1, col=1)
    fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot", width=1), row=1, col=1)
    fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot", width=1), row=1, col=1)

    fig2.add_trace(go.Scatter(
        x=res["az_z"], y=res["eps_z"], mode="lines",
        name="Zoom", line=dict(color="white", width=1.0)), row=2, col=1)
    fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot", width=1), row=2, col=1)
    fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot", width=1), row=2, col=1)

    fig2.add_trace(go.Scatter(
        x=az, y=dREF[1], mode="lines", name="REF err — B",
        line=dict(color="#66bb6a", width=1.0)), row=3, col=1)
    fig2.add_trace(go.Scatter(
        x=az, y=dREF[2], mode="lines", name="REF err — C",
        line=dict(color="#42a5f5", width=1.0)), row=3, col=1)

    fig2.add_trace(go.Histogram(
        x=eps_t[2], nbinsx=40,
        marker=dict(color="rgba(60,90,220,0.75)"),
        name="توزیع خطا (C)"), row=4, col=1)
    fig2.add_vline(x=tol, line=dict(color="magenta", dash="dot", width=1), row=4, col=1)
    fig2.add_vline(x=-tol, line=dict(color="magenta", dash="dot", width=1), row=4, col=1)

    for r in (1, 2, 3):
        fig2.update_xaxes(title_text="آزیموت هواپیما [deg]", row=r, col=1)
    fig2.update_xaxes(title_text="خطای زاویه [deg]", row=4, col=1)
    fig2.update_yaxes(title_text="خطای زاویه [deg]", row=1, col=1)
    fig2.update_yaxes(title_text="خطای زاویه [deg]", row=2, col=1)
    fig2.update_yaxes(title_text="خطای فاز REF [deg]", row=3, col=1)
    fig2.update_yaxes(title_text="شمارش", row=4, col=1)
    fig2.update_layout(
        template="plotly_dark", height=1200, showlegend=True,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=50, r=20, t=80, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 جدول آماری کامل"):
        rows = []
        for s in res["stats"]["stats"]:
            rows.append({
                "حالت": s["name"],
                "max|err| [deg]": f"{s['max_err']:.3f}",
                "RMS [deg]": f"{s['rms']:.3f}",
                "95% [deg]": f"{s['p95']:.3f}",
                "max|REF phase| [deg]": f"{s['max_dref']:.5f}",
            })
        ef = res["stats"]["eps_f_stats"]
        rows.append({
            "حالت": "Eq. (3-14)",
            "max|err| [deg]": f"{ef['max_err']:.3f}",
            "RMS [deg]": f"{ef['rms']:.3f}",
            "95% [deg]": f"{ef['p95']:.3f}",
            "max|REF phase| [deg]": "—",
        })
        st.dataframe(rows, use_container_width=True)


# ------------------------------------------------------------
# صفحه ۴: یک درخت
# ------------------------------------------------------------
def page_single_tree():
    st.title("🌳 شبیه‌ساز خطای CVOR ناشی از یک درخت")
    st.caption("زنجیره کامل سیگنال VAR/REF FM + معادله تحلیلی (3-14) — تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("🌳 درخت")
        d_tree = st.number_input("فاصله درخت از VOR [m]", value=100.0, step=10.0, min_value=10.0)
        az_tree = st.slider("آزیموت درخت [deg]", 0.0, 359.9, 45.0, 0.5)
        h_tree = st.number_input("ارتفاع مؤثر پراکندگی [m]", value=10.0, step=1.0, min_value=1.0)
        rho = st.slider("ضریب پراکندگی ρ", 0.0, 1.0, 0.10, 0.01)

        st.header("✈️ هواپیما")
        R_air = st.number_input("شعاع دایره پرواز [m]", value=10000.0, step=500.0, min_value=500.0)
        h_air = st.number_input("ارتفاع هواپیما [m]", value=1000.0, step=50.0, min_value=50.0)
        h_ant = st.number_input("ارتفاع آنتن VOR [m]", value=5.0, step=1.0, min_value=1.0)

        st.header("📡 سیستم")
        f0 = st.number_input("فرکانس حامل f0 [Hz]", value=112e6, step=1e6, format="%.0f")
        fLF = st.number_input("فرکانس LF [Hz]", value=30.0, step=1.0)
        fsc = st.number_input("زیرحامل FM [Hz]", value=9960.0, step=10.0)
        mf = st.number_input("شاخص مدولاسیون FM", value=16.0, step=1.0)
        ma = st.slider("عمق مدولاسیون AM", 0.0, 1.0, 0.30, 0.01)

        st.header("🎯 تحلیل")
        tol = st.number_input("تلورانس [deg]", value=1.0, step=0.5, min_value=0.1)
        az_step = st.number_input("گام آزیموت [deg]", value=0.25, step=0.05, min_value=0.05)

        st.caption("⚠️ گام کوچک‌تر = دقت بیشتر ولی کندتر.")
        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if not run:
        st.info(
            "👈 پارامترهای درخت، هواپیما و سیستم را در نوار کناری تنظیم کنید "
            "و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**پیشنهاد:** برای اجرای سریع، گام آزیموت = 0.5°، "
            "برای دقت بالاتر گام = 0.1°.")
        return

    progress_bar, update_progress = make_progress()
    t0 = time.time()
    with st.spinner("در حال اجرا..."):
        res = run_single_tree_simulation(
            f0=f0, fLF=fLF, fsc=fsc, mf=mf, ma=ma,
            R_air=R_air, h_air=h_air, h_ant=h_ant,
            d_tree=d_tree, az_tree=az_tree, h_tree=h_tree,
            rho=rho, tol=tol, az_step=az_step,
            progress_cb=update_progress,
        )
    progress_bar.empty()
    elapsed = time.time() - t0

    st.success(f"✅ شبیه‌سازی در {elapsed:.1f} ثانیه انجام شد.")

    if res["max_dtheta"] > np.pi / 2:
        st.warning(
            f"⚠️ تغییر فاز مولتی‌پات در هر گام آزیموت = "
            f"{res['max_dtheta']:.2f} rad (> π/2). "
            f"گام آزیموت را کوچک‌تر کنید تا ریپل سریع به‌درستی نمونه‌برداری شود.")
    else:
        st.info(f"✅ تغییر فاز مولتی‌پات در هر گام آزیموت = "
                f"{res['max_dtheta']:.2f} rad (خوب).")

    st.markdown("### 📊 خلاصه آماری")
    cols = st.columns(3)
    for ci, s in enumerate(res["stats"]):
        show_card(cols[ci], s["name"], f'{s["max_err"]:.2f}°',
                  sub=f'RMS={s["rms"]:.2f}° | max|REF|={s["max_dref"]:.4f}°')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("خط مبنای بدون درخت", f"{res['base_err']:.5f}°")
    with c2:
        st.metric("حداکثر |A − Eq.(3-14)|", f"{res['max_A_vs_eq']:.4f}°")
    with c3:
        st.metric("حداکثر |C − A|", f"{res['max_C_vs_A']:.4f}°")

    st.caption(
        f"λ = {res['lambda']:.3f} m | "
        f"حداکثر تأخیر اضافی = {res['max_tau_us']:.3f} µs | "
        f"حداکثر |خطای معادله تحلیلی| = {res['eq_max']:.4f}°")
    st.divider()

    st.subheader("📈 خطای زاویه در طول آزیموت")
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f"خطای زاویه سه حالت (درخت در {d_tree:.0f} m، آزیموت {az_tree:.1f}°)",
            "اسکن دقیق ۱۰۰ تا ۱۲۵ درجه",
            "خطای فاز مرجع 30Hz (REF) — حالت‌های B و C",
            "طیف پاکت اطراف زیرحامل 9960 Hz (حدود باند پاس در قرمز)"),
        vertical_spacing=0.08,
        row_heights=[0.28, 0.24, 0.22, 0.26])

    az = res["az"]; eps_t = res["eps_t"]; eps_f = res["eps_f"]; dREF = res["dREF"]
    tol = res["tol"]

    colors = ["#ef5350", "#66bb6a", "#42a5f5"]
    for ci in range(3):
        fig.add_trace(go.Scatter(
            x=az, y=eps_t[ci], mode="lines", name=res["names"][ci],
            line=dict(color=colors[ci], width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=az, y=eps_f, mode="lines", name="Eq. (3-14)",
        line=dict(color="white", dash="dash", width=1.0)), row=1, col=1)
    fig.add_hline(y=tol, line=dict(color="magenta", dash="dot", width=1), row=1, col=1)
    fig.add_hline(y=-tol, line=dict(color="magenta", dash="dot", width=1), row=1, col=1)

    zr = (az >= 100) & (az <= 125)
    if np.any(zr):
        fig.add_trace(go.Scatter(
            x=az[zr], y=eps_t[0][zr], mode="lines+markers",
            name="A (zoom)", line=dict(color="#ef5350", width=1.0),
            marker=dict(size=3)), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=az[zr], y=eps_t[2][zr], mode="lines",
            name="C (zoom)", line=dict(color="#42a5f5", width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=az[zr], y=eps_f[zr], mode="lines", name="Eq. 3-14 (zoom)",
            line=dict(color="white", dash="dash", width=1.0)), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=az, y=dREF[1], mode="lines", name="REF err — B",
        line=dict(color="#66bb6a", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=az, y=dREF[2], mode="lines", name="REF err — C",
        line=dict(color="#42a5f5", width=1.2)), row=3, col=1)

    f_axis = res["f_axis"]; E = res["E"]
    sel = (f_axis > 8000) & (f_axis < 12000)
    fig.add_trace(go.Scatter(
        x=f_axis[sel], y=E[sel], mode="lines", name="|Envelope spectrum|",
        line=dict(color="#42a5f5", width=1.0)), row=4, col=1)
    fig.add_vline(x=res["fsc"] - res["bw"],
                  line=dict(color="red", dash="dash", width=1), row=4, col=1)
    fig.add_vline(x=res["fsc"] + res["bw"],
                  line=dict(color="red", dash="dash", width=1), row=4, col=1)

    for r in (1, 2, 3):
        fig.update_xaxes(title_text="آزیموت هواپیما [deg]", row=r, col=1)
    fig.update_xaxes(title_text="فرکانس [Hz]", row=4, col=1)
    fig.update_yaxes(title_text="خطای زاویه [deg]", row=1, col=1)
    fig.update_yaxes(title_text="خطای زاویه [deg]", row=2, col=1)
    fig.update_yaxes(title_text="خطای فاز REF [deg]", row=3, col=1)
    fig.update_yaxes(title_text="|طیف پاکت|", row=4, col=1)
    fig.update_layout(
        template="plotly_dark", height=1300, showlegend=True,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=50, r=20, t=80, b=40))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 جدول آماری کامل"):
        rows = []
        for s in res["stats"]:
            rows.append({
                "حالت": s["name"],
                "max|err| [deg]": f"{s['max_err']:.4f}",
                "RMS [deg]": f"{s['rms']:.4f}",
                "max|REF phase| [deg]": f"{s['max_dref']:.5f}",
            })
        rows.append({
            "حالت": "Eq. (3-14)",
            "max|err| [deg]": f"{res['eq_max']:.4f}",
            "RMS [deg]": "—",
            "max|REF phase| [deg]": "—",
        })
        st.dataframe(rows, use_container_width=True)

        st.markdown("**مقایسه‌ها:**")
        st.markdown(f"- حداکثر |A − Eq.(3-14)| = **{res['max_A_vs_eq']:.4f}°**")
        st.markdown(f"- حداکثر |C − A| = **{res['max_C_vs_A']:.4f}°**")
        st.markdown(f"- خط مبنای بدون درخت = **{res['base_err']:.5f}°**")
        st.markdown(f"- حداکثر تأخیر اضافی τ = **{res['max_tau_us']:.3f} µs**")


# ------------------------------------------------------------
# صفحه ۵: یک ساختمان
# ------------------------------------------------------------
def page_single_building():
    st.title("🏢 شبیه‌ساز اثر ساختمان تنها بر خطای زاویه VOR")
    st.caption("مدل پراش تیغه‌چاقو (ITU-R P.526) + بازتاب آینه‌ای با مقیاس مساحت نما — "
               "تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("🏢 هندسه ساختمان")
        D_b = st.number_input("فاصله ساختمان از VOR — D_b [m]", value=220.0, step=10.0, min_value=10.0)
        H_b = st.number_input("ارتفاع ساختمان — H_b [m]", value=15.0, step=1.0, min_value=1.0)
        W_b = st.number_input("عرض نمای ساختمان — W_b [m]", value=20.0, step=1.0, min_value=1.0)

        st.header("✈️ هواپیما")
        h_ac = st.number_input("ارتفاع هواپیما AGL — h_ac [m]", value=300.0, step=50.0, min_value=50.0)
        R_max = st.number_input("شروع رویکرد — R_max [m]", value=15000.0, step=1000.0, min_value=2000.0)
        R_min = st.number_input("پایان رویکرد — R_min [m]", value=300.0, step=50.0, min_value=100.0)

        st.header("🔧 ضرایب بازتاب/پراش")
        base_rho = st.slider("ρ پایه بازتاب آینه‌ای", 0.0, 1.0, 0.12, 0.01)
        area_ref = st.number_input("مساحت مرجع نما — area_ref [m²]", value=200.0, step=10.0, min_value=10.0)
        rho_diff = st.slider("ρ_diff — بازتاب پراش لبه", 0.0, 1.0, 0.25, 0.01)

        st.header("📡 سیستم")
        f_c = st.number_input("فرکانس حامل [Hz]", value=113e6, step=1e6, format="%.0f")
        tol = st.number_input("تلورانس ICAO [deg]", value=1.0, step=0.5, min_value=0.1)
        N = st.number_input("تعداد نقاط شبیه‌سازی", value=1200, step=100, min_value=200)

        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if R_min >= R_max:
        st.error("R_min باید کوچکتر از R_max باشد.")
        st.stop()

    if not run:
        st.info(
            "👈 پارامترهای ساختمان، هواپیما و سیستم را در نوار کناری تنظیم کنید "
            "و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته:** این سناریو با تلورانس سخت‌گیرانه ±۱° ارزیابی می‌شود "
            "(مطابق کد MATLAB اصلی).")
        return

    with st.spinner("در حال شبیه‌سازی..."):
        res = simulate_building_vor(
            f_c=f_c, D_b=D_b, H_b=H_b, W_b=W_b,
            base_rho=base_rho, area_ref=area_ref, rho_diff=rho_diff,
            h_ac=h_ac, R_max=R_max, R_min=R_min,
            N=int(N), tol=tol,
        )

    c1, c2, c3 = st.columns(3)
    show_card(c1, "بیشینه |خطا|", f'{res["max_err"]:.2f}°')
    show_card(c2, "در R بیشینه خطا", f'{res["R"][res["max_err_idx"]] / 1000:.2f} km')
    frac = 100 * res["frac_exceed"]
    show_card(c3, f"درصد نقض ±{tol:.1f}°", f"{frac:.1f}%", color=frac_color(frac))

    st.info(
        f"بیشینه خطا در حالت **{res['mode'][res['max_err_idx']]}** رخ می‌دهد. "
        f"ضریب مقیاس مساحت نما = **{res['area_factor']:.3f}** "
        f"(H×W = {H_b:.0f}×{W_b:.0f} = {H_b*W_b:.0f} m² نسبت به {area_ref:.0f} m²)")
    st.divider()

    R = res["R"]
    st.subheader("📐 شکل ۱ — هندسه صفحه عمودی (VOR — ساختمان — هواپیما)")
    building = go.Scatter(
        x=[D_b - 15, D_b - 15, D_b + 15, D_b + 15],
        y=[0, H_b, H_b, 0],
        mode="lines", fill="toself",
        fillcolor="rgba(100,140,200,0.7)",
        line=dict(color="black"), name="ساختمان")
    plot_approach_geometry(R, h_ac, [building], height=420, legend_y=-0.15)

    st.subheader("📊 شکل ۲ — ضریب پراکندگی و اختلاف طول مسیر")
    plot_rho_and_path(R, res["rho"], res["dDelta"])

    st.subheader("📉 شکل ۳ — خطای زاویه رمزگشایی‌شده vs فاصله (نتیجه اصلی)")
    err = res["bearing_error_deg"]
    yl = max(2.0, np.ceil(np.max(np.abs(err))) + 0.5)
    plot_error_vs_range(R, err, res["exceed"], tol, yl,
                        res["trans_idx"], "LOS از ساختمان رد می‌شود")

    with st.expander("📋 خلاصه پارامترها و خروجی"):
        st.markdown("**پارامترهای ساختمان:**")
        st.markdown(f"- فاصله از VOR: **{D_b:.0f} m**")
        st.markdown(f"- ارتفاع: **{H_b:.1f} m**")
        st.markdown(f"- عرض نما: **{W_b:.1f} m**")
        st.markdown(f"- مساحت نما: **{H_b*W_b:.1f} m²**")
        st.markdown(f"- ضریب مقیاس مساحت: **{res['area_factor']:.3f}**")
        st.markdown(f"- λ = **{res['lambda']:.3f} m**")
        st.json({
            "parameters": {
                "f_c_Hz": f_c,
                "D_b_m": D_b, "H_b_m": H_b, "W_b_m": W_b,
                "h_ac_m": h_ac,
                "R_max_m": R_max, "R_min_m": R_min,
                "base_rho": base_rho, "area_ref_m2": area_ref,
                "rho_diff": rho_diff,
                "tol_deg": tol, "N": int(N),
            },
            "results": {
                "max_abs_error_deg": res["max_err"],
                "R_at_max_error_m": float(R[res["max_err_idx"]]),
                "mode_at_max_error": res["mode"][res["max_err_idx"]],
                "fraction_exceeding_tol_percent": 100 * res["frac_exceed"],
            }
        })


# ------------------------------------------------------------
# صفحه ۶: چند ساختمان
# ------------------------------------------------------------
def page_multi_building():
    st.title("🏙️ شبیه‌ساز اثر چند ساختمان بر خطای زاویه VOR")
    st.caption("جمع فازوری چند پراکنده مستقل روی کانال VAR — تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("🏢 مجموعه ساختمان‌ها")
        st.caption("هر ردیف یک ساختمان: نام، D، H، W")

        default_buildings = pd.DataFrame([
            {"name": "B1 Tall&narrow", "D": 180.0, "H": 28.0, "W": 8.0},
            {"name": "B2 Short&squat", "D": 260.0, "H": 7.0, "W": 12.0},
            {"name": "B3 Wide", "D": 330.0, "H": 12.0, "W": 35.0},
        ])

        edited = st.data_editor(
            default_buildings,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("نام", width="medium"),
                "D": st.column_config.NumberColumn("فاصله D [m]", min_value=10.0, step=10.0),
                "H": st.column_config.NumberColumn("ارتفاع H [m]", min_value=1.0, step=1.0),
                "W": st.column_config.NumberColumn("عرض W [m]", min_value=1.0, step=1.0),
            },
            key="building_table"
        )

        st.header("🔧 ضرایب بازتاب/پراش")
        base_rho = st.slider("ρ پایه بازتاب آینه‌ای", 0.0, 1.0, 0.12, 0.01)
        area_ref = st.number_input("مساحت مرجع نما — area_ref [m²]", value=200.0, step=10.0, min_value=10.0)
        rho_diff = st.slider("ρ_diff — بازتاب پراش لبه", 0.0, 1.0, 0.25, 0.01)

        st.header("✈️ هواپیما")
        h_ac = st.number_input("ارتفاع هواپیما AGL — h_ac [m]", value=300.0, step=50.0, min_value=50.0)
        R_max = st.number_input("شروع رویکرد — R_max [m]", value=15000.0, step=1000.0, min_value=2000.0)
        R_min = st.number_input("پایان رویکرد — R_min [m]", value=300.0, step=50.0, min_value=100.0)

        st.header("📡 سیستم")
        f_c = st.number_input("فرکانس حامل [Hz]", value=113e6, step=1e6, format="%.0f")
        tol = st.number_input("تلورانس ICAO [deg]", value=1.0, step=0.5, min_value=0.1)
        N = st.number_input("تعداد نقاط شبیه‌سازی", value=1200, step=100, min_value=200)

        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if R_min >= R_max:
        st.error("R_min باید کوچکتر از R_max باشد.")
        st.stop()

    if not run:
        st.info(
            "👈 جدول ساختمان‌ها را در نوار کناری ویرایش کنید "
            "(می‌توانید ردیف اضافه/حذف کنید) و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته:** می‌توانید تا هر تعداد ساختمان که می‌خواهید اضافه کنید — "
            "مدل به‌طور خودکار همه فازورها را جمع می‌کند.")
        return

    buildings = []
    for _, row in edited.iterrows():
        if (pd.isna(row["name"]) or pd.isna(row["D"])
                or pd.isna(row["H"]) or pd.isna(row["W"])):
            continue
        buildings.append({
            "name": str(row["name"]),
            "D": float(row["D"]),
            "H": float(row["H"]),
            "W": float(row["W"]),
        })

    if len(buildings) == 0:
        st.error("حداقل یک ساختمان لازم است.")
        st.stop()

    with st.spinner("در حال شبیه‌سازی..."):
        res = simulate_multi_building(
            buildings=buildings, f_c=f_c,
            base_rho=base_rho, area_ref=area_ref, rho_diff=rho_diff,
            h_ac=h_ac, R_max=R_max, R_min=R_min,
            N=int(N), tol=tol,
        )

    c1, c2, c3 = st.columns(3)
    show_card(c1, "بیشینه |خطا| ترکیبی", f'{res["max_err"]:.2f}°')
    show_card(c2, "در R بیشینه خطا", f'{res["R"][res["max_err_idx"]] / 1000:.2f} km')
    frac = 100 * res["frac_exceed"]
    show_card(c3, f"درصد نقض ±{tol:.1f}°", f"{frac:.1f}%", color=frac_color(frac))

    st.info(
        f"**{len(buildings)} ساختمان** در حال شبیه‌سازی | "
        f"λ = **{res['lambda']:.3f} m** | "
        f"ضریب مقیاس مساحت هر ساختمان: "
        + "، ".join(f"{n}: **{af:.2f}**"
                    for n, af in zip(res["names"], res["area_factor"])))
    st.divider()

    R = res["R"]
    D_arr = res["D_arr"]; H_arr = res["H_arr"]; W_arr = res["W_arr"]
    names = res["names"]

    st.subheader("📐 شکل ۱ — هندسه صفحه عمودی (VOR — ساختمان‌ها — هواپیما)")
    traces = []
    for i in range(res["Nb"]):
        half_w = 10.0
        traces.append(go.Scatter(
            x=[D_arr[i] - half_w, D_arr[i] - half_w,
               D_arr[i] + half_w, D_arr[i] + half_w],
            y=[0, H_arr[i], H_arr[i], 0],
            mode="lines", fill="toself",
            fillcolor=PALETTE[i % len(PALETTE)],
            line=dict(color="black"),
            name=f"{names[i]} (D={D_arr[i]:.0f}، H={H_arr[i]:.0f}، W={W_arr[i]:.0f})",
            opacity=0.8))
    plot_approach_geometry(R, res["h_ac"], traces, height=460)

    st.subheader("📊 شکل ۲ — ضریب پراکندگی |ρᵢ(R)| هر ساختمان")
    plot_rho_multi(R, res["rho"], names, PALETTE)

    st.subheader("📉 شکل ۳ — خطای ترکیبی زاویه vs فاصله (نتیجه اصلی)")
    err = res["bearing_error_deg"]
    yl = max(2.0, np.ceil(np.max(np.abs(err))) + 0.5)
    plot_error_vs_range(R, err, res["exceed"], tol, yl)

    with st.expander("🔍 تحلیل تفکیکی ساختمان‌ها (مشارکت هر کدام)"):
        st.markdown(
            "برای دیدن سهم هر ساختمان به‌تنهایی، ستون «حالت غالب» و «max |ρᵢ|» را ببینید.")
        rows = []
        for i in range(res["Nb"]):
            mode_counts = {}
            for m in res["mode"][i]:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            dominant = max(mode_counts, key=mode_counts.get)
            rows.append({
                "ساختمان": names[i],
                "D [m]": f"{D_arr[i]:.0f}",
                "H [m]": f"{H_arr[i]:.0f}",
                "W [m]": f"{W_arr[i]:.0f}",
                "مساحت نما [m²]": f"{H_arr[i]*W_arr[i]:.0f}",
                "ضریب مقیاس": f"{res['area_factor'][i]:.3f}",
                "max |ρᵢ|": f"{np.max(res['rho'][i]):.4f}",
                "حالت غالب": dominant,
            })
        st.dataframe(rows, use_container_width=True)
        st.markdown("**جدول کامل ساختمان‌ها:**")
        st.dataframe(edited, use_container_width=True)

    with st.expander("📋 خلاصه پارامترها و خروجی"):
        st.json({
            "buildings": buildings,
            "parameters": {
                "f_c_Hz": f_c, "h_ac_m": h_ac,
                "R_max_m": R_max, "R_min_m": R_min,
                "base_rho": base_rho, "area_ref_m2": area_ref,
                "rho_diff": rho_diff,
                "tol_deg": tol, "N": int(N),
            },
            "results": {
                "max_abs_combined_error_deg": res["max_err"],
                "R_at_max_error_m": float(R[res["max_err_idx"]]),
                "fraction_exceeding_tol_percent": 100 * res["frac_exceed"],
            }
        })


# ------------------------------------------------------------
# صفحه ۷: دکل انتقال نیرو
# ------------------------------------------------------------
def page_power_tower():
    st.title("🗼 شبیه‌ساز اثر دکل انتقال نیرو بر خطای زاویه VOR")
    st.caption("۴ المان ساختاری فولادی مستقل با جمع فازوری روی کانال VAR — "
               "تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("🗼 موقعیت دکل")
        D_t = st.number_input("فاصله دکل از VOR — D_t [m]", value=250.0, step=10.0, min_value=10.0)

        st.header("🏗️ المان‌های ساختاری")
        st.caption("هر ردیف یک المان: نام، ارتفاع، عرض مؤثر")

        default_elements = pd.DataFrame([
            {"name": "Mast tip", "h": 35.0, "w": 3.0},
            {"name": "Arm 1 (top)", "h": 32.0, "w": 9.0},
            {"name": "Arm 2 (mid)", "h": 26.0, "w": 11.0},
            {"name": "Arm 3 (low)", "h": 20.0, "w": 13.0},
        ])

        edited = st.data_editor(
            default_elements,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("نام", width="medium"),
                "h": st.column_config.NumberColumn("ارتفاع h [m]", min_value=1.0, step=1.0),
                "w": st.column_config.NumberColumn("عرض مؤثر w [m]", min_value=0.5, step=0.5),
            },
            key="element_table"
        )

        st.header("🔧 ضرایب بازتاب/پراش فولاد")
        base_rho = st.slider("ρ پایه بازتاب آینه‌ای (فولاد)", 0.0, 1.0, 0.20, 0.01)
        area_ref = st.number_input("مساحت مرجع [m²]", value=60.0, step=5.0, min_value=5.0)
        rho_diff = st.slider("ρ_diff — بازتاب پراش لبه فلزی", 0.0, 1.0, 0.30, 0.01)
        arm_depth = st.number_input("عمق مؤثر بازو [m]", value=1.5, step=0.5, min_value=0.1)

        st.header("✈️ هواپیما")
        h_ac = st.number_input("ارتفاع هواپیما AGL — h_ac [m]", value=300.0, step=50.0, min_value=50.0)
        R_max = st.number_input("شروع رویکرد — R_max [m]", value=15000.0, step=1000.0, min_value=2000.0)
        R_min = st.number_input("پایان رویکرد — R_min [m]", value=300.0, step=50.0, min_value=100.0)

        st.header("📡 سیستم")
        f_c = st.number_input("فرکانس حامل [Hz]", value=113e6, step=1e6, format="%.0f")
        tol = st.number_input("تلورانس [deg]", value=1.0, step=0.5, min_value=0.1)
        N = st.number_input("تعداد نقاط شبیه‌سازی", value=1200, step=100, min_value=200)

        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if R_min >= R_max:
        st.error("R_min باید کوچکتر از R_max باشد.")
        st.stop()

    if not run:
        st.info(
            "👈 موقعیت دکل و المان‌های ساختاری را در نوار کناری تنظیم کنید "
            "و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته:** برخلاف مدل چند-ساختمان، اینجا همه المان‌ها در **یک "
            "فاصله D_t** قرار دارند (چون همگی اجزای یک دکل واحدند). "
            "این باعث می‌شود فاز نسبی آن‌ها عمدتاً از **ارتفاع متفاوت** ناشی شود، "
            "نه از فاصله افقی.")
        return

    elements = []
    for _, row in edited.iterrows():
        if pd.isna(row["name"]) or pd.isna(row["h"]) or pd.isna(row["w"]):
            continue
        elements.append({"name": str(row["name"]),
                         "h": float(row["h"]), "w": float(row["w"])})

    if len(elements) == 0:
        st.error("حداقل یک المان ساختاری لازم است.")
        st.stop()

    with st.spinner("در حال شبیه‌سازی..."):
        res = simulate_power_tower(
            elements=elements, D_t=D_t, f_c=f_c,
            base_rho=base_rho, area_ref=area_ref, rho_diff=rho_diff,
            arm_depth=arm_depth,
            h_ac=h_ac, R_max=R_max, R_min=R_min,
            N=int(N), tol=tol,
        )

    c1, c2, c3 = st.columns(3)
    show_card(c1, "بیشینه |خطا| ترکیبی", f'{res["max_err"]:.2f}°')
    show_card(c2, "در R بیشینه خطا", f'{res["R"][res["max_err_idx"]] / 1000:.2f} km')
    frac = 100 * res["frac_exceed"]
    show_card(c3, f"درصد نقض ±{tol:.1f}°", f"{frac:.1f}%", color=frac_color(frac))

    st.info(
        f"**{len(elements)} المان** در فاصله **{D_t:.0f} m** از VOR | "
        f"λ = **{res['lambda']:.3f} m** | "
        f"base_rho = **{base_rho:.2f}** (فولاد) | "
        f"area_ref = **{area_ref:.0f} m²**")
    st.divider()

    R = res["R"]
    h_e = res["h_e"]; w_e = res["w_e"]; names = res["names"]; Ne = res["Ne"]

    st.subheader("📐 شکل ۱ — هندسه صفحه عمودی (VOR — دکل — هواپیما)")
    traces = [go.Scatter(
        x=[D_t, D_t], y=[0, float(np.max(h_e))],
        mode="lines", name="بدنه دکل",
        line=dict(color="white", width=3))]
    for i in range(Ne):
        traces.append(go.Scatter(
            x=[D_t - w_e[i] / 2, D_t + w_e[i] / 2],
            y=[h_e[i], h_e[i]],
            mode="lines+text",
            name=f"{names[i]} (h={h_e[i]:.0f}، w={w_e[i]:.0f})",
            line=dict(color=PALETTE[i % len(PALETTE)], width=5),
            text=["", f"{names[i]}"],
            textposition="middle right",
            textfont=dict(color=PALETTE[i % len(PALETTE)], size=9)))
    plot_approach_geometry(R, res["h_ac"], traces, height=460)

    st.subheader("📊 شکل ۲ — ضریب پراکندگی |ρᵢ(R)| هر المان ساختاری")
    plot_rho_multi(R, res["rho"], names, PALETTE)

    st.subheader("📉 شکل ۳ — خطای ترکیبی زاویه vs فاصله (نتیجه اصلی)")
    err = res["bearing_error_deg"]
    yl = max(2.0, np.ceil(np.max(np.abs(err))) + 0.5)
    plot_error_vs_range(R, err, res["exceed"], tol, yl)

    with st.expander("🔍 تحلیل تفکیکی المان‌های دکل"):
        st.markdown(
            "سهم هر المان ساختاری در جمع برداری نهایی — "
            "چون همه در یک فاصله D_t هستند، فاز نسبی آن‌ها "
            "عمدتاً از ارتفاع متفاوتشان می‌آید.")
        rows = []
        for i in range(Ne):
            mode_counts = {}
            for m in res["mode"][i]:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            dominant = max(mode_counts, key=mode_counts.get) if mode_counts else "—"
            rows.append({
                "المان": names[i],
                "ارتفاع h [m]": f"{h_e[i]:.0f}",
                "عرض w [m]": f"{w_e[i]:.0f}",
                "مساحت مؤثر [m²]": f"{w_e[i] * res['arm_depth']:.1f}",
                "ضریب مقیاس": f"{res['area_factor'][i]:.3f}",
                "max |ρᵢ|": f"{np.max(res['rho'][i]):.4f}",
                "حالت غالب": dominant,
            })
        st.dataframe(rows, use_container_width=True)

        st.markdown("**مقایسه با ساختمان تک‌نما:**")
        st.markdown(
            f"- ρ پایه فولاد: **{base_rho:.2f}** (ساختمان: 0.12)\n"
            f"- area_ref المان نازک: **{area_ref:.0f} m²** (ساختمان: 200 m²)\n"
            f"- ρ_diff لبه فلزی: **{rho_diff:.2f}** (ساختمان: 0.25)")

    with st.expander("📋 خلاصه پارامترها و خروجی"):
        st.json({
            "elements": elements,
            "tower_position_m": D_t,
            "parameters": {
                "f_c_Hz": f_c, "h_ac_m": h_ac,
                "R_max_m": R_max, "R_min_m": R_min,
                "base_rho": base_rho, "area_ref_m2": area_ref,
                "rho_diff": rho_diff, "arm_depth_m": arm_depth,
                "tol_deg": tol, "N": int(N),
            },
            "results": {
                "max_abs_combined_error_deg": res["max_err"],
                "R_at_max_error_m": float(R[res["max_err_idx"]]),
                "fraction_exceeding_tol_percent": 100 * res["frac_exceed"],
            }
        })


# ------------------------------------------------------------
# صفحه ۸: نویز کرونا / عایق دکل فشار قوی
# ------------------------------------------------------------
def page_corona():
    st.title("⚡ اثر نویز کرونا/عایق دکل فشار قوی بر دقت زاویه VOR")
    st.caption("مدل EMI: نویز گاوسی باند پهن + پالس‌های ضربه‌ای — بدون هیچ بازتاب/پراش — "
               "تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("⚡ دکل فشار قوی")
        HV_kV = st.number_input("سطح ولتاژ [kV]", value=400.0, step=10.0, min_value=10.0)
        distance_m = st.number_input("فاصله دکل تا هواپیما [m]", value=800.0, step=50.0, min_value=50.0)
        faultSeverity = st.slider("شدت نقص/کرونا (0=سالم، 1=شدید)", 0.0, 1.0, 0.85, 0.01)
        weatherFactor = st.slider("ضریب آب‌وهوا (1=خشک، >1=مرطوب)", 0.5, 3.0, 1.5, 0.1)

        st.header("📡 سیستم VOR")
        fVOR_MHz = st.number_input("فرکانس VOR [MHz]", value=113.8, step=0.1,
                                   min_value=108.0, max_value=118.0)
        f30 = st.number_input("فرکانس مرجع f30 [Hz]", value=30.0, step=1.0)
        fSC = st.number_input("زیرحامل FM fSC [Hz]", value=9960.0, step=10.0)
        bearing_deg = st.slider("زاویه ارسالی [deg]", 0.0, 359.9, 40.0, 0.1)
        m_ref = st.slider("شاخص مدولاسیون AM مرجع", 0.0, 1.0, 0.30, 0.01)
        m_fm = st.number_input("شاخص مدولاسیون FM", value=16.0, step=1.0)

        st.header("🔧 سطوح نویز")
        sigma_bg = st.number_input("نویز گاوسی گیرنده (front-end)", value=0.0025,
                                   step=0.0005, format="%.4f")
        sigma_corona = st.number_input("نویز کرونای باند پهن", value=0.010,
                                       step=0.001, format="%.4f")

        st.header("🎛️ تنظیمات شبیه‌سازی")
        fs = st.number_input("فرکانس نمونه‌برداری [Hz]", value=200e3, step=10e3, format="%.0f")
        T = st.number_input("مدت شبیه‌سازی [s]", value=4.0, step=0.5, min_value=1.0)
        seed = st.number_input("بذر تصادفی", value=7, step=1, min_value=0)

        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if not run:
        st.info(
            "👈 پارامترهای دکل، سیستم VOR و نویز را در نوار کناری تنظیم کنید "
            "و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته:** این مدل **هیچ** بازتاب، پراش یا مولتی‌پات ندارد — "
            "اثر دکل فقط از طریق **EMI کرونا + تخلیه عایق** (نویز باند پهن + "
            "پالس‌های ضربه‌ای) است.\n\n"
            "**پیشنهاد:** ابتدا با `faultSeverity=0` (سالم) و `weatherFactor=1` "
            "(خشک) اجرا کنید تا خطای پایه را ببینید، سپس شدت را افزایش دهید.")
        return

    progress_bar, update_progress = make_progress()
    with st.spinner("در حال اجرا..."):
        res = run_pylon_corona_simulation(
            fVOR=fVOR_MHz * 1e6, f30=f30, fSC=fSC,
            bearing_deg=bearing_deg,
            m_ref=m_ref, m_fm=m_fm,
            fs=fs, T=T,
            distance_m=distance_m, HV_kV=HV_kV,
            faultSeverity=faultSeverity, weatherFactor=weatherFactor,
            sigma_bg=sigma_bg, sigma_corona=sigma_corona,
            seed=int(seed), progress_cb=update_progress,
        )
    progress_bar.empty()
    st.success("✅ شبیه‌سازی با موفقیت انجام شد.")

    stats = res["stats"]
    c1, c2, c3, c4 = st.columns(4)
    show_card(c1, "میانگین خطا", f'{stats["mean_err"]:.4f}°')
    show_card(c2, "RMS خطا", f'{stats["rms_err"]:.4f}°')
    show_card(c3, "بیشینه |خطا|", f'{stats["max_abs_err"]:.4f}°')
    show_card(c4, "درصد > 1°", f'{100*stats["frac_outside_1deg"]:.1f}%')

    st.info(
        f"**{stats['nPulses']} پالس** در {T:.1f} s | "
        f"نرخ پالس ≈ **{stats['pulseRate']:.0f} Hz** | "
        f"عرض پالس ≈ **{stats['pulseWidth_us']:.1f} µs** | "
        f"ضریب نویز = **{stats['noiseScale']:.3f}** | "
        f"بایاس کالیبراسیون = **{stats['bias_deg']:.3f}°**")
    st.divider()

    st.subheader("📶 شکل ۱ — سیگنال ایده‌آل، نویز کرونا، و سیگنال دریافتی")
    t = res["t"]; N = res["N"]
    idxPlot = min(N, int(round(0.15 * res["fs"])))

    fig1 = make_subplots(
        rows=3, cols=1,
        subplot_titles=("سیگنال ترکیبی VOR ایده‌آل",
                        "نویز کرونا / عایق آسیب‌دیده (باند پهن + ضربه‌ای)",
                        "سیگنال VOR دریافتی در حضور EMI دکل"),
        vertical_spacing=0.10)
    fig1.add_trace(go.Scatter(
        x=t[:idxPlot], y=res["vor_clean"][:idxPlot], mode="lines",
        line=dict(color="#4fc3f7", width=1), name="ایده‌آل"), row=1, col=1)
    fig1.add_trace(go.Scatter(
        x=t[:idxPlot], y=res["coronaNoise"][:idxPlot], mode="lines",
        line=dict(color="#ffb74d", width=1), name="نویز کرونا"), row=2, col=1)
    fig1.add_trace(go.Scatter(
        x=t[:idxPlot], y=res["vor_received"][:idxPlot], mode="lines",
        line=dict(color="#ef5350", width=1), name="دریافتی"), row=3, col=1)
    for r in (1, 2, 3):
        fig1.update_xaxes(title_text="زمان [s]", row=r, col=1)
    fig1.update_yaxes(title_text="دامنه", row=1, col=1)
    fig1.update_yaxes(title_text="نویز", row=2, col=1)
    fig1.update_yaxes(title_text="دامنه", row=3, col=1)
    fig1.update_layout(template="plotly_dark", height=750,
                       showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 شکل ۲ — طیف سیگنال ایده‌آل و دریافتی")
    f_axis_kHz = res["f_axis_kHz"]; P1 = res["P1"]; P2 = res["P2"]
    sel = (f_axis_kHz >= -25) & (f_axis_kHz <= 25)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=f_axis_kHz[sel], y=P1[sel], mode="lines", name="VOR ایده‌آل",
        line=dict(color="#4fc3f7", width=1.1)))
    fig2.add_trace(go.Scatter(
        x=f_axis_kHz[sel], y=P2[sel], mode="lines", name="VOR + EMI دکل",
        line=dict(color="#ef5350", width=1.0)))
    fig2.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="فرکانس baseband معادل [kHz]",
        yaxis_title="دامنه [dB]", yaxis_range=[-100, 10],
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🧭 شکل ۳ — زاویه بازیابی‌شده")
    valid = res["valid"]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=t[valid], y=res["bearing_est"][valid], mode="lines", name="زاویه تخمینی",
        line=dict(color="#4fc3f7", width=1.2)))
    fig3.add_hline(
        y=bearing_deg, line=dict(color="white", dash="dash", width=1.2),
        annotation_text=f"زاویه واقعی {bearing_deg:.2f}°",
        annotation_position="top")
    fig3.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="زمان [s]", yaxis_title="زاویه [deg]",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📉 شکل ۴ — خطای زاویه فقط ناشی از EMI کرونا/عایق")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=t[valid], y=res["bearing_error"][valid], mode="lines", name="خطای زاویه",
        line=dict(color="#ef5350", width=1.1)))
    fig4.add_hline(y=1.0, line=dict(color="white", dash="dash", width=1))
    fig4.add_hline(y=-1.0, line=dict(color="white", dash="dash", width=1))
    fig4.add_hline(y=3.0, line=dict(color="magenta", dash="dot", width=1))
    fig4.add_hline(y=-3.0, line=dict(color="magenta", dash="dot", width=1))
    fig4.add_hline(y=0.0, line=dict(color="white", dash="dot", width=1))
    fig4.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="زمان [s]", yaxis_title="خطای زاویه [deg]",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("📋 آماره‌ها و پارامترها"):
        st.markdown(f"""
**پارامترهای سیستم:**
- فرکانس VOR: **{stats['fVOR_MHz']:.1f} MHz**
- طول موج: **{stats['lambda_m']:.3f} m**
- زاویه واقعی: **{bearing_deg:.2f}°**

**پارامترهای دکل:**
- سطح ولتاژ: **{stats['HV_kV']:.0f} kV**
- فاصله: **{stats['distance_m']:.0f} m**
- شدت نقص: **{stats['faultSeverity']:.2f}**
- ضریب آب‌وهوا: **{stats['weatherFactor']:.2f}**
- نرخ پالس: **{stats['pulseRate']:.0f} pulse/s**
- عرض پالس: **{stats['pulseWidth_us']:.1f} µs**
- ضریب نویز نهایی: **{stats['noiseScale']:.3f}**

**نتایج خطای زاویه:**
- میانگین: **{stats['mean_err']:.4f}°**
- RMS: **{stats['rms_err']:.4f}°**
- بیشینه |خطا|: **{stats['max_abs_err']:.4f}°**
- درصد > 1°: **{100*stats['frac_outside_1deg']:.2f}%**
- درصد > 3°: **{100*stats['frac_outside_3deg']:.2f}%**
- بایاس کالیبراسیون: **{stats['bias_deg']:.3f}°**
        """)
        st.markdown("**خروجی خام JSON:**")
        st.json(stats)


# ------------------------------------------------------------
# صفحه ۹: توربین بادی
# ------------------------------------------------------------
def page_wind_turbine():
    st.title("🌀 اثر توربین بادی و مزرعه بادی بر خطای زاویه VOR")
    st.caption("مدل تحلیلی مهندسی بر اساس Morlaas et al. (2008) — تبدیل‌شده از MATLAB")

    with st.sidebar:
        st.header("📡 سیستم VOR")
        fVOR_MHz = st.number_input("فرکانس VOR [MHz]", value=110.0, step=0.1,
                                   min_value=108.0, max_value=118.0)
        Rrx_km = st.number_input("فاصله هواپیما تا VOR [km]", value=20.0, step=1.0, min_value=1.0)

        st.header("🌀 هندسه مزرعه بادی")
        Rfarm = st.number_input("فاصله مرکز مزرعه از VOR [m]", value=5250.0, step=100.0, min_value=100.0)
        Ntur = st.number_input("تعداد توربین‌ها", value=9, step=1, min_value=1, max_value=50)
        spacing = st.number_input("فاصله بین توربین‌ها [m]", value=240.0, step=10.0, min_value=10.0)

        st.header("🔧 ضرایب پراکندگی")
        K_mast = st.number_input("K_mast — ضریب دکل", value=0.0045, step=0.0005, format="%.4f")
        K_blade = st.number_input("K_blade — ضریب پره", value=0.0008, step=0.0001, format="%.4f")

        st.header("⚙️ پارامترهای هندسی توربین")
        blade_radius = st.number_input("شعاع پره مزرعه [m]", value=15.0, step=1.0, min_value=1.0)
        single_blade_radius = st.number_input("فاصله مرکز پره تک‌توربین [m]", value=30.0, step=1.0, min_value=1.0)
        blade_orientation_offset = st.number_input("آفست چرخش پره بین توربین‌ها [deg]", value=35.0, step=5.0)

        st.header("🎯 تحلیل")
        phi_step = st.number_input("گام آزیموت هواپیما [deg]", value=0.25, step=0.05, min_value=0.05)
        clamp = st.number_input("محدودسازی خطا به ±[deg]", value=3.0, step=0.5, min_value=0.5)

        run = st.button("🚀 اجرای شبیه‌سازی", use_container_width=True, type="primary")

    if not run:
        st.info(
            "👈 پارامترهای VOR، مزرعه بادی و ضرایب پراکندگی را در نوار کناری "
            "تنظیم کنید و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته:** این مدل بر اساس Morlaas et al. (2008) است — "
            "هر توربین = دکل + ۳ پره، همه به‌عنوان پراکنده‌های نقطه‌ای. "
            "خطا به ±3° محدود می‌شود (مطابق مقاله).")
        return

    progress_bar, update_progress = make_progress()
    with st.spinner("در حال اجرا..."):
        res = simulate_wind_turbine(
            fVOR=fVOR_MHz * 1e6, Rrx=Rrx_km * 1000,
            Rfarm=Rfarm, Ntur=int(Ntur), spacing=spacing,
            K_mast=K_mast, K_blade=K_blade,
            phi_step=phi_step, clamp=clamp,
            blade_radius=blade_radius,
            blade_orientation_offset=blade_orientation_offset,
            single_blade_radius=single_blade_radius,
            progress_cb=update_progress,
        )
    progress_bar.empty()
    st.success("✅ شبیه‌سازی با موفقیت انجام شد.")

    stats = res["stats"]
    st.markdown("### 📊 خلاصه نتایج")
    c1, c2, c3, c4 = st.columns(4)
    show_card(c1, "تک‌توربین — دکل", f'{stats["mast_max"]:.3f}°', sub=f'RMS {stats["mast_rms"]:.3f}°')
    show_card(c2, "تک‌توربین — دکل+پره", f'{stats["full_max"]:.3f}°', sub=f'RMS {stats["full_rms"]:.3f}°')
    show_card(c3, "مزرعه — فقط دکل", f'{stats["farm_mast_max"]:.3f}°', sub=f'RMS {stats["farm_mast_rms"]:.3f}°')
    show_card(c4, "مزرعه — کامل", f'{stats["farm_max"]:.3f}°', sub=f'RMS {stats["farm_rms"]:.3f}°')

    st.info(
        f"λ = **{stats['lambda_m']:.3f} m** | "
        f"فاصله مزرعه = **{stats['Rfarm_m']:.0f} m** | "
        f"{stats['Ntur']} توربین، فاصله بین‌شان = **{stats['spacing_m']:.0f} m** | "
        f"K_mast = **{stats['K_mast']:.4f}**، K_blade = **{stats['K_blade']:.4f}**")
    st.divider()

    phi0 = res["phi0"]

    st.subheader("📈 شکل ۱ — خطای زاویه در برابر آزیموت هواپیما")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=phi0, y=res["errorIdeal"], mode="lines", name="VOR ایده‌آل",
                              line=dict(color="white", width=1.5)))
    fig1.add_trace(go.Scatter(x=phi0, y=res["errorMast"], mode="lines", name="تک‌توربین — دکل",
                              line=dict(color="#4fc3f7", width=1.4)))
    fig1.add_trace(go.Scatter(x=phi0, y=res["errorFull"], mode="lines", name="تک‌توربین — دکل + پره",
                              line=dict(color="#66bb6a", width=1.4)))
    fig1.add_trace(go.Scatter(x=phi0, y=res["errorFarm"], mode="lines",
                              name=f"مزرعه {stats['Ntur']} توربینی",
                              line=dict(color="#ffb74d", width=1.5)))
    fig1.add_hline(y=clamp, line=dict(color="magenta", dash="dash", width=1),
                   annotation_text=f"+{clamp:.1f}°", annotation_position="top")
    fig1.add_hline(y=-clamp, line=dict(color="magenta", dash="dash", width=1),
                   annotation_text=f"-{clamp:.1f}°", annotation_position="bottom")
    fig1.add_hline(y=0, line=dict(color="white", dash="dot", width=1))
    fig1.update_layout(
        template="plotly_dark", height=500,
        xaxis_title="آزیموت هواپیما [deg]", yaxis_title="خطای زاویه VOR [deg]",
        xaxis_range=[0, 360],
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 شکل ۲ — قدر مطلق خطای زاویه")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=phi0, y=np.abs(res["errorMast"]), mode="lines", name="فقط دکل",
                              line=dict(color="#4fc3f7", width=1.4)))
    fig2.add_trace(go.Scatter(x=phi0, y=np.abs(res["errorFull"]), mode="lines", name="دکل + پره",
                              line=dict(color="#66bb6a", width=1.4)))
    fig2.add_trace(go.Scatter(x=phi0, y=np.abs(res["errorFarm"]), mode="lines", name="مزرعه",
                              line=dict(color="#ffb74d", width=1.5)))
    fig2.add_hline(y=clamp, line=dict(color="magenta", dash="dash", width=1),
                   annotation_text=f"{clamp:.1f}°", annotation_position="top")
    fig2.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="آزیموت هواپیما [deg]", yaxis_title="|خطای زاویه| [deg]",
        xaxis_range=[0, 360],
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🧭 شکل ۳ — نمودار قطبی خطای مزرعه (شعاع = |خطا|)")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=np.abs(res["errorFarm"]), theta=phi0, mode="lines", name="|خطای مزرعه|",
        line=dict(color="#ffb74d", width=1.4)))
    fig3.add_trace(go.Scatterpolar(
        r=np.full_like(phi0, clamp), theta=phi0, mode="lines",
        name=f"حد ±{clamp:.1f}°",
        line=dict(color="magenta", dash="dash", width=1)))
    fig3.update_layout(
        template="plotly_dark", height=550,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, clamp], tickfont=dict(color="white")),
            angularaxis=dict(
                tickmode="array",
                tickvals=list(range(0, 360, 30)),
                ticktext=[f"{d}°" for d in range(0, 360, 30)],
                direction="counterclockwise")),
        showlegend=True,
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🗺️ شکل ۴ — هندسه مزرعه بادی نسبت به VOR")
    farmX = res["farmX"]; farmY = res["farmY"]; Rrx_m = res["Rrx"]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(symbol="triangle-up", size=16, color="black"),
        text=["VOR"], textposition="bottom center", name="VOR"))
    fig4.add_trace(go.Scatter(
        x=farmX, y=farmY, mode="markers+text",
        marker=dict(symbol="star", size=14, color="#ffb74d"),
        text=[f"T{i+1}" for i in range(len(farmX))],
        textposition="top center", name="توربین‌ها"))
    th = np.linspace(0, 2 * np.pi, 200)
    fig4.add_trace(go.Scatter(
        x=Rrx_m * np.cos(th), y=Rrx_m * np.sin(th),
        mode="lines", line=dict(color="#4fc3f7", dash="dot", width=1),
        name=f"مسیر هواپیما (r={Rrx_m/1000:.0f} km)"))
    for i in range(len(farmX)):
        fig4.add_trace(go.Scatter(
            x=[0, farmX[i]], y=[0, farmY[i]],
            mode="lines", line=dict(color="rgba(255,183,77,0.3)", width=1),
            showlegend=False, hoverinfo="skip"))
    fig4.update_layout(
        template="plotly_dark", height=550,
        xaxis_title="شرق [m]", yaxis_title="شمال [m]",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("📋 جدول آماره‌های کامل"):
        rows = [
            {"سناریو": "تک‌توربین — فقط دکل",
             "بیشینه |خطا| [°]": f"{stats['mast_max']:.3f}", "RMS [°]": f"{stats['mast_rms']:.3f}"},
            {"سناریو": "تک‌توربین — دکل + ۳ پره",
             "بیشینه |خطا| [°]": f"{stats['full_max']:.3f}", "RMS [°]": f"{stats['full_rms']:.3f}"},
            {"سناریو": "مزرعه — فقط دکل‌ها",
             "بیشینه |خطا| [°]": f"{stats['farm_mast_max']:.3f}", "RMS [°]": f"{stats['farm_mast_rms']:.3f}"},
            {"سناریو": "مزرعه — توربین کامل",
             "بیشینه |خطا| [°]": f"{stats['farm_max']:.3f}", "RMS [°]": f"{stats['farm_rms']:.3f}"},
        ]
        st.dataframe(rows, use_container_width=True)
        st.markdown("**پارامترها:**")
        st.json(stats)


# ############################################################
#  بخش ۴: برنامه اصلی — منوی انتخاب شبیه‌ساز
# ############################################################
SIMULATORS = {
    "📡 ۱) CVOR ایده‌آل (استخراج زاویه)": page_cvor_ideal,
    "🏔️ ۲) اثر توپوگرافی کوه": page_mountain,
    "🌲 ۳) جنگل (CVOR)": page_forest,
    "🌳 ۴) یک درخت (CVOR)": page_single_tree,
    "🏢 ۵) یک ساختمان": page_single_building,
    "🏙️ ۶) چند ساختمان": page_multi_building,
    "🗼 ۷) دکل انتقال نیرو": page_power_tower,
    "⚡ ۸) نویز کرونا/عایق دکل فشار قوی": page_corona,
    "🌀 ۹) توربین بادی / مزرعه بادی": page_wind_turbine,
}


def main():
    st.set_page_config(page_title="VOR / CVOR Simulators",
                       page_icon="📡", layout="wide")

    st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #4fc3f7; text-align: center; }
    .metric-card { background:#1c2333; padding:18px; border-radius:12px;
                   border:1px solid #2e3b55; text-align:center; }
    .metric-value { font-size: 26px; font-weight: bold; color:#4fc3f7; }
    .metric-label { color:#9aa7c7; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 📡 مجموعه شبیه‌سازهای VOR")
        choice = st.selectbox("🎛️ شبیه‌ساز مورد نظر را انتخاب کنید",
                              list(SIMULATORS.keys()))
        st.divider()

    SIMULATORS[choice]()


main()
