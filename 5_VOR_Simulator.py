"""
CVOR Simulators — برنامه یکپارچه (Streamlit)
============================================
پنج شبیه‌ساز CVOR در یک برنامه؛ کاربر ابتدا شبیه‌ساز را از نوار کناری انتخاب می‌کند:

  ۱) ایده‌آل            — کانال بدون نویز/مولتی‌پات (استخراج زاویه از VAR/REF)
  ۲) توپوگرافی زمین     — فاست‌های زمین (constant-gamma) + Eq. (3-14)
  ۳) جنگل               — قطعات جنگلی (دایره/مستطیل/چندضلعی) + Eq. (3-14)
  ۴) ساختمان‌ها         — فاست‌بندی دیوارها (فیزیک-نوری) + Eq. (3-14)
  ۵) مزرعه بادی         — دکل + هاب + پره‌ها + Eq. (3-14)

اجرا:
    pip install streamlit numpy pandas plotly scipy
    streamlit run cvor_simulators.py
"""

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import RegularGridInterpolator, griddata


# ============================================================
# بخش مشترک
# ============================================================
def p95(x):
    xa = np.sort(np.abs(x))
    return float(xa[max(int(np.ceil(0.95 * len(xa))) - 1, 0)])


CSS = """
<style>
.main { background-color: #0e1117; }
h1 { color: #4fc3f7; text-align: center; }
.metric-card { background:#1c2333; padding:18px; border-radius:12px;
               border:1px solid #2e3b55; text-align:center; }
.metric-value { font-size: 24px; font-weight: bold; color:#4fc3f7; }
.metric-label { color:#9aa7c7; font-size: 13px; }
.sim-card { background:#1c2333; padding:16px; border-radius:12px;
            border:1px solid #2e3b55; margin-bottom:8px; min-height:110px; }
.sim-card b { color:#4fc3f7; }
</style>
"""


# ##########################################################
# ۱) شبیه‌ساز ایده‌آل
# ##########################################################
# ============================================================
# هسته محاسباتی — معادل دقیق کد MATLAB
# ============================================================
def simulate_cvor_ideal(true_bearing_deg=250.0,
                        fs=200e3, T=1.0,
                        f_ref=30.0, f_sub=9960.0,
                        dev=480.0,
                        m_ref=0.3, m_var=0.3,
                        progress_cb=None):
    """شبیه‌سازی ایده‌آل CVOR و تخمین زاویه."""
    c = 3e8
    fc = 113e6
    t = np.arange(0, T, 1/fs)
    N = len(t)

    beta = dev / f_ref
    theta = np.deg2rad(true_bearing_deg)

    if progress_cb:
        progress_cb(0.05)

    # --- 1. تولید اجزای CVOR ---
    var_sig = m_var * np.cos(2 * np.pi * f_ref * t - theta)
    ref_fm = m_ref * np.cos(2 * np.pi * f_sub * t
                            + beta * np.sin(2 * np.pi * f_ref * t))
    m_t = var_sig + ref_fm
    rx = m_t.copy()

    if progress_cb:
        progress_cb(0.20)

    # --- 2. استخراج مؤلفه ۳۰Hz متغیر (VAR) از AM ---
    bv, av = butter(4, 100 / (fs / 2), btype='low')
    var_rx = filtfilt(bv, av, rx)

    I_var = 2 * np.mean(var_rx * np.cos(2 * np.pi * f_ref * t))
    Q_var = -2 * np.mean(var_rx * np.sin(2 * np.pi * f_ref * t))
    phase_var = np.arctan2(Q_var, I_var)

    if progress_cb:
        progress_cb(0.40)

    # --- 3. استخراج مؤلفه مرجع از زیرحامل FM ---
    br, ar = butter(4, [(f_sub - 1500) / (fs / 2),
                        (f_sub + 1500) / (fs / 2)], btype='bandpass')
    sub_rx = filtfilt(br, ar, rx)

    analytic_sub = hilbert(sub_rx)
    inst_phase = np.unwrap(np.angle(analytic_sub))
    inst_freq = np.concatenate(([0.0], np.diff(inst_phase))) * fs / (2 * np.pi)
    ref_demod = inst_freq - f_sub

    valid = (t > 0.05) & (t < (T - 0.05))
    tv = t[valid]
    ref_valid = ref_demod[valid]

    I_ref = 2 * np.mean(ref_valid * np.cos(2 * np.pi * f_ref * tv))
    Q_ref = -2 * np.mean(ref_valid * np.sin(2 * np.pi * f_ref * tv))
    phase_ref = np.arctan2(Q_ref, I_ref)

    if progress_cb:
        progress_cb(0.55)

    # --- 4. تخمین زاویه ---
    measured_bearing_deg = np.mod(np.rad2deg(phase_ref - phase_var), 360)
    bearing_error_deg = np.mod(measured_bearing_deg - true_bearing_deg
                               + 180, 360) - 180

    # --- 5. طیف سیگنال ---
    M = np.abs(np.fft.fft(m_t)) / N
    f_axis = np.arange(N) * (fs / N)
    half = N // 2

    if progress_cb:
        progress_cb(0.65)

    # --- 6. خطای زاویه در کل بازه 0 تا 350 ---
    test_bearings = np.arange(0, 360, 10)
    errors = np.zeros_like(test_bearings, dtype=float)
    total = len(test_bearings)

    for k, tb in enumerate(test_bearings):
        th_k = np.deg2rad(tb)
        var_k = m_var * np.cos(2 * np.pi * f_ref * t - th_k)
        ref_k = m_ref * np.cos(2 * np.pi * f_sub * t
                               + beta * np.sin(2 * np.pi * f_ref * t))
        rx_k = var_k + ref_k

        var_k_rx = filtfilt(bv, av, rx_k)
        I_v = 2 * np.mean(var_k_rx * np.cos(2 * np.pi * f_ref * t))
        Q_v = -2 * np.mean(var_k_rx * np.sin(2 * np.pi * f_ref * t))
        ph_v = np.arctan2(Q_v, I_v)

        sub_k = filtfilt(br, ar, rx_k)
        ph_k = np.unwrap(np.angle(hilbert(sub_k)))
        fi_k = np.concatenate(([0.0], np.diff(ph_k))) * fs / (2 * np.pi)
        rd_k = fi_k - f_sub
        rd_valid = rd_k[valid]

        I_r = 2 * np.mean(rd_valid * np.cos(2 * np.pi * f_ref * tv))
        Q_r = -2 * np.mean(rd_valid * np.sin(2 * np.pi * f_ref * tv))
        ph_r = np.arctan2(Q_r, I_r)

        bearing_k = np.mod(np.rad2deg(ph_r - ph_v), 360)
        errors[k] = np.mod(bearing_k - tb + 180, 360) - 180

        if progress_cb:
            progress_cb(0.65 + 0.35 * (k + 1) / total)

    return {
        "t": t, "m_t": m_t, "var_rx": var_rx, "sub_rx": sub_rx,
        "tv": tv, "ref_valid": ref_valid, "valid": valid,
        "f_axis": f_axis[:half], "spectrum": 2 * M[:half],
        "fs": fs, "N": N,
        "measured_bearing_deg": float(measured_bearing_deg),
        "bearing_error_deg": float(bearing_error_deg),
        "test_bearings": test_bearings, "errors": errors,
        "params": dict(true_bearing_deg=true_bearing_deg, fs=fs, T=T,
                       f_ref=f_ref, f_sub=f_sub, dev=dev,
                       m_ref=m_ref, m_var=m_var,
                       fc=fc, c=c, beta=beta,
                       lambda_=c / fc),
        "max_abs_err": float(np.max(np.abs(errors))),
        "rms_err": float(np.sqrt(np.mean(errors ** 2))),
    }


def page_ideal():
    st.title("📡 شبیه‌ساز ایده‌آل CVOR")
    st.caption(
        "استخراج زاویه از سیگنال مرجع FM (۹۹۶۰ Hz) و متغیر AM (۳۰ Hz) — "
        "کانال ایده‌آل بدون نویز/مولتی‌پات — تبدیل‌شده از MATLAB"
    )

    # --- نوار کناری: ورودی‌ها ---
    with st.sidebar:
        st.header("⚙️ پارامترها")
        true_bearing = st.slider("زاویه واقعی (درجه)", 0.0, 359.9, 250.0, 0.1)

        st.subheader("سیستم")
        fs = st.number_input("فرکانس نمونه‌برداری fs (Hz)",
                             value=200e3, step=1e3, format="%.0f")
        T = st.number_input("مدت شبیه‌سازی T (s)",
                            value=1.0, step=0.1, min_value=0.2)

        st.subheader("مدولاسیون")
        f_ref = st.number_input("فرکانس مرجع f_ref (Hz)", value=30.0, step=1.0)
        f_sub = st.number_input("زیرحامل FM f_sub (Hz)", value=9960.0, step=10.0)
        dev = st.number_input("انحراف FM (Hz)", value=480.0, step=10.0)
        m_ref = st.slider("دامنه مرجع", 0.0, 1.0, 0.3, 0.01)
        m_var = st.slider("دامنه متغیر", 0.0, 1.0, 0.3, 0.01)

        st.markdown("---")
        st.caption("ℹ️ کانال ایده‌آل: هیچ نویز، مولتی‌پات، "
                   "تضعیف یا تأخیری وجود ندارد.")
        run = st.button("🚀 اجرای شبیه‌سازی",
                        use_container_width=True, type="primary")


    if run:
        progress_bar = st.progress(0, text="در حال شبیه‌سازی...")

        def update_progress(frac):
            progress_bar.progress(min(max(frac, 0.0), 1.0),
                                  text=f"در حال شبیه‌سازی... {frac*100:.0f}%")

        t0 = time.time()
        with st.spinner("در حال اجرا..."):
            res = simulate_cvor_ideal(
                true_bearing_deg=true_bearing,
                fs=fs, T=T,
                f_ref=f_ref, f_sub=f_sub, dev=dev,
                m_ref=m_ref, m_var=m_var,
                progress_cb=update_progress,
            )
        progress_bar.empty()
        elapsed = time.time() - t0

        st.success(f"✅ شبیه‌سازی در {elapsed:.2f} ثانیه انجام شد.")

        # --- کارت‌های نتیجه ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">زاویه واقعی</div>'
                f'<div class="metric-value">{true_bearing:.2f}°</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">زاویه اندازه‌گیری‌شده</div>'
                f'<div class="metric-value">{res["measured_bearing_deg"]:.2f}°</div></div>',
                unsafe_allow_html=True)
        with c3:
            err = res["bearing_error_deg"]
            color = "#4caf50" if abs(err) < 1 else "#ff9800" if abs(err) < 5 else "#f44336"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">خطای زاویه</div>'
                f'<div class="metric-value" style="color:{color}">{err:.4f}°</div></div>',
                unsafe_allow_html=True)

        # --- خطوط وضعیت ---
        st.info(
            f"λ = **{res['params']['lambda_']:.3f} m** | "
            f"β = dev/f_ref = **{res['params']['beta']:.2f}** | "
            f"بیشینه |خطا| در بازه ۰–۳۵۰° = **{res['max_abs_err']:.4f}°** | "
            f"RMS = **{res['rms_err']:.4f}°**"
        )

        st.divider()

        # ============================================================
        # شکل ۱: سیگنال پایه و طیف
        # ============================================================
        st.subheader("📶 سیگنال پایه CVOR و طیف آن")
        nshow = int(round(0.05 * fs))
        t = res["t"]; m_t = res["m_t"]
        f_axis = res["f_axis"]; spec = res["spectrum"]

        fig1 = make_subplots(
            rows=2, cols=1,
            subplot_titles=("سیگنال پایه ترکیبی CVOR (۵۰ ms اول)",
                            "طیف سیگنال CVOR"),
            vertical_spacing=0.12)
        fig1.add_trace(go.Scatter(
            x=t[:nshow], y=m_t[:nshow], mode="lines",
            name="سیگنال CVOR",
            line=dict(color="#4fc3f7", width=1.2)), row=1, col=1)
        fig1.add_trace(go.Scatter(
            x=f_axis, y=spec, mode="lines",
            name="طیف",
            line=dict(color="#ba68c8", width=1.2)), row=2, col=1)

        fig1.update_xaxes(title_text="زمان [s]", row=1, col=1)
        fig1.update_xaxes(title_text="فرکانس [Hz]", range=[0, 12000], row=2, col=1)
        fig1.update_yaxes(title_text="دامنه", row=1, col=1)
        fig1.update_yaxes(title_text="دامنه", row=2, col=1)
        fig1.update_layout(
            height=650, template="plotly_dark",
            showlegend=False, margin=dict(l=50, r=20, t=60, b=40))
        st.plotly_chart(fig1, use_container_width=True)

        # ============================================================
        # شکل ۲: بازیابی مؤلفه‌های ۳۰Hz
        # ============================================================
        st.subheader("🔄 بازیابی مؤلفه‌های ۳۰Hz — REF از FM، VAR از AM")
        tv = res["tv"]; ref_v = res["ref_valid"]
        var_rx = res["var_rx"]; valid = res["valid"]

        ref_norm = ref_v / np.max(np.abs(ref_v))
        var_norm = var_rx[valid] / np.max(np.abs(var_rx[valid]))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=tv, y=ref_norm, mode="lines", name="REF (FM)",
            line=dict(color="#4fc3f7", width=1.4)))
        fig2.add_trace(go.Scatter(
            x=tv, y=var_norm, mode="lines", name="VAR (AM)",
            line=dict(color="#ffb74d", width=1.4)))
        fig2.update_layout(
            template="plotly_dark", height=420,
            xaxis_title="زمان [s]",
            yaxis_title="دامنه نرمال‌شده",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=50, r=20, t=40, b=40))
        st.plotly_chart(fig2, use_container_width=True)

        # ============================================================
        # شکل ۳: خطای زاویه در بازه ۰ تا ۳۵۰
        # ============================================================
        st.subheader("📊 خطای زاویه در کل بازه ۰ تا ۳۵۰ درجه")
        tb = res["test_bearings"]; errs = res["errors"]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=tb, y=errs,
            marker_color="#66bb6a",
            name="خطای زاویه (درجه)"))
        fig3.add_hline(y=1.0, line=dict(color="white", dash="dash", width=1),
                       annotation_text="±1°", annotation_position="top left")
        fig3.add_hline(y=-1.0, line=dict(color="white", dash="dash", width=1))
        fig3.add_hline(y=0, line=dict(color="white", dash="dot", width=1))
        fig3.update_layout(
            template="plotly_dark", height=420,
            xaxis_title="زاویه واقعی [درجه]",
            yaxis_title="خطای زاویه [درجه]",
            margin=dict(l=50, r=20, t=40, b=40))
        st.plotly_chart(fig3, use_container_width=True)

        # ============================================================
        # جدول خلاصه
        # ============================================================
        with st.expander("📋 خلاصه پارامترها و خروجی"):
            st.markdown("**پارامترهای سیستم:**")
            st.markdown(f"- فرکانس حامل: **{res['params']['fc']/1e6:.1f} MHz**")
            st.markdown(f"- طول موج: **{res['params']['lambda_']:.3f} m**")
            st.markdown(f"- fs = **{res['params']['fs']:.0f} Hz** | "
                        f"T = **{res['params']['T']:.2f} s**")
            st.markdown(f"- f_ref = **{res['params']['f_ref']:.1f} Hz** | "
                        f"f_sub = **{res['params']['f_sub']:.1f} Hz**")
            st.markdown(f"- انحراف FM = **{res['params']['dev']:.1f} Hz** | "
                        f"β = **{res['params']['beta']:.2f}**")
            st.markdown(f"- m_ref = **{res['params']['m_ref']:.2f}** | "
                        f"m_var = **{res['params']['m_var']:.2f}**")
            st.markdown("")
            st.markdown("**نتایج:**")
            st.markdown(f"- زاویه واقعی: **{true_bearing:.2f}°**")
            st.markdown(f"- زاویه اندازه‌گیری‌شده: **{res['measured_bearing_deg']:.4f}°**")
            st.markdown(f"- خطای زاویه: **{res['bearing_error_deg']:.6f}°**")
            st.markdown(f"- بیشینه |خطا| در ۰–۳۵۰°: **{res['max_abs_err']:.6f}°**")
            st.markdown(f"- RMS در ۰–۳۵۰°: **{res['rms_err']:.6f}°**")

            st.markdown("**خروجی خام JSON:**")
            st.json({
                **res["params"],
                "measured_bearing_deg": res["measured_bearing_deg"],
                "bearing_error_deg": res["bearing_error_deg"],
                "max_abs_err": res["max_abs_err"],
                "rms_err": res["rms_err"],
            })

    else:
        st.info(
            "👈 پارامترها را در نوار کناری تنظیم کنید و دکمه «اجرای شبیه‌سازی» "
            "را بزنید.\n\n"
            "**نکته:** این شبیه‌ساز کانال ایده‌آل را مدل می‌کند — "
            "خطای زاویه باید بسیار کوچک (~0.01°) باشد و صرفاً از "
            "خطای عددی فیلترها و دمدولاسیون ناشی می‌شود."
        )


# ##########################################################
# ۲) توپوگرافی زمین
# ##########################################################
# ============================================================
# توابع کمکی
# ============================================================
def build_terrain_xy(vor_xy, L, cell, h_base, features,
                     dem_X=None, dem_Y=None, dem_Z=None,
                     dem_points=None):
    """ساخت شبکه زمین (مختصات نسبت به VOR)."""
    xv = np.arange(vor_xy[0] - L, vor_xy[0] + L + cell / 2, cell)
    yv = np.arange(vor_xy[1] - L, vor_xy[1] + L + cell / 2, cell)
    X, Y = np.meshgrid(xv, yv)
    Z = np.full_like(X, h_base, dtype=float)

    # --- اعمال ویژگی‌های مصنوعی ---
    for ft in features:
        if ft["type"] == "hill":
            Z += ft["H"] * np.exp(
                -((X - ft["x"]) ** 2 + (Y - ft["y"]) ** 2) / (2 * ft["s"] ** 2)
            )
        elif ft["type"] == "ridge":
            p1 = np.array([ft["x1"], ft["y1"]])
            p2 = np.array([ft["x2"], ft["y2"]])
            v = p2 - p1
            L2 = max(v @ v, 1e-12)
            t = ((X - p1[0]) * v[0] + (Y - p1[1]) * v[1]) / L2
            t = np.clip(t, 0, 1)
            dx = X - (p1[0] + t * v[0])
            dy = Y - (p1[1] + t * v[1])
            Z += ft["H"] * np.exp(-(dx ** 2 + dy ** 2) / (2 * ft["w"] ** 2))

    # --- اعمال DEM (در صورت وجود) ---
    if dem_X is not None and dem_Y is not None and dem_Z is not None:
        if dem_X.ndim == 1 and dem_Y.ndim == 1:
            interp = RegularGridInterpolator(
                (dem_Y, dem_X), dem_Z,
                method="linear", bounds_error=False, fill_value=h_base)
            pts = np.column_stack([Y.ravel(), X.ravel()])
            Z_dem = interp(pts).reshape(X.shape)
            Z = Z_dem
    elif dem_points is not None and len(dem_points) > 0:
        Z_dem = griddata(dem_points[:, :2], dem_points[:, 2],
                         (X, Y), method="linear")
        Z_dem = np.where(np.isnan(Z_dem), h_base, Z_dem)
        Z = Z_dem

    Xr = X - vor_xy[0]
    Yr = Y - vor_xy[1]
    D = cell
    z_vor = float(np.interp(0.0,
                            Xr[0, :], Z[0, :])) if False else None
    # interpolate (0,0) properly
    try:
        from scipy.interpolate import RegularGridInterpolator as RGI
        interp_z = RGI((Yr[:, 0], Xr[0, :]), Z, method="linear",
                       bounds_error=False, fill_value=h_base)
        z_vor = float(interp_z([0.0, 0.0])[0])
    except Exception:
        z_vor = float(h_base)

    return dict(X=Xr, Y=Yr, Z=Z, D=D, z_vor=z_vor,
                X_abs=X, Y_abs=Y, vor_xy=vor_xy, h_base=h_base,
                max_z=float(np.max(Z)))


def los_clear(G, P0, P1, s):
    """آیا قطعه P0→P1 در تمام کسرهای s بالای زمین است؟"""
    P0 = np.atleast_2d(P0)
    P1 = np.atleast_2d(P1)
    if P0.shape[0] == 1:
        P0 = np.repeat(P0, P1.shape[0], axis=0)
    N = P1.shape[0]
    x = P0[:, 0:1] + (P1[:, 0:1] - P0[:, 0:1]) * s[None, :]
    y = P0[:, 1:2] + (P1[:, 1:2] - P0[:, 1:2]) * s[None, :]
    z = P0[:, 2:3] + (P1[:, 2:3] - P0[:, 2:3]) * s[None, :]

    # interp2 روی شبکه G.X, G.Y
    xg = G["X"][0, :]
    yg = G["Y"][:, 0]
    try:
        from scipy.interpolate import RegularGridInterpolator as RGI
        interp = RGI((yg, xg), G["Z"], method="linear",
                     bounds_error=False, fill_value=G["h_base"])
        pts = np.column_stack([y.ravel(), x.ravel()])
        zt = interp(pts).reshape(z.shape)
    except Exception:
        zt = np.full_like(z, G["h_base"])

    vis = np.all(z > zt + 0.5, axis=1)
    return vis


def terrain_build_facets(cfg, G):
    """ساخت فاست‌های قابل مشاهده از VOR."""
    rng = np.random.default_rng(int(cfg["seed"]))
    D = G["D"]
    Z = G["Z"]
    # gradient
    gz_dy, gz_dx = np.gradient(Z, D)
    X_abs = G["X_abs"]; Y_abs = G["Y_abs"]

    x = X_abs.ravel() + (rng.random(X_abs.size) - 0.5) * D
    y = Y_abs.ravel() + (rng.random(Y_abs.size) - 0.5) * D
    x = np.clip(x, X_abs.min(), X_abs.max())
    y = np.clip(y, Y_abs.min(), Y_abs.max())

    try:
        from scipy.interpolate import RegularGridInterpolator as RGI
        xg = G["X"][0, :]; yg = G["Y"][:, 0]
        ip_z = RGI((yg, xg), Z, method="linear",
                   bounds_error=False, fill_value=G["h_base"])
        ip_gx = RGI((yg, xg), gz_dx, method="linear",
                    bounds_error=False, fill_value=0.0)
        ip_gy = RGI((yg, xg), gz_dy, method="linear",
                    bounds_error=False, fill_value=0.0)
        z = ip_z(np.column_stack([y, x]))
        gx = ip_gx(np.column_stack([y, x]))
        gy = ip_gy(np.column_stack([y, x]))
    except Exception:
        z = np.full_like(x, G["h_base"])
        gx = np.zeros_like(x); gy = np.zeros_like(x)

    nz = 1.0 / np.sqrt(1 + gx ** 2 + gy ** 2)
    n = np.column_stack([-gx * nz, -gy * nz, nz])
    slope = np.rad2deg(np.arccos(np.clip(nz, -1, 1)))
    area = D ** 2 / nz

    # مختصات نسبت به VOR
    xr = x - cfg["vor_xy"][0]
    yr = y - cfg["vor_xy"][1]

    # فیلترها
    keep = (np.hypot(xr, yr) >= cfg["r_min"]) & (slope >= cfg["min_slope"])
    xr = xr[keep]; yr = yr[keep]; z = z[keep]
    n = n[keep]; area = area[keep]

    S = np.array([0.0, 0.0, G["z_vor"] + cfg["h_ant"]])
    P = np.column_stack([xr, yr, z])
    v = S - P
    d1 = np.linalg.norm(v, axis=1)
    sin_i = np.sum(n * (v / d1[:, None]), axis=1)
    keep2 = sin_i > 0.02
    P = P[keep2]; n = n[keep2]; area = area[keep2]
    d1 = d1[keep2]; sin_i = sin_i[keep2]

    # LOS از VOR
    s_vec = np.linspace(0.02, 0.97, 40)
    vis = los_clear(G, S[None, :], P, s_vec)

    P = P[vis]; n = n[vis]; area = area[vis]
    d1 = d1[vis]; sin_i = sin_i[vis]

    return dict(P=P, n=n, area=area, d1=d1, sin_i=sin_i, S=S, N=len(P))


def terrain_scatterers(cfg, G, F, Ax, Ay):
    """پراکنده‌های فعال برای یک موقعیت هواپیما."""
    c = 299792458.0
    lam = c / (cfg["f0_MHz"] * 1e6)
    phi_ref = np.deg2rad(cfg["phi_ref_deg"])
    gamma = 10 ** (cfg["gamma_dB"] / 10)

    A = np.array([Ax, Ay, cfg["h_air"]])
    if F["N"] == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(0, dtype=int))

    P = F["P"]
    n = F["n"]
    v = A - P
    d2 = np.linalg.norm(v, axis=1)
    sin_s = np.sum(n * (v / d2[:, None]), axis=1)

    idx = np.where(sin_s > 0.02)[0]
    if len(idx) == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(0, dtype=int))

    s_vec = np.logspace(np.log10(0.003), 0, 32)
    vis = los_clear(G, P[idx], np.repeat(A[None, :], len(idx), axis=0), s_vec)
    idx = idx[vis]

    if len(idx) == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(0, dtype=int))

    d2v = d2[idx]
    sin_sv = sin_s[idx]
    sigma0 = gamma * np.sqrt(F["sin_i"][idx] * sin_sv)
    sigma = sigma0 * F["area"][idx]
    dd = np.linalg.norm(A - F["S"])
    a = np.sqrt(sigma / (4 * np.pi)) * dd / (F["d1"][idx] * d2v)
    theta = -2 * np.pi * ((F["d1"][idx] + d2v) - dd) / lam + phi_ref
    azt = np.arctan2(P[idx, 0], P[idx, 1])
    return a, theta, azt, idx


def terrain_compute_eq314(a, theta, azt, pa):
    """معادله (3-14)."""
    if len(a) == 0:
        return 0.0
    phin = azt - pa
    num = -np.sum(a * np.cos(theta) * np.sin(phin))
    den = 1.0 + np.sum(a * np.cos(theta) * np.cos(phin))
    return float(np.rad2deg(np.arctan2(num, den)))


def td_envelope(a, theta, azt, pa, ma=0.30, w=2*np.pi*30,
                fs=3000, T=1.0):
    """اعتبارسنجی زمان‌دامنه با envelope detector (Eq. 2-4)."""
    t = np.arange(0, T, 1 / fs)
    e30 = np.exp(-1j * w * t)
    ejwt = np.exp(1j * w * t)

    if len(a) == 0:
        s = 1 + ma * np.cos(w * t - pa)
    else:
        g = a * np.exp(1j * theta)
        M0 = np.sum(g)
        Cc = np.sum(g * np.exp(-1j * azt))
        Cp = np.sum(g * np.exp(1j * azt))
        s = (1 + ma * np.cos(w * t - pa)
             + M0 + 0.5 * ma * (Cc * ejwt + Cp * np.conj(ejwt)))

    X = np.sum(np.abs(s) * e30)
    return float(np.rad2deg(np.angle(np.exp(1j * (pa - (-np.angle(X)))))))


def terrain_run_simulation(cfg, G, F, progress_cb=None):
    """اجرای اسکن اصلی + zoom."""
    az = np.arange(0, 360, cfg["az_step"])
    pa = np.deg2rad(az)
    N = len(az)
    eps_314 = np.zeros(N)
    eps_td = np.full(N, np.nan)
    nsc = np.zeros(N, dtype=int)

    for k in range(N):
        a, th, azt, _ = terrain_scatterers(
            cfg, G, F, cfg["R_air"] * np.sin(pa[k]),
            cfg["R_air"] * np.cos(pa[k]))
        nsc[k] = len(a)
        eps_314[k] = terrain_compute_eq314(a, th, azt, pa[k])
        if cfg["td_check"]:
            eps_td[k] = td_envelope(a, th, azt, pa[k])
        if progress_cb and (k % max(1, N // 20) == 0 or k == N - 1):
            progress_cb(k / N)

    # --- zoom ---
    ip = int(np.argmax(np.abs(eps_314)))
    az_z = (az[ip] - 0.25) + np.arange(0, 0.5 + 0.001, 0.002)
    paz = np.deg2rad(az_z)
    eps_z = np.zeros(len(az_z))
    for k in range(len(az_z)):
        a, th, azt, _ = terrain_scatterers(
            cfg, G, F, cfg["R_air"] * np.sin(paz[k]),
            cfg["R_air"] * np.cos(paz[k]))
        eps_z[k] = terrain_compute_eq314(a, th, azt, paz[k])

    # --- فاست‌های موثر در بدترین آزیموت ---
    a, th, _, idx = terrain_scatterers(
        cfg, G, F, cfg["R_air"] * np.sin(pa[ip]),
        cfg["R_air"] * np.cos(pa[ip]))
    worst = dict(az=float(az[ip]),
                 x=F["P"][idx, 0], y=F["P"][idx, 1],
                 a=a, act=a * np.cos(th))

    return dict(az=az, eps_314=eps_314, eps_td=eps_td, nsc=nsc,
                az_z=az_z, eps_z=eps_z, worst=worst)


def page_terrain():
    st.title("⛰️ خطای CVOR ناشی از توپوگرافی زمین")
    st.caption(
        "مدل فاست‌های نقطه‌ای (constant-gamma) + Eq. (3-14) با REF ایده‌آل — "
        "تبدیل‌شده از MATLAB"
    )

    # --- نوار کناری ---
    with st.sidebar:
        st.header("📡 VOR")
        vor_x = st.number_input("VOR x East [m]", value=0.0, step=100.0)
        vor_y = st.number_input("VOR y North [m]", value=0.0, step=100.0)
        h_ant = st.number_input("ارتفاع آنتن [m]", value=5.0, step=1.0, min_value=0.5)

        st.header("🗺️ شبکه زمین")
        h_base = st.number_input("ارتفاع پایه [m MSL]", value=0.0, step=10.0)
        L = st.number_input("نصف اندازه ناحیه [m]", value=6000.0, step=500.0, min_value=500.0)
        cell = st.number_input("اندازه سلول [m]", value=50.0, step=10.0, min_value=10.0)
        min_slope = st.number_input("حداقل شیب فاست [deg]", value=3.0, step=0.5, min_value=0.5)
        r_min = st.number_input("حداقل فاصله از VOR [m]", value=100.0, step=10.0, min_value=10.0)

        st.header("🔧 پراکندگی")
        gamma_dB = st.number_input("γ (constant-gamma) [dB]",
                                    value=-20.0, step=1.0)
        phi_ref_deg = st.number_input("φ_ref [deg]", value=180.0, step=10.0)

        st.header("✈️ هواپیما")
        R_air = st.number_input("شعاع پرواز [m]", value=10000.0, step=500.0, min_value=500.0)
        h_air = st.number_input("ارتفاع هواپیما [m MSL]", value=1000.0, step=50.0)
        az_step = st.number_input("گام آزیموت [deg]", value=1.0, step=0.25, min_value=0.25)
        tol = st.number_input("تلورانس [deg]", value=1.0, step=0.5, min_value=0.1)

        st.header("📻 سیستم")
        f0_MHz = st.number_input("فرکانس [MHz]", value=112.0, step=0.5,
                                  min_value=108.0, max_value=118.0)
        seed = st.number_input("بذر تصادفی", value=1, step=1, min_value=0)
        td_check = st.checkbox("اعتبارسنجی زمان‌دامنه (Eq. 2-4)",
                                value=False,
                                help="محاسبه اضافی و کندتر")

        st.header("⛰️ ویژگی‌های زمین")
        st.caption("تپه/پشته اضافه کنید. اگر خالی باشد، زمین صاف است.")

        default_features = pd.DataFrame([
            {"type": "hill", "x1": 2000.0, "y1": 2000.0, "x2": np.nan,
             "y2": np.nan, "H": 150.0, "s": 500.0},
        ])
        feats = st.data_editor(
            default_features, num_rows="dynamic",
            use_container_width=True,
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "نوع", options=["hill", "ridge"], required=True),
                "x1": st.column_config.NumberColumn("x1 [m] (hill:cx)"),
                "y1": st.column_config.NumberColumn("y1 [m] (hill:cy)"),
                "x2": st.column_config.NumberColumn("x2 [m] (ridge:end)"),
                "y2": st.column_config.NumberColumn("y2 [m] (ridge:end)"),
                "H": st.column_config.NumberColumn("H [m]", min_value=0.0),
                "s": st.column_config.NumberColumn("s یا w [m]", min_value=1.0),
            }, key="terrain_features")

        st.header("📂 DEM اختیاری")
        uploaded = st.file_uploader(
            "فایل DEM (.csv / .txt / .mat)",
            type=["csv", "txt", "mat"],
            help="ستون‌ها: x y z (CSV/TXT) یا متغیرهای X,Y,Z (MAT)")

        st.markdown("---")
        st.caption("⚠️ با سلول کوچک و ناحیه بزرگ، شبیه‌سازی کند می‌شود.")
        run = st.button("🚀 اجرای شبیه‌سازی",
                        use_container_width=True, type="primary")


    # --- پردازش ورودی‌ها و اجرا ---
    if run:
        # --- آماده‌سازی ویژگی‌های زمین ---
        features = []
        for _, row in feats.iterrows():
            if row["type"] == "hill":
                if pd.isna(row["x1"]) or pd.isna(row["y1"]) or \
                   pd.isna(row["H"]) or pd.isna(row["s"]):
                    continue
                features.append({"type": "hill", "x": float(row["x1"]),
                                 "y": float(row["y1"]), "H": float(row["H"]),
                                 "s": float(row["s"])})
            elif row["type"] == "ridge":
                vals = [row["x1"], row["y1"], row["x2"], row["y2"],
                        row["H"], row["s"]]
                if any(pd.isna(v) for v in vals):
                    continue
                features.append({"type": "ridge",
                                 "x1": float(row["x1"]), "y1": float(row["y1"]),
                                 "x2": float(row["x2"]), "y2": float(row["y2"]),
                                 "H": float(row["H"]), "w": float(row["s"])})

        # --- DEM ---
        dem_X = dem_Y = dem_Z = None
        dem_points = None
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".mat"):
                    import scipy.io as sio
                    S = sio.loadmat(io.BytesIO(uploaded.read()))
                    if "X" in S and "Y" in S and "Z" in S:
                        dem_X = np.asarray(S["X"]).squeeze()
                        dem_Y = np.asarray(S["Y"]).squeeze()
                        dem_Z = np.asarray(S["Z"])
                else:
                    raw = uploaded.read().decode("utf-8", errors="ignore")
                    arr = np.loadtxt(io.StringIO(raw), delimiter=None,
                                     comments="#")
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        dem_points = arr[:, :3]
            except Exception as e:
                st.warning(f"خطا در بارگذاری DEM: {e}. از زمین مسطح استفاده می‌شود.")

        cfg = dict(vor_xy=(vor_x, vor_y), h_ant=h_ant, h_base=h_base,
                   L=L, cell=cell, gamma_dB=gamma_dB,
                   min_slope=min_slope, r_min=r_min,
                   phi_ref_deg=phi_ref_deg, R_air=R_air, h_air=h_air,
                   az_step=az_step, tol=tol, f0_MHz=f0_MHz,
                   seed=seed, td_check=td_check)

        # --- زمین ---
        with st.spinner("ساخت شبکه زمین..."):
            G = build_terrain_xy(cfg["vor_xy"], L, cell, h_base, features,
                                 dem_X=dem_X, dem_Y=dem_Y, dem_Z=dem_Z,
                                 dem_points=dem_points)
        st.success(
            f"🌍 زمین: {G['Z'].shape[1]}×{G['Z'].shape[0]} سلول | "
            f"z_vor={G['z_vor']:.1f} m | "
            f"بیشینه {G['max_z']:.1f} m"
        )

        # --- فاست‌ها ---
        with st.spinner("ساخت فاست‌ها..."):
            F = terrain_build_facets(cfg, G)
        st.info(f"🔷 فاست‌های قابل مشاهده از VOR: **{F['N']}**")

        if F["N"] == 0:
            st.warning("هیچ فاستی از VOR دیده نمی‌شود — خطا صفر خواهد بود. "
                       "شیب/ویژگی‌ها را بررسی کنید.")
            st.stop()

        # --- اجرا ---
        progress_bar = st.progress(0, text="در حال شبیه‌سازی...")

        def upd(frac):
            progress_bar.progress(min(max(frac, 0.0), 1.0),
                                  text=f"در حال شبیه‌سازی... {frac*100:.0f}%")

        t0 = time.time()
        with st.spinner("در حال اجرا..."):
            res = terrain_run_simulation(cfg, G, F, progress_cb=upd)
        progress_bar.empty()
        st.success(f"✅ در {time.time()-t0:.1f} ثانیه انجام شد.")

        # --- آماره‌ها ---
        e = res["eps_314"]
        st.markdown("### 📊 خلاصه آماری")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">بیشینه |خطا|</div>'
                f'<div class="metric-value">{np.max(np.abs(e)):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">RMS</div>'
                f'<div class="metric-value">{np.sqrt(np.mean(e**2)):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">95%</div>'
                f'<div class="metric-value">{p95(e):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">درصد نقض ±{tol:.1f}°</div>'
                f'<div class="metric-value">{100*np.mean(np.abs(e) > tol):.1f}%</div></div>',
                unsafe_allow_html=True)

        st.info(
            f"پراکنده‌های فعال در هر آزیموت: میانگین **{np.mean(res['nsc']):.0f}** | "
            f"بیشینه **{np.max(res['nsc'])}** | "
            f"بدترین آزیموت **{res['worst']['az']:.1f}°**"
        )

        if td_check and not np.all(np.isnan(res["eps_td"])):
            d = res["eps_td"] - e
            st.success(
                f"✅ اعتبارسنجی زمان‌دامنه (Eq. 2-4): "
                f"بیشینه اختلاف **{np.max(np.abs(d)):.4f}°** | "
                f"RMS اختلاف **{np.sqrt(np.mean(d**2)):.4f}°**"
            )

        st.divider()

        # ============================================================
        # شکل ۱: زمین و فاست‌های موثر
        # ============================================================
        st.subheader("🗺️ شکل ۱ — زمین و فاست‌های موثر")

        fig1 = make_subplots(rows=1, cols=2,
                             subplot_titles=("ارتفاع زمین [m MSL]",
                                             f"فاست‌های موثر در آزیموت {res['worst']['az']:.1f}°"))
        # نقشه ارتفاع
        fig1.add_trace(go.Contour(
            x=G["X_abs"][0, :], y=G["Y_abs"][:, 0], z=G["Z"],
            colorscale="Earth", showscale=True,
            line=dict(width=0), ncontours=20), row=1, col=1)
        fig1.add_trace(go.Scatter(
            x=[vor_x], y=[vor_y], mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="red"),
            text=["VOR"], textposition="top center",
            showlegend=False), row=1, col=1)

        # فاست‌های فعال
        wx = res["worst"]["x"] + vor_x
        wy = res["worst"]["y"] + vor_y
        wa = res["worst"]["a"]
        if len(wx) > 0:
            fig1.add_trace(go.Scatter(
                x=wx, y=wy, mode="markers",
                marker=dict(size=6, color=wa, colorscale="Viridis",
                            showscale=True, colorbar=dict(x=1.15)),
                showlegend=False), row=1, col=2)
        fig1.add_trace(go.Scatter(
            x=[vor_x], y=[vor_y], mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="red"),
            showlegend=False), row=1, col=2)
        # خط دید به بدترین آزیموت
        ax = res["worst"]["az"]
        Lp = 0.9 * L
        fig1.add_trace(go.Scatter(
            x=[vor_x, vor_x + Lp * np.sin(np.deg2rad(ax))],
            y=[vor_y, vor_y + Lp * np.cos(np.deg2rad(ax))],
            mode="lines", line=dict(color="black", dash="dash"),
            showlegend=False), row=1, col=2)

        fig1.update_xaxes(title_text="x East [m]", row=1, col=1)
        fig1.update_yaxes(title_text="y North [m]", row=1, col=1,
                          scaleanchor="x", scaleratio=1)
        fig1.update_xaxes(title_text="x East [m]", row=1, col=2)
        fig1.update_yaxes(title_text="y North [m]", row=1, col=2,
                          scaleanchor="x2", scaleratio=1)
        fig1.update_layout(template="plotly_dark", height=520,
                           margin=dict(l=40, r=80, t=60, b=40))
        st.plotly_chart(fig1, use_container_width=True)

        # ============================================================
        # شکل ۲: خطا (سه زیرنمودار)
        # ============================================================
        st.subheader("📈 شکل ۲ — خطای زاویه ناشی از توپوگرافی")

        fig2 = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "خطای زاویه (Eq. 3-14) در کل ۰–۳۶۰°",
                f"zoom حول بدترین آزیموت ({res['worst']['az']:.1f}°)",
                "هیستوگرام خطای زاویه"),
            vertical_spacing=0.08,
            row_heights=[0.40, 0.30, 0.30])

        az = res["az"]; ez = res["az_z"]; eps_z = res["eps_z"]

        fig2.add_trace(go.Scatter(x=az, y=e, mode="lines",
                                  name="Eq. 3-14",
                                  line=dict(color="white", width=1.4)), row=1, col=1)
        if td_check and not np.all(np.isnan(res["eps_td"])):
            fig2.add_trace(go.Scatter(x=az, y=res["eps_td"], mode="markers",
                                      name="زمان‌دامنه",
                                      marker=dict(color="red", size=4)),
                           row=1, col=1)
        fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot"), row=1, col=1)
        fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot"), row=1, col=1)
        fig2.add_hline(y=0, line=dict(color="white", dash="dot"), row=1, col=1)

        fig2.add_trace(go.Scatter(x=ez, y=eps_z, mode="lines",
                                  line=dict(color="white", width=1.0),
                                  showlegend=False), row=2, col=1)
        fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot"), row=2, col=1)
        fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot"), row=2, col=1)

        fig2.add_trace(go.Histogram(x=e, nbinsx=40,
                                    marker_color="rgba(60,90,220,0.75)",
                                    showlegend=False), row=3, col=1)
        fig2.add_vline(x=tol, line=dict(color="magenta", dash="dot"), row=3, col=1)
        fig2.add_vline(x=-tol, line=dict(color="magenta", dash="dot"), row=3, col=1)

        fig2.update_xaxes(title_text="آزیموت هواپیما [deg]", row=1, col=1)
        fig2.update_xaxes(title_text="آزیموت [deg]", row=2, col=1)
        fig2.update_xaxes(title_text="خطای زاویه [deg]", row=3, col=1)
        fig2.update_yaxes(title_text="خطا [deg]", row=1, col=1)
        fig2.update_yaxes(title_text="خطا [deg]", row=2, col=1)
        fig2.update_yaxes(title_text="شمارش", row=3, col=1)
        fig2.update_layout(template="plotly_dark", height=1050,
                           legend=dict(orientation="h", y=1.02),
                           margin=dict(l=50, r=20, t=80, b=40))
        st.plotly_chart(fig2, use_container_width=True)

        # ============================================================
        # خلاصه نهایی
        # ============================================================
        with st.expander("📋 خلاصه پارامترها و نتایج"):
            st.markdown("**پارامترهای شبیه‌سازی:**")
            st.markdown(f"- VOR: ({vor_x:.0f}, {vor_y:.0f}) | h_ant={h_ant:.1f} m")
            st.markdown(f"- شبکه: {G['Z'].shape[1]}×{G['Z'].shape[0]} سلول | "
                        f"سلول={cell:.0f} m | L={L:.0f} m")
            st.markdown(f"- γ={gamma_dB:.1f} dB | φ_ref={phi_ref_deg:.1f}° | "
                        f"min_slope={min_slope:.1f}°")
            st.markdown(f"- R_air={R_air:.0f} m | h_air={h_air:.0f} m | "
                        f"az_step={az_step:.2f}° | tol=±{tol:.1f}°")
            st.markdown(f"- f₀={f0_MHz:.1f} MHz | seed={seed}")
            st.markdown("")
            st.markdown("**نتایج:**")
            st.markdown(f"- فاست‌های قابل مشاهده: **{F['N']}**")
            st.markdown(f"- میانگین پراکنده فعال: **{np.mean(res['nsc']):.1f}** | "
                        f"بیشینه **{np.max(res['nsc'])}**")
            st.markdown(f"- بیشینه |خطا|: **{np.max(np.abs(e)):.3f}°**")
            st.markdown(f"- RMS: **{np.sqrt(np.mean(e**2)):.3f}°**")
            st.markdown(f"- 95%: **{p95(e):.3f}°**")
            st.markdown(f"- درصد نقض ±{tol:.1f}°: **{100*np.mean(np.abs(e) > tol):.1f}%**")
            st.markdown(f"- بدترین آزیموت: **{res['worst']['az']:.2f}°**")

            st.markdown("**خروجی خام JSON:**")
            st.json({
                "vor_xy": list(cfg["vor_xy"]),
                "h_ant": cfg["h_ant"],
                "h_base": cfg["h_base"],
                "L": cfg["L"], "cell": cfg["cell"],
                "gamma_dB": cfg["gamma_dB"],
                "min_slope": cfg["min_slope"],
                "r_min": cfg["r_min"],
                "R_air": cfg["R_air"], "h_air": cfg["h_air"],
                "az_step": cfg["az_step"], "tol": cfg["tol"],
                "f0_MHz": cfg["f0_MHz"], "seed": cfg["seed"],
                "phi_ref_deg": cfg["phi_ref_deg"],
                "n_facets_visible": int(F["N"]),
                "max_abs_err_deg": float(np.max(np.abs(e))),
                "rms_err_deg": float(np.sqrt(np.mean(e**2))),
                "p95_err_deg": p95(e),
                "worst_az_deg": res["worst"]["az"],
                "frac_outside_tol": float(np.mean(np.abs(e) > tol)),
            })

    else:
        st.info(
            "👈 پارامترهای VOR، شبکه زمین، پراکندگی، هواپیما و ویژگی‌های "
            "توپوگرافی را در نوار کناری تنظیم کنید و دکمه «اجرای شبیه‌سازی» "
            "را بزنید.\n\n"
            "**نکته مهم:** مدل زمین در این شبیه‌ساز **ذاتی معادلات تز نیست** — "
            "مدل constant-gamma و پراکنده نقطه‌ای فاست‌ها یک انتخاب مهندسی است "
            "که باید در متن تز **ارجاع و توجیه** شود (مثلاً با ارجاع به "
            "Ulaby & Dobson یا یک مقاله مرتبط با پراکندگی سطح).\n\n"
            "**پیشنهاد برای شروع:** یک تپه گاوسی با H=150 m و s=500 m در "
            "فاصله ~2 km از VOR — این حالت خطای قابل‌توجهی تولید می‌کند."
        )


# ##########################################################
# ۳) جنگل
# ##########################################################
# ============================================================
# توابع کمکی
# ============================================================
def point_in_polygon(x, y, V):
    """بررسی درون/بیرون چندضلعی (ray casting)."""
    n = len(V)
    inside = np.zeros(len(x), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = V[i, 0], V[i, 1]
        xj, yj = V[j, 0], V[j, 1]
        cond1 = ((yi > y) != (yj > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
        hit = cond1 & (x < x_int)
        inside ^= hit
        j = i
    return inside


def polyarea(V):
    """مساحت چندضلعی."""
    x = V[:, 0]; y = V[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ============================================================
# ساخت جنگل از پچ‌ها
# ============================================================
def build_forest(cfg, patches):
    """تولید موقعیت درختان برای همه پچ‌ها."""
    rng = np.random.default_rng(int(cfg["seed"]))
    xt = []; yt = []; zt = []; sg = []; pid = []

    for p_idx, P in enumerate(patches, start=1):
        if P["shape"] == "circle":
            area = np.pi * P["R"] ** 2
            Nt = int(round(P["density"] * area / 1e4))
            r = P["R"] * np.sqrt(rng.random(Nt))
            ang = 2 * np.pi * rng.random(Nt)
            x = P["cx"] + r * np.cos(ang)
            y = P["cy"] + r * np.sin(ang)

        elif P["shape"] == "rect":
            area = P["L"] * P["W"]
            Nt = int(round(P["density"] * area / 1e4))
            u = (rng.random(Nt) - 0.5) * P["L"]
            v = (rng.random(Nt) - 0.5) * P["W"]
            b = np.deg2rad(P["beta"])
            x = P["cx"] + u * np.sin(b) + v * np.cos(b)
            y = P["cy"] + u * np.cos(b) - v * np.sin(b)

        elif P["shape"] == "polygon":
            V = np.asarray(P["V"], dtype=float)
            area = polyarea(V)
            Nt = int(round(P["density"] * area / 1e4))
            bx = [V[:, 0].min(), V[:, 0].max()]
            by = [V[:, 1].min(), V[:, 1].max()]
            x = np.zeros(0); y = np.zeros(0); it = 0
            while len(x) < Nt and it < 1000:
                it += 1
                xs = bx[0] + (bx[1] - bx[0]) * rng.random(max(Nt, 100))
                ys = by[0] + (by[1] - by[0]) * rng.random(max(Nt, 100))
                inside = point_in_polygon(xs, ys, V)
                x = np.concatenate([x, xs[inside]])
                y = np.concatenate([y, ys[inside]])
            Nt = min(Nt, len(x))
            x = x[:Nt]; y = y[:Nt]

        else:
            continue

        if Nt == 0:
            continue

        z = P["hmin"] + (P["hmax"] - P["hmin"]) * rng.random(Nt)

        if cfg["amp_model"] == 1:
            # power → dB/10
            s = P["strength"] * 10 ** (P["strength_sd"] * rng.standard_normal(Nt) / 10)
        else:
            # amplitude → dB/20
            s = P["strength"] * 10 ** (P["strength_sd"] * rng.standard_normal(Nt) / 20)

        xt.append(x); yt.append(y); zt.append(z)
        sg.append(s); pid.append(np.full(Nt, p_idx))

    if not xt:
        return None

    xt = np.concatenate(xt); yt = np.concatenate(yt)
    zt = np.concatenate(zt); sg = np.concatenate(sg)
    pid = np.concatenate(pid)
    dh = np.hypot(xt, yt)
    return dict(x=xt, y=yt, z=zt, strength=sg, pid=pid,
                N=len(xt),
                dmin=float(np.min(dh)), dmax=float(np.max(dh)),
                areas=[np.pi * P["R"] ** 2 if P["shape"] == "circle"
                       else (P["L"] * P["W"] if P["shape"] == "rect"
                             else polyarea(np.asarray(P["V"], float)))
                       for P in patches])


# ============================================================
# پارامترهای پراکنده + Eq. (3-14)
# ============================================================
def scatterer_params(cfg, T, Ax, Ay):
    """محاسبه a_n، theta_n، phi_n برای یک موقعیت هواپیما."""
    c = 299792458.0
    lam = c / (cfg["f0_MHz"] * 1e6)
    phi_ref = np.deg2rad(cfg["phi_ref_deg"])

    d1 = np.sqrt(T["x"] ** 2 + T["y"] ** 2 + (T["z"] - cfg["h_ant"]) ** 2)
    d2 = np.sqrt((Ax - T["x"]) ** 2 + (Ay - T["y"]) ** 2
                 + (cfg["h_air"] - T["z"]) ** 2)
    dd = np.sqrt(Ax ** 2 + Ay ** 2 + (cfg["h_air"] - cfg["h_ant"]) ** 2)

    if cfg["amp_model"] == 1:
        a = np.sqrt(T["strength"] / (4 * np.pi)) * dd / (d1 * d2)
    else:
        a = T["strength"] * dd / (d1 + d2)

    theta = -2 * np.pi * ((d1 + d2) - dd) / lam + phi_ref
    azt = np.arctan2(T["x"], T["y"])
    return a, theta, azt


def forest_compute_eq314(a, theta, azt, pa):
    """Eq. (3-14): eps = atan2(-Σ a cosθ sinφ, 1 + Σ a cosθ cosφ)."""
    if len(a) == 0:
        return 0.0
    phin = azt - pa
    num = -np.sum(a * np.cos(theta) * np.sin(phin))
    den = 1.0 + np.sum(a * np.cos(theta) * np.cos(phin))
    return float(np.rad2deg(np.arctan2(num, den)))


# ============================================================
# اعتبارسنجی زمان‌دامنه (Eq. 2-4)
# ============================================================
def time_domain_check(cfg, T, az):
    """Envelope detector با lock-in در 30 Hz."""
    ma = 0.30; fLF = 30.0; w = 2 * np.pi * fLF
    fs = 3000.0
    t = np.arange(0, 1.0, 1 / fs)
    e30 = np.exp(-1j * w * t)
    ejwt = np.exp(1j * w * t)
    pa = np.deg2rad(az)
    eps_td = np.zeros(len(az))
    for k in range(len(az)):
        a, theta, azt = scatterer_params(
            cfg, T, cfg["R_air"] * np.sin(pa[k]),
            cfg["R_air"] * np.cos(pa[k]))
        g = a * np.exp(1j * theta)
        M0 = np.sum(g)
        Cc = np.sum(g * np.exp(-1j * azt))
        Cp = np.sum(g * np.exp(1j * azt))
        s = ((1 + ma * np.cos(w * t - pa[k]))
             + M0 + 0.5 * ma * (Cc * ejwt + Cp * np.conj(ejwt)))
        env = np.abs(s)
        X = np.sum(env * e30)
        phi_meas = -np.angle(X)
        eps_td[k] = np.rad2deg(
            np.angle(np.exp(1j * (pa[k] - phi_meas))))
    return eps_td


# ============================================================
# اجرای شبیه‌سازی
# ============================================================
def forest_run_simulation(cfg, T, progress_cb=None):
    az = np.arange(0, 360, cfg["az_step"])
    pa = np.deg2rad(az)
    N = len(az)

    eps_314 = np.zeros(N)
    for k in range(N):
        a, theta, azt = scatterer_params(
            cfg, T, cfg["R_air"] * np.sin(pa[k]),
            cfg["R_air"] * np.cos(pa[k]))
        eps_314[k] = forest_compute_eq314(a, theta, azt, pa[k])
        if progress_cb and (k % max(1, N // 20) == 0 or k == N - 1):
            progress_cb(0.8 * k / N)

    # --- zoom ---
    ip = int(np.argmax(np.abs(eps_314)))
    az_z = (az[ip] - 0.25) + np.arange(0, 0.5 + 0.0005, 0.0005)
    paz = np.deg2rad(az_z)
    eps_z = np.zeros(len(az_z))
    for k in range(len(az_z)):
        a, theta, azt = scatterer_params(
            cfg, T, cfg["R_air"] * np.sin(paz[k]),
            cfg["R_air"] * np.cos(paz[k]))
        eps_z[k] = forest_compute_eq314(a, theta, azt, paz[k])
        if progress_cb and k % max(1, len(az_z) // 10) == 0:
            progress_cb(0.8 + 0.1 * k / len(az_z))

    # --- اعتبارسنجی ---
    eps_td = None
    if cfg["td_check"]:
        if progress_cb: progress_cb(0.9)
        eps_td = time_domain_check(cfg, T, az)
        if progress_cb: progress_cb(1.0)

    return dict(az=az, eps_314=eps_314, eps_td=eps_td,
                az_z=az_z, eps_z=eps_z,
                worst_az=float(az[ip]))


def page_forest():
    st.title("🌲 خطای CVOR ناشی از جنگل (Eq. 3-14)")
    st.caption(
        "چند قطعه جنگلی (دایره/مستطیل/چندضلعی) + Eq. (3-14) با REF ایده‌آل — "
        "تبدیل‌شده از MATLAB"
    )

    # --- نوار کناری ---
    with st.sidebar:
        st.header("📡 VOR")
        vor_x = st.number_input("VOR x East [m]", value=0.0, step=100.0)
        vor_y = st.number_input("VOR y North [m]", value=0.0, step=100.0)
        h_ant = st.number_input("ارتفاع آنتن [m]", value=5.0, step=1.0, min_value=0.5)

        st.header("🔧 مدل دامنه a_n")
        amp_model = st.radio(
            "انتخاب مدل",
            [1, 2],
            format_func=lambda x: (
                "۱) پراکندگی bistatic: a = √(σ/4π)·d/(d1·d2)"
                if x == 1 else
                "۲) ضریب بازتاب: a = ρ·d/(d1+d2)"),
            index=0)

        st.header("✈️ هواپیما")
        R_air = st.number_input("شعاع پرواز [m]",
                                value=10000.0, step=500.0, min_value=500.0)
        h_air = st.number_input("ارتفاع هواپیما [m]",
                                value=1000.0, step=50.0, min_value=50.0)
        az_step = st.number_input("گام آزیموت [deg]",
                                  value=0.5, step=0.1, min_value=0.05)
        tol = st.number_input("تلورانس [deg]",
                              value=1.0, step=0.5, min_value=0.1)

        st.header("📻 سیستم")
        f0_MHz = st.number_input("فرکانس [MHz]", value=112.0, step=0.5,
                                  min_value=108.0, max_value=118.0)
        phi_ref_deg = st.number_input("φ_ref [deg]", value=180.0, step=10.0)
        seed = st.number_input("بذر تصادفی", value=1, step=1, min_value=0)
        td_check = st.checkbox("اعتبارسنجی زمان‌دامنه (Eq. 2-4)",
                                value=False,
                                help="محاسبه اضافی و کندتر")

        st.header("🌳 قطعات جنگلی")
        st.caption("هر ردیف یک قطعه. نام + پارامترهای شکل.")

        # --- جدول پچ‌ها ---
        default_patches = pd.DataFrame([
            {"name": "P1", "shape": "circle",
             "cx": 1060.66, "cy": 1060.66, "R": 1200.0,
             "L": np.nan, "W": np.nan, "beta": np.nan,
             "polygon": "",
             "density": 5.0, "strength": 10.0, "strength_sd": 3.0,
             "hmin": 8.0, "hmax": 20.0},
        ])

        patches_edited = st.data_editor(
            default_patches, num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("نام"),
                "shape": st.column_config.SelectboxColumn(
                    "شکل", options=["circle", "rect", "polygon"], required=True),
                "cx": st.column_config.NumberColumn("cx یا مرکز x [m]"),
                "cy": st.column_config.NumberColumn("cy یا مرکز y [m]"),
                "R": st.column_config.NumberColumn("شعاع (circle) [m]"),
                "L": st.column_config.NumberColumn("طول (rect) [m]"),
                "W": st.column_config.NumberColumn("عرض (rect) [m]"),
                "beta": st.column_config.NumberColumn("زاویه محور (rect) [deg]"),
                "polygon": st.column_config.TextColumn(
                    "چندضلعی: 'x1,y1;x2,y2;...'"),
                "density": st.column_config.NumberColumn("تراکم [درخت/هکتار]",
                                                         min_value=0.1),
                "strength": st.column_config.NumberColumn(
                    "σ یا ρ", min_value=0.0),
                "strength_sd": st.column_config.NumberColumn("انحراف [dB]",
                                                             min_value=0.0),
                "hmin": st.column_config.NumberColumn("h_min [m]", min_value=0.0),
                "hmax": st.column_config.NumberColumn("h_max [m]", min_value=0.0),
            }, key="forest_patches")

        st.caption(
            "💡 برای چندضلعی: رشته مثل `300,300;1500,300;1500,1500;300,1500` "
            "(مختصات نسبت به VOR).")

        st.markdown("---")
        st.caption("⚠️ گام آزیموت کوچک = کندتر. cell=0.5° معمولاً کافی است.")
        run = st.button("🚀 اجرای شبیه‌سازی",
                        use_container_width=True, type="primary")


    # --- پردازش و اجرا ---
    if run:
        # --- تبدیل جدول به لیست پچ ---
        patches = []
        for _, row in patches_edited.iterrows():
            shape = str(row["shape"])
            if pd.isna(row["density"]) or pd.isna(row["strength"]):
                continue

            base = dict(
                name=str(row["name"]) if not pd.isna(row["name"]) else "P",
                shape=shape,
                density=float(row["density"]),
                strength=float(row["strength"]),
                strength_sd=float(row["strength_sd"]) if not pd.isna(row["strength_sd"]) else 0.0,
                hmin=float(row["hmin"]) if not pd.isna(row["hmin"]) else 8.0,
                hmax=float(row["hmax"]) if not pd.isna(row["hmax"]) else 20.0,
            )

            if shape == "circle":
                if pd.isna(row["cx"]) or pd.isna(row["cy"]) or pd.isna(row["R"]):
                    continue
                base.update(cx=float(row["cx"]), cy=float(row["cy"]),
                            R=float(row["R"]))
            elif shape == "rect":
                if any(pd.isna(row[c]) for c in ["cx", "cy", "L", "W", "beta"]):
                    continue
                base.update(cx=float(row["cx"]), cy=float(row["cy"]),
                            L=float(row["L"]), W=float(row["W"]),
                            beta=float(row["beta"]))
            elif shape == "polygon":
                s = str(row["polygon"]).strip()
                if not s:
                    continue
                try:
                    coords = []
                    for pt in s.split(";"):
                        xy = [float(v.strip()) for v in pt.split(",")]
                        if len(xy) != 2:
                            raise ValueError
                        coords.append(xy)
                    if len(coords) < 3:
                        continue
                    base.update(V=np.array(coords, dtype=float))
                except Exception:
                    st.warning(f"چندضلعی نامعتبر در ردیف {row['name']}: {s}")
                    continue

            patches.append(base)

        if not patches:
            st.error("حداقل یک قطعه معتبر لازم است.")
            st.stop()

        # --- ساخت cfg ---
        cfg = dict(vor_xy=(vor_x, vor_y), h_ant=h_ant,
                   amp_model=amp_model, R_air=R_air, h_air=h_air,
                   az_step=az_step, tol=tol, f0_MHz=f0_MHz,
                   phi_ref_deg=phi_ref_deg, seed=seed,
                   td_check=td_check)

        # --- ساخت جنگل ---
        with st.spinner("تولید درختان..."):
            T = build_forest(cfg, patches)

        if T is None or T["N"] == 0:
            st.error("هیچ درختی تولید نشد. پارامترهای پچ‌ها را بررسی کنید.")
            st.stop()

        st.success(
            f"🌳 **{T['N']} درخت** | نزدیک‌ترین: {T['dmin']:.0f} m | "
            f"دورترین: {T['dmax']:.0f} m"
        )
        for i, P in enumerate(patches, start=1):
            st.caption(f"  - پچ {i} ({P['shape']}): "
                       f"{np.pi*P['R']**2/1e4:.2f} ha" if P["shape"] == "circle"
                       else f"  - پچ {i} ({P['shape']}): "
                            f"{(P['L']*P['W'] if P['shape']=='rect' else polyarea(np.asarray(P['V'], float)))/1e4:.2f} ha")

        # --- اجرا ---
        progress_bar = st.progress(0, text="در حال شبیه‌سازی...")

        def upd(frac):
            progress_bar.progress(min(max(frac, 0.0), 1.0),
                                  text=f"در حال شبیه‌سازی... {frac*100:.0f}%")

        t0 = time.time()
        with st.spinner("در حال اجرا..."):
            res = forest_run_simulation(cfg, T, progress_cb=upd)
        progress_bar.empty()
        st.success(f"✅ در {time.time()-t0:.1f} ثانیه انجام شد.")

        # --- آماره‌ها ---
        e = res["eps_314"]
        st.markdown("### 📊 خلاصه آماری")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">بیشینه |خطا|</div>'
                f'<div class="metric-value">{np.max(np.abs(e)):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">RMS</div>'
                f'<div class="metric-value">{np.sqrt(np.mean(e**2)):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">95%</div>'
                f'<div class="metric-value">{p95(e):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">نقض ±{tol:.1f}°</div>'
                f'<div class="metric-value">{100*np.mean(np.abs(e)>tol):.1f}%</div></div>',
                unsafe_allow_html=True)

        st.info(
            f"مدل دامنه: **{'bistatic σ' if amp_model==1 else 'reflection ρ'}** | "
            f"بدترین آزیموت: **{res['worst_az']:.2f}°**"
        )

        if td_check and res["eps_td"] is not None:
            d = res["eps_td"] - e
            st.success(
                f"✅ اعتبارسنجی Eq. (2-4): "
                f"بیشینه اختلاف **{np.max(np.abs(d)):.4f}°** | "
                f"RMS اختلاف **{np.sqrt(np.mean(d**2)):.4f}°**"
            )

        st.divider()

        # ============================================================
        # شکل ۱: هندسه جنگل
        # ============================================================
        st.subheader("🗺️ شکل ۱ — هندسه جنگل")

        fig1 = go.Figure()
        # درختان با رنگ بر اساس شماره پچ
        fig1.add_trace(go.Scatter(
            x=T["x"] + vor_x, y=T["y"] + vor_y,
            mode="markers",
            marker=dict(size=3, color=T["pid"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="پچ")),
            name="درختان"))
        # دایره نزدیک‌ترین درخت
        th = np.linspace(0, 2 * np.pi, 200)
        fig1.add_trace(go.Scatter(
            x=vor_x + T["dmin"] * np.sin(th),
            y=vor_y + T["dmin"] * np.cos(th),
            mode="lines", line=dict(color="white", dash="dot"),
            name=f"نزدیک‌ترین درخت ({T['dmin']:.0f} m)"))
        # VOR
        fig1.add_trace(go.Scatter(
            x=[vor_x], y=[vor_y], mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="red"),
            text=["VOR"], textposition="top center",
            name="VOR"))

        fig1.update_layout(
            template="plotly_dark", height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            xaxis_title="x East [m]", yaxis_title="y North [m]",
            margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig1, use_container_width=True)

        # ============================================================
        # شکل ۲: خطا (سه زیرنمودار)
        # ============================================================
        st.subheader("📈 شکل ۲ — خطای زاویه ناشی از جنگل")

        fig2 = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "خطای Eq. (3-14) در کل ۰–۳۶۰°",
                f"zoom حول بدترین آزیموت ({res['worst_az']:.2f}°)",
                "هیستوگرام خطا"),
            vertical_spacing=0.08,
            row_heights=[0.40, 0.30, 0.30])

        az = res["az"]; ez = res["az_z"]; eps_z = res["eps_z"]

        fig2.add_trace(go.Scatter(
            x=az, y=e, mode="lines", name="Eq. 3-14",
            line=dict(color="white", width=1.4)), row=1, col=1)
        if td_check and res["eps_td"] is not None:
            fig2.add_trace(go.Scatter(
                x=az, y=res["eps_td"], mode="markers",
                name="زمان‌دامنه",
                marker=dict(color="red", size=4)), row=1, col=1)
        fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot"), row=1, col=1)
        fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot"), row=1, col=1)
        fig2.add_hline(y=0, line=dict(color="white", dash="dot"), row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=ez, y=eps_z, mode="lines",
            line=dict(color="white", width=1.0),
            showlegend=False), row=2, col=1)
        fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot"), row=2, col=1)
        fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot"), row=2, col=1)

        fig2.add_trace(go.Histogram(
            x=e, nbinsx=40,
            marker_color="rgba(60,90,220,0.75)",
            showlegend=False), row=3, col=1)
        fig2.add_vline(x=tol, line=dict(color="magenta", dash="dot"), row=3, col=1)
        fig2.add_vline(x=-tol, line=dict(color="magenta", dash="dot"), row=3, col=1)

        fig2.update_xaxes(title_text="آزیموت هواپیما [deg]", row=1, col=1)
        fig2.update_xaxes(title_text="آزیموت [deg]", row=2, col=1)
        fig2.update_xaxes(title_text="خطای زاویه [deg]", row=3, col=1)
        fig2.update_yaxes(title_text="خطا [deg]", row=1, col=1)
        fig2.update_yaxes(title_text="خطا [deg]", row=2, col=1)
        fig2.update_yaxes(title_text="شمارش", row=3, col=1)
        fig2.update_layout(template="plotly_dark", height=1050,
                           legend=dict(orientation="h", y=1.02),
                           margin=dict(l=50, r=20, t=80, b=40))
        st.plotly_chart(fig2, use_container_width=True)

        # ============================================================
        # جدول نهایی
        # ============================================================
        with st.expander("📋 خلاصه پارامترها و نتایج"):
            st.markdown("**پارامترهای سیستم:**")
            st.markdown(f"- VOR: ({vor_x:.0f}, {vor_y:.0f}) | h_ant={h_ant:.1f} m")
            st.markdown(f"- f₀={f0_MHz:.1f} MHz | λ={299792458/(f0_MHz*1e6):.3f} m")
            st.markdown(f"- R_air={R_air:.0f} m | h_air={h_air:.0f} m")
            st.markdown(f"- az_step={az_step:.3f}° | tol=±{tol:.1f}°")
            st.markdown(f"- مدل دامنه: {amp_model} | φ_ref={phi_ref_deg:.1f}°")
            st.markdown(f"- seed={seed} | td_check={td_check}")
            st.markdown("")
            st.markdown("**قطعات جنگلی:**")
            for i, P in enumerate(patches, start=1):
                st.markdown(f"- **{P['name']}** ({P['shape']}): "
                            f"density={P['density']:.1f} | "
                            f"strength={P['strength']:.3g} | "
                            f"h=[{P['hmin']:.0f},{P['hmax']:.0f}] m")
            st.markdown("")
            st.markdown("**نتایج:**")
            st.markdown(f"- تعداد درختان: **{T['N']}**")
            st.markdown(f"- نزدیک‌ترین: **{T['dmin']:.0f} m** | "
                        f"دورترین: **{T['dmax']:.0f} m**")
            st.markdown(f"- بیشینه |خطا|: **{np.max(np.abs(e)):.3f}°**")
            st.markdown(f"- RMS: **{np.sqrt(np.mean(e**2)):.3f}°**")
            st.markdown(f"- 95%: **{p95(e):.3f}°**")
            st.markdown(f"- نقض ±{tol:.1f}°: **{100*np.mean(np.abs(e)>tol):.1f}%**")
            st.markdown(f"- بدترین آزیموت: **{res['worst_az']:.2f}°**")

            st.markdown("**خروجی خام JSON:**")
            st.json({
                "vor_xy": list(cfg["vor_xy"]),
                "amp_model": amp_model,
                "N_trees": int(T["N"]),
                "dmin": T["dmin"], "dmax": T["dmax"],
                "R_air": R_air, "h_air": h_air,
                "az_step": az_step, "tol": tol,
                "f0_MHz": f0_MHz, "phi_ref_deg": phi_ref_deg,
                "n_patches": len(patches),
                "max_abs_err_deg": float(np.max(np.abs(e))),
                "rms_err_deg": float(np.sqrt(np.mean(e**2))),
                "p95_err_deg": p95(e),
                "worst_az_deg": res["worst_az"],
                "frac_outside_tol": float(np.mean(np.abs(e) > tol)),
                "td_max_diff_deg": (
                    float(np.max(np.abs(res["eps_td"] - e)))
                    if td_check and res["eps_td"] is not None else None),
            })

    else:
        st.info(
            "👈 پارامترهای VOR، قطعات جنگلی و هواپیما را در نوار کناری تنظیم "
            "کنید و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته مهم:** این نسخه **نسخه ایده‌آل** است — REF فرض می‌شود "
            "بدون اغتشاش (فاز صفر) و تنها VAR تحلیل می‌شود. این همان "
            "رویکرد کد MATLAB مرجع است (vor_forest_eq314_ideal_xy.m).\n\n"
            "**پیشنهاد شروع:** یک پچ دایره‌ای با "
            "`cx=1060, cy=1060, R=1200, density=5` — تا خطای قابل‌توجهی ببینید."
        )


# ##########################################################
# ۴) ساختمان‌ها
# ##########################################################
# ============================================================
# توابع کمکی
# ============================================================
def build_buildings(cfg, groups):
    """تولید لیست ساختمان‌ها از همه گروه‌ها."""
    rng = np.random.default_rng(int(cfg["seed"]))
    x = []; y = []; W = []; D = []; H = []; beta = []; gid = []

    for g_idx, G in enumerate(groups, start=1):
        if G["type"] == "single":
            x.append(G["cx"]); y.append(G["cy"])
            W.append(G["W"]); D.append(G["D"])
            H.append(G["H"]); beta.append(G["beta"])
            gid.append(g_idx)

        elif G["type"] == "grid":
            b = np.deg2rad(G["beta"])
            e1 = np.array([np.sin(b), np.cos(b)])
            e2 = np.array([np.cos(b), -np.sin(b)])
            xs = []; ys = []; Ws = []; Ds = []; Hs = []; bs = []
            for i in range(G["rows"]):
                for j in range(G["cols"]):
                    off_w = (j - (G["cols"] - 1) / 2) * (G["W"] + G["gapW"])
                    off_d = (i - (G["rows"] - 1) / 2) * (G["D"] + G["gapD"])
                    off = off_w * e1 + off_d * e2
                    xs.append(G["cx"] + off[0])
                    ys.append(G["cy"] + off[1])
                    Ws.append(G["W"] * (1 + G["jit"] * (2 * rng.random() - 1)))
                    Ds.append(G["D"] * (1 + G["jit"] * (2 * rng.random() - 1)))
                    Hs.append(G["Hmin"] + (G["Hmax"] - G["Hmin"]) * rng.random())
                    bs.append(G["beta"])
            x.extend(xs); y.extend(ys); W.extend(Ws); D.extend(Ds)
            H.extend(Hs); beta.extend(bs)
            gid.extend([g_idx] * len(xs))

        elif G["type"] == "random":
            nb = int(round(G["density"] * np.pi * G["R"] ** 2 / 1e4))
            r = G["R"] * np.sqrt(rng.random(nb))
            ang = 2 * np.pi * rng.random(nb)
            xs = G["cx"] + r * np.cos(ang)
            ys = G["cy"] + r * np.sin(ang)
            Ws = G["Wmin"] + (G["Wmax"] - G["Wmin"]) * rng.random(nb)
            Ds = G["Dmin"] + (G["Dmax"] - G["Dmin"]) * rng.random(nb)
            Hs = G["Hmin"] + (G["Hmax"] - G["Hmin"]) * rng.random(nb)
            if G["beta"] < 0:
                bs = 180 * rng.random(nb)
            else:
                bs = np.full(nb, G["beta"])
            x.extend(xs); y.extend(ys); W.extend(Ws); D.extend(Ds)
            H.extend(Hs); beta.extend(bs)
            gid.extend([g_idx] * nb)

    return dict(
        x=np.array(x), y=np.array(y),
        W=np.array(W), D=np.array(D), H=np.array(H),
        beta=np.array(beta), gid=np.array(gid),
        N=len(x))


def bld_build_facets(cfg, B):
    """ساخت فاست‌های دیواری رو به VOR."""
    c = 299792458.0
    lam = c / (cfg["f0_MHz"] * 1e6)
    S = np.array([0.0, 0.0, cfg["h_ant"]])
    df = cfg["df"]

    P_all = []; N_all = []; A_all = []; Ui_all = []; D1_all = []

    for i in range(B["N"]):
        b = np.deg2rad(B["beta"][i])
        e1 = np.array([np.sin(b), np.cos(b)])
        e2 = np.array([np.cos(b), -np.sin(b)])
        C = np.array([B["x"][i], B["y"][i]])
        W = B["W"][i]; D = B["D"][i]; H = B["H"][i]

        walls = [
            (e1, W / 2, e2, D),        # +e1 face
            (-e1, W / 2, e2, D),       # -e1 face
            (e2, D / 2, e1, W),        # +e2 face
            (-e2, D / 2, e1, W),       # -e2 face
        ]

        for nrm2, off, tng2, Lh in walls:
            nh = max(1, int(np.ceil(Lh / df)))
            nv = max(1, int(np.ceil(H / df)))
            u = (np.arange(nh) + 0.5) / nh * Lh - Lh / 2
            v = (np.arange(nv) + 0.5) / nv * H
            U, Vv = np.meshgrid(u, v)
            xy = (C[None, :]
                  + off * nrm2[None, :]
                  + U.ravel()[:, None] * tng2[None, :])
            z = Vv.ravel()
            Pw = np.column_stack([xy, z])
            Nw = np.tile(np.array([nrm2[0], nrm2[1], 0.0]),
                         (len(U.ravel()), 1))
            Aw = np.full(len(U.ravel()),
                         (Lh / nh) * (H / nv))

            vv = S[None, :] - Pw
            d1 = np.linalg.norm(vv, axis=1)
            ui = vv / d1[:, None]
            keep = np.sum(Nw * ui, axis=1) > 0
            P_all.append(Pw[keep])
            N_all.append(Nw[keep])
            A_all.append(Aw[keep])
            Ui_all.append(ui[keep])
            D1_all.append(d1[keep])

    if not P_all:
        return None

    P = np.vstack(P_all)
    n = np.vstack(N_all)
    area = np.concatenate(A_all)
    ui = np.vstack(Ui_all)
    d1 = np.concatenate(D1_all)

    return dict(P=P, n=n, area=area, ui=ui, d1=d1, S=S, N=len(P))


def building_scatterers(cfg, F, Ax, Ay):
    """محاسبه a، θ، φ برای یک موقعیت هواپیما."""
    c = 299792458.0
    lam = c / (cfg["f0_MHz"] * 1e6)
    phi_ref = np.deg2rad(cfg["phi_ref_deg"])
    A = np.array([Ax, Ay, cfg["h_air"]])

    if F is None or F["N"] == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    v = A[None, :] - F["P"]
    d2 = np.linalg.norm(v, axis=1)
    us = v / d2[:, None]
    ok = np.sum(F["n"] * us, axis=1) > 0

    if not np.any(ok):
        return np.zeros(0), np.zeros(0), np.zeros(0)

    f = np.sum(F["n"] * (F["ui"] + us), axis=1) / 2
    dd = np.linalg.norm(A - F["S"])

    a = (cfg["gamma_w"] * F["area"] * f / lam
         * dd / (F["d1"] * d2))
    theta = -2 * np.pi * ((F["d1"] + d2) - dd) / lam + phi_ref + np.pi / 2
    azt = np.arctan2(F["P"][:, 0], F["P"][:, 1])

    return a[ok], theta[ok], azt[ok]


def bld_compute_eq314(a, theta, azt, pa):
    """Eq. (3-14)."""
    if len(a) == 0:
        return 0.0, 0.0
    phin = azt - pa
    num = -np.sum(a * np.cos(theta) * np.sin(phin))
    den = 1.0 + np.sum(a * np.cos(theta) * np.cos(phin))
    eps = float(np.rad2deg(np.arctan2(num, den)))
    sum_a = float(np.abs(np.sum(a * np.exp(1j * theta))))
    return eps, sum_a


def bld_run_simulation(cfg, F, progress_cb=None):
    """اسکن اصلی + zoom."""
    az = np.arange(0, 360, cfg["az_step"])
    pa = np.deg2rad(az)
    N = len(az)

    eps_314 = np.zeros(N)
    sum_a = np.zeros(N)

    for k in range(N):
        a, theta, azt = building_scatterers(
            cfg, F, cfg["R_air"] * np.sin(pa[k]),
            cfg["R_air"] * np.cos(pa[k]))
        eps_314[k], sum_a[k] = bld_compute_eq314(a, theta, azt, pa[k])
        if progress_cb and (k % max(1, N // 20) == 0 or k == N - 1):
            progress_cb(0.75 * k / N)

    # --- zoom ---
    ip = int(np.argmax(np.abs(eps_314)))
    az_z = (az[ip] - 0.5) + np.arange(0, 1.0 + 0.005, 0.005)
    paz = np.deg2rad(az_z)
    eps_z = np.zeros(len(az_z))
    for k in range(len(az_z)):
        a, theta, azt = building_scatterers(
            cfg, F, cfg["R_air"] * np.sin(paz[k]),
            cfg["R_air"] * np.cos(paz[k]))
        eps_z[k], _ = bld_compute_eq314(a, theta, azt, paz[k])
        if progress_cb and k % max(1, len(az_z) // 10) == 0:
            progress_cb(0.75 + 0.2 * k / len(az_z))

    if progress_cb:
        progress_cb(1.0)

    return dict(
        az=az, eps_314=eps_314, sum_a=sum_a,
        az_measured=az - eps_314, az_ideal=az,
        eps_ideal=np.zeros_like(az),
        az_z=az_z, eps_z=eps_z,
        worst_az=float(az[ip]))


def page_buildings():
    st.title("🏢 خطای CVOR ناشی از ساختمان‌ها (Eq. 3-14)")
    st.caption(
        "فاست‌بندی دیوارها با مدل فیزیک-نوری + Eq. (3-14) با REF ایده‌آل — "
        "تبدیل‌شده از MATLAB"
    )

    # --- نوار کناری ---
    with st.sidebar:
        st.header("📡 VOR")
        vor_x = st.number_input("VOR x East [m]", value=0.0, step=100.0)
        vor_y = st.number_input("VOR y North [m]", value=0.0, step=100.0)
        h_ant = st.number_input("ارتفاع آنتن [m]", value=5.0, step=1.0, min_value=0.5)

        st.header("🔧 مدل دیوار")
        gamma_w = st.slider("|Γ| (ضریب بازتاب دیوار)", 0.0, 1.0, 0.5, 0.01)
        df = st.number_input("اندازه فاست df [m]",
                              value=1.0, step=0.1, min_value=0.2,
                              help="پیشنهاد: df ≤ λ/2 ≈ 1.3 m در 112 MHz")

        st.header("✈️ هواپیما")
        R_air = st.number_input("شعاع پرواز [m]",
                                value=10000.0, step=500.0, min_value=500.0)
        h_air = st.number_input("ارتفاع هواپیما [m]",
                                value=1000.0, step=50.0, min_value=50.0)
        az_step = st.number_input("گام آزیموت [deg]",
                                  value=0.5, step=0.1, min_value=0.05)
        tol = st.number_input("تلورانس [deg]",
                              value=1.0, step=0.5, min_value=0.1)

        st.header("📻 سیستم")
        f0_MHz = st.number_input("فرکانس [MHz]", value=112.0, step=0.5,
                                  min_value=108.0, max_value=118.0)
        phi_ref_deg = st.number_input("φ_ref [deg]", value=180.0, step=10.0)
        seed = st.number_input("بذر تصادفی", value=1, step=1, min_value=0)

        st.header("🏢 گروه‌های ساختمان")
        st.caption("هر ردیف یک گروه. جزئیات هر نوع متفاوت است.")

        # --- انتخاب نوع گروه ---
        default_groups = pd.DataFrame([
            {"name": "B1", "type": "single",
             "cx": 353.55, "cy": 353.55,
             "W": 30.0, "D": 20.0, "H": 15.0, "beta": 0.0,
             "rows": np.nan, "cols": np.nan,
             "Hmin": np.nan, "Hmax": np.nan,
             "gapW": np.nan, "gapD": np.nan, "jit": np.nan,
             "R": np.nan, "density": np.nan,
             "Wmin": np.nan, "Wmax": np.nan,
             "Dmin": np.nan, "Dmax": np.nan},
        ])

        groups_edited = st.data_editor(
            default_groups, num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("نام"),
                "type": st.column_config.SelectboxColumn(
                    "نوع", options=["single", "grid", "random"], required=True),
                "cx": st.column_config.NumberColumn("cx [m]"),
                "cy": st.column_config.NumberColumn("cy [m]"),
                "W": st.column_config.NumberColumn("W [m]"),
                "D": st.column_config.NumberColumn("D [m]"),
                "H": st.column_config.NumberColumn("H [m] (single)"),
                "beta": st.column_config.NumberColumn("β [deg]"),
                "rows": st.column_config.NumberColumn("rows (grid)"),
                "cols": st.column_config.NumberColumn("cols (grid)"),
                "Hmin": st.column_config.NumberColumn("Hmin (grid/random)"),
                "Hmax": st.column_config.NumberColumn("Hmax (grid/random)"),
                "gapW": st.column_config.NumberColumn("gapW (grid) [m]"),
                "gapD": st.column_config.NumberColumn("gapD (grid) [m]"),
                "jit": st.column_config.NumberColumn("jit (grid) [0-1]"),
                "R": st.column_config.NumberColumn("R (random) [m]"),
                "density": st.column_config.NumberColumn("density (random) [b/ha]"),
                "Wmin": st.column_config.NumberColumn("Wmin (random) [m]"),
                "Wmax": st.column_config.NumberColumn("Wmax (random) [m]"),
                "Dmin": st.column_config.NumberColumn("Dmin (random) [m]"),
                "Dmax": st.column_config.NumberColumn("Dmax (random) [m]"),
            }, key="building_groups")

        st.caption(
            "💡 **single**: cx,cy,W,D,H,beta | "
            "**grid**: cx,cy,W,D,Hmin,Hmax,gapW,gapD,jit,beta,rows,cols | "
            "**random**: cx,cy,R,density,Wmin,Wmax,Dmin,Dmax,Hmin,Hmax,beta(-1=random)")

        st.markdown("---")
        st.caption("⚠️ df کوچک = فاست‌های زیاد = کند. df=1 m در 112 MHz خوب است.")
        run = st.button("🚀 اجرای شبیه‌سازی",
                        use_container_width=True, type="primary")


    # --- پردازش و اجرا ---
    if run:
        # --- تبدیل جدول به لیست گروه ---
        groups = []
        for _, row in groups_edited.iterrows():
            t = str(row["type"])
            base = dict(type=t, name=str(row["name"]) if not pd.isna(row["name"]) else "G")

            if t == "single":
                vals = [row["cx"], row["cy"], row["W"], row["D"], row["H"], row["beta"]]
                if any(pd.isna(v) for v in vals):
                    continue
                base.update(cx=float(row["cx"]), cy=float(row["cy"]),
                            W=float(row["W"]), D=float(row["D"]),
                            H=float(row["H"]), beta=float(row["beta"]))

            elif t == "grid":
                vals = [row["cx"], row["cy"], row["W"], row["D"],
                        row["Hmin"], row["Hmax"], row["gapW"], row["gapD"],
                        row["jit"], row["beta"], row["rows"], row["cols"]]
                if any(pd.isna(v) for v in vals):
                    continue
                base.update(cx=float(row["cx"]), cy=float(row["cy"]),
                            W=float(row["W"]), D=float(row["D"]),
                            Hmin=float(row["Hmin"]), Hmax=float(row["Hmax"]),
                            gapW=float(row["gapW"]), gapD=float(row["gapD"]),
                            jit=float(row["jit"]), beta=float(row["beta"]),
                            rows=int(row["rows"]), cols=int(row["cols"]))

            elif t == "random":
                vals = [row["cx"], row["cy"], row["R"], row["density"],
                        row["Wmin"], row["Wmax"], row["Dmin"], row["Dmax"],
                        row["Hmin"], row["Hmax"], row["beta"]]
                if any(pd.isna(v) for v in vals):
                    continue
                base.update(cx=float(row["cx"]), cy=float(row["cy"]),
                            R=float(row["R"]), density=float(row["density"]),
                            Wmin=float(row["Wmin"]), Wmax=float(row["Wmax"]),
                            Dmin=float(row["Dmin"]), Dmax=float(row["Dmax"]),
                            Hmin=float(row["Hmin"]), Hmax=float(row["Hmax"]),
                            beta=float(row["beta"]))
            else:
                continue
            groups.append(base)

        if not groups:
            st.error("حداقل یک گروه معتبر لازم است.")
            st.stop()

        # --- ساخت cfg ---
        cfg = dict(vor_xy=(vor_x, vor_y), h_ant=h_ant,
                   gamma_w=gamma_w, df=df,
                   R_air=R_air, h_air=h_air,
                   az_step=az_step, tol=tol,
                   f0_MHz=f0_MHz, phi_ref_deg=phi_ref_deg, seed=seed)

        lam = 299792458.0 / (f0_MHz * 1e6)
        if df > lam / 2:
            st.warning(f"⚠️ df={df:.2f} m > λ/2={lam/2:.2f} m — "
                       f"جمع فاست‌ها ممکن است دقیق نباشد.")

        # --- ساخت ساختمان‌ها ---
        with st.spinner("تولید ساختمان‌ها..."):
            B = build_buildings(cfg, groups)

        if B["N"] == 0:
            st.error("هیچ ساختمانی تولید نشد.")
            st.stop()

        st.success(f"🏢 **{B['N']} ساختمان** در {len(groups)} گروه")

        # --- ساخت فاست‌ها ---
        with st.spinner("ساخت فاست‌های دیوار..."):
            F = bld_build_facets(cfg, B)

        if F is None or F["N"] == 0:
            st.error("هیچ فاستی رو به VOR نیست.")
            st.stop()

        st.info(f"🔷 **{F['N']} فاست** رو به VOR | λ = {lam:.3f} m | df = {df:.2f} m")

        if F["N"] > 200000:
            st.warning("⚠️ بیش از ۲۰۰,۰۰۰ فاست — ممکن است کند باشد. df را بزرگ‌تر یا ساختمان‌ها را کم کنید.")

        # --- اجرا ---
        progress_bar = st.progress(0, text="در حال شبیه‌سازی...")

        def upd(frac):
            progress_bar.progress(min(max(frac, 0.0), 1.0),
                                  text=f"در حال شبیه‌سازی... {frac*100:.0f}%")

        t0 = time.time()
        with st.spinner("در حال اجرا..."):
            res = bld_run_simulation(cfg, F, progress_cb=upd)
        progress_bar.empty()
        st.success(f"✅ در {time.time()-t0:.1f} ثانیه انجام شد.")

        # --- آماره‌ها ---
        e = res["eps_314"]
        st.markdown("### 📊 خلاصه آماری")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">بیشینه |خطا|</div>'
                f'<div class="metric-value">{np.max(np.abs(e)):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">RMS</div>'
                f'<div class="metric-value">{np.sqrt(np.mean(e**2)):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">95%</div>'
                f'<div class="metric-value">{p95(e):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">نقض ±{tol:.1f}°</div>'
                f'<div class="metric-value">{100*np.mean(np.abs(e)>tol):.1f}%</div></div>',
                unsafe_allow_html=True)

        st.info(
            f"بدترین آزیموت: **{res['worst_az']:.2f}°** | "
            f"بیشینه \\|Σ فازورها\\| = **{np.max(res['sum_a']):.3f}**"
        )

        if np.max(res["sum_a"]) > 0.5:
            st.warning(
                "⚠️ دامنه مجموع مولتی‌پات > 0.5 — "
                "Eq. (3-14) یک فرمول مرتبه اول است و در این رژیم "
                "باید با احتیاط تفسیر شود.")

        st.divider()

        # ============================================================
        # شکل ۱: نمای بالا
        # ============================================================
        st.subheader("🗺️ شکل ۱ — نمای بالا از ساختمان‌ها")

        fig1 = go.Figure()

        H_max_all = max(float(np.max(B["H"])), 1e-9)
        # رسم هر ساختمان به‌عنوان چندضلعی
        for i in range(B["N"]):
            b = np.deg2rad(B["beta"][i])
            e1 = np.array([np.sin(b), np.cos(b)])
            e2 = np.array([np.cos(b), -np.sin(b)])
            C = np.array([B["x"][i], B["y"][i]])
            W = B["W"][i]; D = B["D"][i]; H = B["H"][i]

            corners = [
                C + 0.5 * W * e1 + 0.5 * D * e2,
                C - 0.5 * W * e1 + 0.5 * D * e2,
                C - 0.5 * W * e1 - 0.5 * D * e2,
                C + 0.5 * W * e1 - 0.5 * D * e2,
                C + 0.5 * W * e1 + 0.5 * D * e2,
            ]
            corners = np.array(corners)
            fig1.add_trace(go.Scatter(
                x=corners[:, 0] + vor_x,
                y=corners[:, 1] + vor_y,
                mode="lines",
                fill="toself",
                fillcolor=f"rgba(100,140,200,{0.3 + 0.6*H/H_max_all})",
                line=dict(color="black", width=1),
                showlegend=False,
                hoverinfo="skip"))

        # VOR
        fig1.add_trace(go.Scatter(
            x=[vor_x], y=[vor_y], mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="red"),
            text=["VOR"], textposition="top center",
            name="VOR"))

        fig1.update_layout(
            template="plotly_dark", height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            xaxis_title="x East [m]", yaxis_title="y North [m]",
            margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig1, use_container_width=True)

        # ============================================================
        # شکل ۲: زاویه و خطا (۴ زیرنمودار)
        # ============================================================
        st.subheader("📈 شکل ۲ — زاویه و خطا")

        fig2 = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                "زاویه اندازه‌گیری‌شده vs آزیموت واقعی",
                "خطای Eq. (3-14) در کل ۰–۳۶۰°",
                f"zoom حول بدترین آزیموت ({res['worst_az']:.2f}°)",
                "هیستوگرام خطا"),
            vertical_spacing=0.08,
            row_heights=[0.25, 0.25, 0.25, 0.25])

        az = res["az"]

        # ردیف 1: زاویه واقعی vs اندازه‌گیری‌شده
        fig2.add_trace(go.Scatter(
            x=az, y=res["az_ideal"], mode="lines", name="ایده‌آل",
            line=dict(color="green", dash="dash", width=1.5)), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=az, y=res["az_measured"], mode="lines", name="با ساختمان",
            line=dict(color="#4fc3f7", width=1.2)), row=1, col=1)

        # ردیف 2: خطا
        fig2.add_trace(go.Scatter(
            x=az, y=res["eps_ideal"], mode="lines", name="ایده‌آل",
            line=dict(color="green", dash="dash", width=1.5)), row=2, col=1)
        fig2.add_trace(go.Scatter(
            x=az, y=e, mode="lines", name="خطا Eq.3-14",
            line=dict(color="#4fc3f7", width=1.2)), row=2, col=1)
        fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot"), row=2, col=1)
        fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot"), row=2, col=1)
        fig2.add_hline(y=0, line=dict(color="white", dash="dot"), row=2, col=1)

        # ردیف 3: zoom
        fig2.add_trace(go.Scatter(
            x=res["az_z"], y=res["eps_z"], mode="lines",
            line=dict(color="#4fc3f7", width=1.2),
            showlegend=False), row=3, col=1)
        fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot"), row=3, col=1)
        fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot"), row=3, col=1)
        fig2.add_hline(y=0, line=dict(color="green", dash="dash"), row=3, col=1)

        # ردیف 4: هیستوگرام
        fig2.add_trace(go.Histogram(
            x=e, nbinsx=40,
            marker_color="rgba(60,90,220,0.75)",
            showlegend=False), row=4, col=1)
        fig2.add_vline(x=tol, line=dict(color="magenta", dash="dot"), row=4, col=1)
        fig2.add_vline(x=-tol, line=dict(color="magenta", dash="dot"), row=4, col=1)

        fig2.update_xaxes(title_text="آزیموت واقعی [deg]", row=1, col=1)
        fig2.update_xaxes(title_text="آزیموت واقعی [deg]", row=2, col=1)
        fig2.update_xaxes(title_text="آزیموت [deg]", row=3, col=1)
        fig2.update_xaxes(title_text="خطای زاویه [deg]", row=4, col=1)
        fig2.update_yaxes(title_text="زاویه اندازه‌گیری‌شده [deg]", row=1, col=1)
        fig2.update_yaxes(title_text="خطا [deg]", row=2, col=1)
        fig2.update_yaxes(title_text="خطا [deg]", row=3, col=1)
        fig2.update_yaxes(title_text="شمارش", row=4, col=1)
        fig2.update_layout(template="plotly_dark", height=1200,
                           legend=dict(orientation="h", y=1.02),
                           margin=dict(l=50, r=20, t=80, b=40))
        st.plotly_chart(fig2, use_container_width=True)

        # ============================================================
        # جدول پایانی
        # ============================================================
        with st.expander("📋 خلاصه پارامترها و نتایج"):
            st.markdown("**پارامترهای سیستم:**")
            st.markdown(f"- VOR: ({vor_x:.0f}, {vor_y:.0f}) | h_ant={h_ant:.1f} m")
            st.markdown(f"- λ = {lam:.3f} m | df = {df:.2f} m "
                        f"({df/lam:.2f}·λ)")
            st.markdown(f"- |Γ| = {gamma_w:.2f} | φ_ref={phi_ref_deg:.1f}°")
            st.markdown(f"- R_air={R_air:.0f} m | h_air={h_air:.0f} m")
            st.markdown(f"- az_step={az_step:.3f}° | tol=±{tol:.1f}° | seed={seed}")
            st.markdown("")
            st.markdown("**گروه‌های ساختمان:**")
            for i, G in enumerate(groups, start=1):
                st.markdown(f"- **{G['name']}** ({G['type']})")
            st.markdown("")
            st.markdown("**نتایج:**")
            st.markdown(f"- ساختمان‌ها: **{B['N']}**")
            st.markdown(f"- فاست‌های رو به VOR: **{F['N']}**")
            st.markdown(f"- بیشینه |خطا|: **{np.max(np.abs(e)):.3f}°** "
                        f"(در {res['worst_az']:.2f}°)")
            st.markdown(f"- RMS: **{np.sqrt(np.mean(e**2)):.3f}°**")
            st.markdown(f"- 95%: **{p95(e):.3f}°**")
            st.markdown(f"- نقض ±{tol:.1f}°: **{100*np.mean(np.abs(e)>tol):.1f}%**")
            st.markdown(f"- بیشینه |Σ فازورها|: **{np.max(res['sum_a']):.3f}**")

            st.markdown("**خروجی خام JSON:**")
            st.json({
                "vor_xy": list(cfg["vor_xy"]),
                "h_ant": h_ant,
                "gamma_w": gamma_w,
                "df_m": df,
                "lambda_m": lam,
                "N_buildings": int(B["N"]),
                "N_facets": int(F["N"]),
                "R_air": R_air, "h_air": h_air,
                "az_step": az_step, "tol": tol,
                "f0_MHz": f0_MHz, "phi_ref_deg": phi_ref_deg,
                "n_groups": len(groups),
                "max_abs_err_deg": float(np.max(np.abs(e))),
                "rms_err_deg": float(np.sqrt(np.mean(e**2))),
                "p95_err_deg": p95(e),
                "worst_az_deg": res["worst_az"],
                "frac_outside_tol": float(np.mean(np.abs(e) > tol)),
                "max_sum_phasors": float(np.max(res["sum_a"])),
            })

    else:
        st.info(
            "👈 پارامترها و گروه‌های ساختمان را در نوار کناری تنظیم کنید و "
            "دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته مهم:** مدل فاست دیوار (physical-optics) **ذاتی معادلات تز نیست** — "
            "مدل `f_n = n·(u_i+u_s)/2` و ضریب `j` (یعنی `+π/2` در فاز) "
            "باید در متن تز **ارجاع و توجیه** شود (مثلاً با ارجاع به "
            "**Physical Optics** یا **Bowman, Senior & Uslenghi** — "
            "بازتاب از سطوح رسانا).\n\n"
            "**پیشنهاد برای شروع:** یک تک‌ساختمان W=30، D=20، H=15، "
            "β=0 در (353,353) — خطای حدود ۰٫۵–۲° انتظار می‌رود."
        )


# ##########################################################
# ۵) مزرعه بادی
# ##########################################################
# ============================================================
# توابع کمکی
# ============================================================
def wrap360(x):
    return np.mod(x, 360)


def make_layout(N, center, spacing):
    """چیدمان خطی حول مرکز."""
    xx = (np.arange(1, N + 1) - (N + 1) / 2) * spacing
    return np.column_stack([center[0] + xx, np.full(N, center[1])])


def build_turbine_elements(cfg, turbineXY):
    """ساخت لیست المان‌های هر توربین."""
    xs = []; ys = []; zs = []
    types = []; Ks = []; tids = []

    for it in range(cfg["numTurbines"]):
        cx, cy = turbineXY[it]

        # دکل (در نصف ارتفاع)
        xs.append(cx); ys.append(cy); zs.append(cfg["towerHeight"] / 2)
        types.append("Tower"); Ks.append(cfg["Kmast"]); tids.append(it + 1)

        # هاب
        if cfg["includeHub"]:
            xs.append(cx); ys.append(cy); zs.append(cfg["hubHeight"])
            types.append("Hub"); Ks.append(cfg["Khub"]); tids.append(it + 1)

        # پره‌ها (در صفحه عمودی x-z)
        if cfg["includeBlades"]:
            for ib in range(cfg["numBlades"]):
                ang = (cfg["rotorAngle0"]
                       + it * cfg["rotorAngleStep"]
                       + ib * 360.0 / cfg["numBlades"])
                xs.append(cx); ys.append(cy)
                zs.append(cfg["hubHeight"] + cfg["rotorRadius"] * np.cos(np.deg2rad(ang)))
                types.append(f"Blade {ib+1}")
                Ks.append(cfg["Kblade"])
                tids.append(it + 1)

    return (np.array(xs), np.array(ys), np.array(zs),
            np.array(types), np.array(Ks), np.array(tids))


def scatter_params(cfg, xs, ys, zs, K, xA, yA, zA):
    """محاسبه a، θ، φ برای همه المان‌ها."""
    xV, yV = cfg["vorXY"]
    zV = cfg["vorZ"]

    Rdirect = np.sqrt((xA - xV) ** 2 + (yA - yV) ** 2 + (zA - zV) ** 2)
    R1 = np.sqrt((xs - xV) ** 2 + (ys - yV) ** 2 + (zs - zV) ** 2)
    R2 = np.sqrt((xA - xs) ** 2 + (yA - ys) ** 2 + (zA - zs) ** 2)
    Rmulti = R1 + R2

    theta = cfg["kWave"] * (Rmulti - Rdirect)

    phiS = np.arctan2(xs - xV, ys - yV)
    phiA = np.arctan2(xA - xV, yA - yV)
    phi = phiS - phiA

    a = K * (Rdirect / np.maximum(Rmulti, 1.0))
    return a, theta, phi


def eq314(a, theta, phi):
    """Eq. (3-14) با atan2."""
    if len(a) == 0:
        return 0.0
    num = -np.sum(a * np.cos(theta) * np.sin(phi))
    den = 1.0 + np.sum(a * np.cos(theta) * np.cos(phi))
    return float(np.rad2deg(np.arctan2(num, den)))


def simulate_vor(cfg, xs, ys, zs, types, K, progress_cb=None):
    """اسکن اصلی + محاسبه دو حالت."""
    az = np.arange(0, 360, cfg["azStep"])
    pa = np.deg2rad(az)
    N = len(az)

    epsMast = np.zeros(N)
    epsAll = np.zeros(N)
    sumMast = np.zeros(N)
    sumAll = np.zeros(N)

    isMast = types == "Tower"

    for ia in range(N):
        xA = cfg["vorXY"][0] + cfg["airRange"] * np.sin(pa[ia])
        yA = cfg["vorXY"][1] + cfg["airRange"] * np.cos(pa[ia])
        zA = cfg["airZ"]

        # همه المان‌ها
        aAll, thAll, phiAll = scatter_params(cfg, xs, ys, zs, K, xA, yA, zA)
        epsAll[ia] = eq314(aAll, thAll, phiAll)
        sumAll[ia] = np.sum(np.abs(aAll))

        # فقط دکل‌ها
        aM, thM, phiM = scatter_params(
            cfg, xs[isMast], ys[isMast], zs[isMast], K[isMast],
            xA, yA, zA)
        epsMast[ia] = eq314(aM, thM, phiM)
        sumMast[ia] = np.sum(np.abs(aM))

        if progress_cb and (ia % max(1, N // 20) == 0 or ia == N - 1):
            progress_cb((ia + 1) / N)

    return dict(
        az=az,
        errorIdeal=np.zeros_like(az),
        bearingIdeal=az,
        errorTowerOnly=epsMast,
        errorAllElements=epsAll,
        bearingTowerOnly=wrap360(az - epsMast),
        bearingAllElements=wrap360(az - epsAll),
        sumAmpTower=sumMast,
        sumAmpAll=sumAll)


def page_windfarm():
    st.title("🌀 شبیه‌ساز تعاملی توربین بادی و CVOR")
    st.caption(
        "دکل + هاب + پره‌ها با ضریب موثر + Eq. (3-14) — "
        "تبدیل‌شده از MATLAB"
    )

    # --- نوار کناری ---
    with st.sidebar:
        st.header("📡 VOR")
        vor_x = st.number_input("VOR x East [m]", value=0.0, step=100.0)
        vor_y = st.number_input("VOR y North [m]", value=0.0, step=100.0)
        vor_z = st.number_input("VOR antenna height [m]",
                                 value=5.0, step=1.0, min_value=0.5)

        st.header("📻 امواج رادیویی")
        f_MHz = st.number_input("فرکانس [MHz]", value=113.8, step=0.5,
                                 min_value=108.0, max_value=118.0)
        c_light = st.number_input("سرعت نور [m/s]",
                                   value=299792458.0, step=1000.0, format="%.0f")
        airRange = st.number_input("فاصله افقی هواپیما [m]",
                                    value=20000.0, step=1000.0, min_value=500.0)
        airZ = st.number_input("ارتفاع هواپیما [m]",
                                value=1000.0, step=50.0, min_value=50.0)
        azStep = st.number_input("گام آزیموت [deg]",
                                  value=0.5, step=0.1, min_value=0.05)
        seed = st.number_input("بذر تصادفی", value=1, step=1, min_value=0)

        lam = c_light / (f_MHz * 1e6)
        kWave = 2 * np.pi / lam
        st.caption(f"λ = {lam:.4f} m | k = {kWave:.4f} rad/m")

        st.header("🌀 چیدمان مزرعه")
        numTurbines = st.number_input("تعداد توربین‌ها", value=9,
                                        step=1, min_value=1, max_value=50)
        farm_x = st.number_input("مرکز مزرعه x East [m]",
                                  value=0.0, step=100.0)
        farm_y = st.number_input("مرکز مزرعه y North [m]",
                                  value=5250.0, step=100.0)
        spacing = st.number_input("فاصله بین توربین‌ها [m]",
                                   value=240.0, step=10.0, min_value=10.0)
        auto_layout = st.checkbox("چیدمان خودکار (ردیف خطی)", value=True)

        if auto_layout:
            turbineXY = make_layout(int(numTurbines), (farm_x, farm_y), spacing)
            st.caption(f"✅ {numTurbines} توربین در یک ردیف، فاصله {spacing:.0f} m")
        else:
            st.caption("مختصات هر توربین را وارد کنید:")
            default_xy = make_layout(int(numTurbines), (farm_x, farm_y), spacing)
            turbine_df = pd.DataFrame({
                "turbine": [f"T{i+1}" for i in range(int(numTurbines))],
                "x": default_xy[:, 0],
                "y": default_xy[:, 1],
            })
            edited = st.data_editor(
                turbine_df, num_rows="fixed", use_container_width=True,
                column_config={
                    "turbine": st.column_config.TextColumn("توربین", disabled=True),
                    "x": st.column_config.NumberColumn("x East [m]"),
                    "y": st.column_config.NumberColumn("y North [m]"),
                }, key="turbine_manual")
            turbineXY = edited[["x", "y"]].values.astype(float)

        st.header("🌀 هندسه توربین")
        towerHeight = st.number_input("ارتفاع برج [m]",
                                        value=100.0, step=10.0, min_value=0.0)
        hubHeight = st.number_input("ارتفاع هاب [m]",
                                      value=100.0, step=10.0, min_value=0.0)
        rotorRadius = st.number_input("شعاع روتور [m]",
                                        value=40.0, step=5.0, min_value=0.0)
        numBlades = st.number_input("تعداد پره", value=3,
                                      step=1, min_value=1, max_value=10)
        rotorAngle0 = st.number_input("زاویه اولیه روتور [deg]",
                                        value=35.0, step=5.0)
        rotorAngleStep = st.number_input("افزایش زاویه بین توربین‌ها [deg]",
                                           value=17.0, step=1.0)

        st.header("🔧 ضرایب پراکندگی موثر")
        Kmast = st.number_input("Kmast (دکل)",
                                 value=0.0045, step=0.0005, format="%.4f")
        Khub = st.number_input("Khub (هاب)",
                                value=0.0010, step=0.0005, format="%.4f")
        Kblade = st.number_input("Kblade (هر پره)",
                                  value=0.0008, step=0.0001, format="%.4f")
        includeHub = st.checkbox("شامل هاب", value=True)
        includeBlades = st.checkbox("شامل پره‌ها", value=True)

        st.header("🎯 تحلیل")
        tol = st.number_input("تلورانس [deg]", value=3.0,
                               step=0.5, min_value=0.1)
        run = st.button("🚀 اجرای شبیه‌سازی",
                        use_container_width=True, type="primary")


    # --- اجرا ---
    if run:
        # ساخت cfg
        cfg = dict(
            vorXY=(vor_x, vor_y), vorZ=vor_z,
            frequencyMHz=f_MHz, c=c_light,
            airRange=airRange, airZ=airZ, azStep=azStep,
            seed=int(seed),
            lambda_=lam, kWave=kWave,
            numTurbines=int(numTurbines),
            farmCenter=(farm_x, farm_y), spacing=spacing,
            autoLayout=auto_layout,
            towerHeight=towerHeight, hubHeight=hubHeight,
            rotorRadius=rotorRadius, numBlades=int(numBlades),
            rotorAngle0=rotorAngle0, rotorAngleStep=rotorAngleStep,
            Kmast=Kmast, Khub=Khub, Kblade=Kblade,
            includeHub=includeHub, includeBlades=includeBlades,
            tolerance=tol,
        )

        # ساخت المان‌ها
        with st.spinner("ساخت المان‌های توربین..."):
            xs, ys, zs, types, Ks, tids = build_turbine_elements(cfg, turbineXY)

        if len(xs) == 0:
            st.error("هیچ المانی ساخته نشد.")
            st.stop()

        n_tower = int(np.sum(types == "Tower"))
        n_hub = int(np.sum(types == "Hub"))
        n_blade = int(np.sum(np.char.startswith(types, "Blade")))
        st.success(
            f"✅ {len(xs)} المان در {numTurbines} توربین: "
            f"**{n_tower}** دکل، **{n_hub}** هاب، **{n_blade}** پره"
        )

        # اجرا
        progress_bar = st.progress(0, text="در حال شبیه‌سازی...")

        def upd(frac):
            progress_bar.progress(min(max(float(frac), 0.0), 1.0),
                                  text=f"در حال شبیه‌سازی... {frac*100:.0f}%")

        t0 = time.time()
        with st.spinner("در حال اجرا..."):
            res = simulate_vor(cfg, xs, ys, zs, types, Ks, progress_cb=upd)
        progress_bar.empty()
        st.success(f"✅ در {time.time()-t0:.1f} ثانیه انجام شد.")

        # --- آماره ---
        st.markdown("### 📊 خلاصه آماری")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">بیشینه |خطا| — فقط دکل</div>'
                f'<div class="metric-value">{np.max(np.abs(res["errorTowerOnly"])):.3f}°</div>'
                f'<div style="color:#9aa7c7;font-size:12px">'
                f'RMS {np.sqrt(np.mean(res["errorTowerOnly"]**2)):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">بیشینه |خطا| — همه المان‌ها</div>'
                f'<div class="metric-value">{np.max(np.abs(res["errorAllElements"])):.3f}°</div>'
                f'<div style="color:#9aa7c7;font-size:12px">'
                f'RMS {np.sqrt(np.mean(res["errorAllElements"]**2)):.3f}°</div></div>',
                unsafe_allow_html=True)
        with c3:
            frac = 100 * np.mean(np.abs(res["errorAllElements"]) > tol)
            color = "#4caf50" if frac < 10 else "#ff9800" if frac < 40 else "#f44336"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">نقض ±{tol:.1f}°</div>'
                f'<div class="metric-value" style="color:{color}">{frac:.1f}%</div></div>',
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">بیشینه Σ|a|</div>'
                f'<div class="metric-value">{np.max(res["sumAmpAll"]):.4f}</div></div>',
                unsafe_allow_html=True)

        st.info(
            f"λ = **{lam:.4f} m** | k = **{kWave:.4f} rad/m** | "
            f"{numTurbines} توربین | spacing = **{spacing:.0f} m**"
        )

        if np.max(res["sumAmpAll"]) > 0.5:
            st.warning(
                "⚠️ مجموع دامنه موثر پراکندگی > 0.5 — "
                "Eq. (3-14) یک فرمول مرتبه اول است و در این رژیم با احتیاط تفسیر شود.")

        st.divider()

        # ============================================================
        # شکل ۱: هندسه (نمای بالا)
        # ============================================================
        st.subheader("🗺️ شکل ۱ — هندسه: VOR، مسیر هواپیما، توربین‌ها")

        fig1 = go.Figure()
        # VOR
        fig1.add_trace(go.Scatter(
            x=[vor_x], y=[vor_y], mode="markers+text",
            marker=dict(symbol="triangle-up", size=16, color="red"),
            text=["VOR"], textposition="top center", name="VOR"))
        # دایره مسیر هواپیما
        th = np.linspace(0, 2 * np.pi, 400)
        fig1.add_trace(go.Scatter(
            x=vor_x + airRange * np.cos(th),
            y=vor_y + airRange * np.sin(th),
            mode="lines", line=dict(color="white", dash="dash", width=1),
            name=f"مسیر هواپیما (r={airRange/1000:.0f} km)"))
        # توربین‌ها
        fig1.add_trace(go.Scatter(
            x=turbineXY[:, 0], y=turbineXY[:, 1],
            mode="markers+text",
            marker=dict(symbol="star", size=16, color="#ffb74d"),
            text=[f"T{i+1}" for i in range(numTurbines)],
            textposition="top center", name="توربین‌ها"))

        fig1.update_layout(
            template="plotly_dark", height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            xaxis_title="x East [m]", yaxis_title="y North [m]",
            margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig1, use_container_width=True)

        # ============================================================
        # شکل ۲: زاویه و خطا
        # ============================================================
        st.subheader("📈 شکل ۲ — زاویه اندازه‌گیری‌شده و خطا")

        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=("زاویه اندازه‌گیری‌شده vs آزیموت واقعی",
                            "خطای Eq. (3-14)"),
            vertical_spacing=0.12)

        az = res["az"]

        # ردیف ۱: زاویه
        fig2.add_trace(go.Scatter(
            x=az, y=res["bearingIdeal"], mode="lines",
            name="ایده‌آل (خط‌چین)", line=dict(color="white", dash="dash", width=1.3)),
            row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=az, y=res["bearingAllElements"], mode="lines",
            name="با مزرعه بادی", line=dict(color="#4fc3f7", width=1.2)),
            row=1, col=1)

        # ردیف ۲: خطا
        fig2.add_trace(go.Scatter(
            x=az, y=res["errorIdeal"], mode="lines",
            name="ایده‌آل (خط‌چین)", line=dict(color="white", dash="dash", width=1.3),
            showlegend=False), row=2, col=1)
        fig2.add_trace(go.Scatter(
            x=az, y=res["errorAllElements"], mode="lines",
            name="خطای همه المان‌ها", line=dict(color="#4fc3f7", width=1.2),
            showlegend=False), row=2, col=1)
        fig2.add_hline(y=tol, line=dict(color="magenta", dash="dot"),
                       row=2, col=1)
        fig2.add_hline(y=-tol, line=dict(color="magenta", dash="dot"),
                       row=2, col=1)
        fig2.add_hline(y=0, line=dict(color="white", dash="dot"), row=2, col=1)

        fig2.update_xaxes(title_text="آزیموت هواپیما [deg]", row=1, col=1)
        fig2.update_xaxes(title_text="آزیموت هواپیما [deg]", row=2, col=1)
        fig2.update_yaxes(title_text="زاویه اندازه‌گیری [deg]", row=1, col=1)
        fig2.update_yaxes(title_text="خطا [deg]", row=2, col=1)
        fig2.update_layout(template="plotly_dark", height=750,
                           legend=dict(orientation="h", y=1.05),
                           margin=dict(l=50, r=20, t=80, b=40))
        st.plotly_chart(fig2, use_container_width=True)

        # ============================================================
        # شکل ۳: مقایسه فقط دکل vs همه المان‌ها
        # ============================================================
        st.subheader("📊 شکل ۳ — مقایسه: فقط دکل vs همه المان‌ها")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=az, y=res["errorIdeal"], mode="lines",
            name="ایده‌آل", line=dict(color="white", dash="dash", width=1.3)))
        fig3.add_trace(go.Scatter(
            x=az, y=res["errorTowerOnly"], mode="lines",
            name="فقط دکل", line=dict(color="#ef5350", width=1.2)))
        fig3.add_trace(go.Scatter(
            x=az, y=res["errorAllElements"], mode="lines",
            name="همه المان‌ها", line=dict(color="#4fc3f7", width=1.2)))
        fig3.add_hline(y=tol, line=dict(color="white", dash="dot", width=1))
        fig3.add_hline(y=-tol, line=dict(color="white", dash="dot", width=1))
        fig3.add_hline(y=0, line=dict(color="white", dash="dot", width=1))

        fig3.update_layout(
            template="plotly_dark", height=460,
            xaxis_title="آزیموت هواپیما [deg]",
            yaxis_title="خطا [deg]",
            xaxis_range=[0, 360],
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=50, r=20, t=40, b=40))
        st.plotly_chart(fig3, use_container_width=True)

        # ============================================================
        # شکل ۴: هندسه سه‌بعدی (ساده‌شده)
        # ============================================================
        st.subheader("🏗️ شکل ۴ — هندسه سه‌بعدی توربین‌ها")

        fig4 = go.Figure()
        # VOR
        fig4.add_trace(go.Scatter3d(
            x=[vor_x], y=[vor_y], z=[vor_z],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="red"),
            name="VOR"))
        # برای هر توربین: دکل + هاب + پره‌ها
        for it in range(numTurbines):
            cx, cy = turbineXY[it]
            # دکل
            fig4.add_trace(go.Scatter3d(
                x=[cx, cx], y=[cy, cy], z=[0, towerHeight],
                mode="lines", line=dict(color="white", width=4),
                showlegend=False))
            # هاب
            if includeHub:
                fig4.add_trace(go.Scatter3d(
                    x=[cx], y=[cy], z=[hubHeight],
                    mode="markers",
                    marker=dict(size=5, color="#4fc3f7"),
                    showlegend=False))
            # پره‌ها
            if includeBlades:
                for ib in range(numBlades):
                    ang = (rotorAngle0
                           + it * rotorAngleStep
                           + ib * 360.0 / numBlades)
                    ztip = hubHeight + rotorRadius * np.cos(np.deg2rad(ang))
                    fig4.add_trace(go.Scatter3d(
                        x=[cx, cx], y=[cy, cy], z=[hubHeight, ztip],
                        mode="lines",
                        line=dict(color="#66bb6a", width=3),
                        showlegend=False))

        fig4.update_layout(
            template="plotly_dark", height=650,
            scene=dict(
                xaxis_title="x East [m]",
                yaxis_title="y North [m]",
                zaxis_title="Height [m]",
                aspectmode="data"),
            margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig4, use_container_width=True)

        # ============================================================
        # جدول نهایی
        # ============================================================
        with st.expander("📋 خلاصه پارامترها و نتایج"):
            st.markdown("**پارامترهای سیستم:**")
            st.markdown(f"- VOR: ({vor_x:.0f}, {vor_y:.0f}) | h_ant={vor_z:.1f} m")
            st.markdown(f"- f = {f_MHz:.2f} MHz | λ = {lam:.4f} m | k = {kWave:.4f}")
            st.markdown(f"- R_air = {airRange:.0f} m | h_air = {airZ:.0f} m")
            st.markdown(f"- az_step = {azStep:.3f}° | tol = ±{tol:.1f}°")
            st.markdown("")
            st.markdown("**توربین‌ها:**")
            st.markdown(f"- تعداد: **{numTurbines}**")
            st.markdown(f"- برج: {towerHeight:.0f} m | هاب: {hubHeight:.0f} m | "
                        f"روتور: {rotorRadius:.0f} m")
            st.markdown(f"- پره‌ها: {numBlades} | زاویه اولیه: {rotorAngle0:.0f}° | "
                        f"افزایش: {rotorAngleStep:.0f}°")
            st.markdown(f"- چیدمان: {'خودکار' if auto_layout else 'دستی'}")
            st.markdown("")
            st.markdown("**ضرایب پراکندگی:**")
            st.markdown(f"- Kmast = **{Kmast:.4f}** | Khub = **{Khub:.4f}** | "
                        f"Kblade = **{Kblade:.4f}**")
            st.markdown(f"- شامل هاب: {includeHub} | شامل پره‌ها: {includeBlades}")
            st.markdown("")
            st.markdown("**نتایج:**")
            st.markdown(f"- المان‌های کل: **{len(xs)}** "
                        f"({n_tower} دکل، {n_hub} هاب، {n_blade} پره)")
            st.markdown(f"- بیشینه |خطا| (فقط دکل): "
                        f"**{np.max(np.abs(res['errorTowerOnly'])):.3f}°**")
            st.markdown(f"- بیشینه |خطا| (همه المان‌ها): "
                        f"**{np.max(np.abs(res['errorAllElements'])):.3f}°**")
            st.markdown(f"- RMS (همه): "
                        f"**{np.sqrt(np.mean(res['errorAllElements']**2)):.3f}°**")
            st.markdown(f"- نقض ±{tol:.1f}°: "
                        f"**{100*np.mean(np.abs(res['errorAllElements'])>tol):.1f}%**")
            st.markdown(f"- بیشینه Σ|a|: **{np.max(res['sumAmpAll']):.4f}**")

            st.markdown("**خروجی خام JSON:**")
            st.json({
                "vorXY": [vor_x, vor_y], "vorZ": vor_z,
                "frequencyMHz": f_MHz, "lambda_m": lam,
                "airRange": airRange, "airZ": airZ,
                "azStep": azStep, "tol": tol,
                "numTurbines": int(numTurbines),
                "n_elements": int(len(xs)),
                "n_tower": n_tower, "n_hub": n_hub, "n_blade": n_blade,
                "towerHeight": towerHeight, "hubHeight": hubHeight,
                "rotorRadius": rotorRadius, "numBlades": int(numBlades),
                "Kmast": Kmast, "Khub": Khub, "Kblade": Kblade,
                "includeHub": includeHub, "includeBlades": includeBlades,
                "max_abs_error_tower_deg": float(np.max(np.abs(res["errorTowerOnly"]))),
                "max_abs_error_all_deg": float(np.max(np.abs(res["errorAllElements"]))),
                "rms_error_all_deg": float(np.sqrt(np.mean(res["errorAllElements"]**2))),
                "frac_outside_tol": float(np.mean(np.abs(res["errorAllElements"]) > tol)),
                "max_sum_amp": float(np.max(res["sumAmpAll"])),
            })

    else:
        st.info(
            "👈 پارامترهای VOR، مزرعه بادی، هندسه توربین و ضرایب پراکندگی را "
            "در نوار کناری تنظیم کنید و دکمه «اجرای شبیه‌سازی» را بزنید.\n\n"
            "**نکته مهم:** ضرایب `Kmast، Khub، Kblade` **پارامترهای مهندسی موثر** "
            "هستند، نه ثابت‌های جهانی اندازه‌گیری‌شده. در تز باید "
            "**ارجاع و توجیه** شوند و در صورت وجود داده اندازه‌گیری، "
            "**کالیبره** شوند. مدل المان‌ها یک تقریب ساده‌شده نقطه‌ای است — "
            "حل الکترومغناطیسی کامل توربین واقعی نیست.\n\n"
            "**پیشنهاد شروع:** با تنظیمات پیش‌فرض (۹ توربین، فاصله ۲۴۰ m، "
            "K_mast=0.0045) اجرا کنید تا خطای ~۰٫۵–۲° ببینید."
        )


# ============================================================
# انتخاب شبیه‌ساز و اجرای برنامه
# ============================================================
SIMULATORS = {
    "📡 ۱) شبیه‌ساز ایده‌آل CVOR": page_ideal,
    "⛰️ ۲) خطای ناشی از توپوگرافی زمین": page_terrain,
    "🌲 ۳) خطای ناشی از جنگل": page_forest,
    "🏢 ۴) خطای ناشی از ساختمان‌ها": page_buildings,
    "🌀 ۵) خطای ناشی از مزرعهٔ بادی": page_windfarm
}
DESCRIPTIONS = {
    "📡 ۱) شبیه‌ساز ایده‌آل CVOR": "کانال ایده‌آل بدون نویز و مولتی‌پات؛ استخراج زاویه از REF (FM) و VAR (AM) و خطای عددی.",
    "⛰️ ۲) خطای ناشی از توپوگرافی زمین": "تپه/پشته یا DEM، فاست‌های زمین با مدل constant-gamma و Eq. (3-14).",
    "🌲 ۳) خطای ناشی از جنگل": "چند قطعهٔ جنگلی (دایره/مستطیل/چندضلعی) با تراکم دلخواه و Eq. (3-14).",
    "🏢 ۴) خطای ناشی از ساختمان‌ها": "تک‌ساختمان، شبکهٔ منظم یا ساختمان‌های تصادفی؛ فاست‌بندی دیوارها.",
    "🌀 ۵) خطای ناشی از مزرعهٔ بادی": "دکل، هاب و پره‌های توربین‌ها؛ مقایسهٔ «فقط دکل» با «همهٔ المان‌ها»."
}


def _select(name):
    st.session_state["sim_choice"] = name


def welcome_page():
    st.title("📡 شبیه‌سازهای CVOR")
    st.caption("برای شروع، یکی از شبیه‌سازها را انتخاب کنید "
               "(از نوار کناری یا دکمه‌های زیر).")
    names = list(SIMULATORS)
    cols = st.columns(3)
    for i, name in enumerate(names):
        with cols[i % 3]:
            st.markdown(
                f'<div class="sim-card"><b>{name}</b><br>'
                f'<span style="color:#9aa7c7;font-size:13px">'
                f'{DESCRIPTIONS[name]}</span></div>',
                unsafe_allow_html=True)
            st.button("انتخاب", key=f"pick_{i}", on_click=_select,
                      args=(name,), use_container_width=True)


def main():
    st.set_page_config(page_title="CVOR Simulators", page_icon="📡",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🧭 انتخاب شبیه‌ساز")
        choice = st.selectbox(
            "کدام شبیه‌ساز؟", list(SIMULATORS), index=None,
            placeholder="یک شبیه‌ساز انتخاب کنید...", key="sim_choice")
        if choice:
            st.caption(DESCRIPTIONS[choice])
        st.markdown("---")

    if choice is None:
        welcome_page()
    else:
        SIMULATORS[choice]()


if __name__ == "__main__":
    main()
