import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="資産運用シミュレーター")

# CSS注入（既存のまま）
st.markdown("""
    <style>
    .stat-box {
        font-family: 'Courier New', 'MS Gothic', monospace;
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4682b4;
        min-width: 350px;
        margin-bottom: 10px;
    }
    .stat-val { font-size: 1.6em; font-weight: bold; text-align: right; display: block; width: 100%; }
    .stat-small-val { font-size: 1.2em; font-weight: bold; text-align: right; display: block; width: 100%; color: #333; }
    </style>
    """, unsafe_allow_html=True)

st.title("資産運用シミュレーター（幾何ブラウン運動）")

# --- サイドバー：パラメータ完全連動（既存のまま） ---
with st.sidebar:
    st.header("パラメータ設定")
    initial_asset = st.number_input("初期投資元本（万円）", value=7500, step=100)

    def update_ret_slider(): st.session_state.ret_slider = st.session_state.ret_num
    def update_ret_num(): st.session_state.ret_num = st.session_state.ret_slider
    annual_return_pct = st.number_input("期待リターン（％）", min_value=0.0, max_value=30.0, value=13.0, step=0.1, format="%.1f", key="ret_num", on_change=update_ret_slider)
    st.slider("（スライダー調整）", 0.0, 30.0, value=13.0, key="ret_slider", on_change=update_ret_num, label_visibility="collapsed")

    def update_vol_slider(): st.session_state.vol_slider = st.session_state.vol_num
    def update_vol_num(): st.session_state.vol_num = st.session_state.vol_slider
    annual_volatility_pct = st.number_input("ボラティリティ（％）", min_value=0.0, max_value=50.0, value=22.0, step=0.1, format="%.1f", key="vol_num", on_change=update_vol_slider)
    st.slider("（スライダー調整）", 0.0, 50.0, value=22.0, key="vol_slider", on_change=update_vol_num, label_visibility="collapsed")

    annual_withdrawal = st.number_input("年間の取り崩し額（万円）", value=150, step=10)

    def update_years_slider(): st.session_state.years_slider = st.session_state.years_num
    def update_years_num(): st.session_state.years_num = st.session_state.years_slider
    years = st.number_input("シミュレーション期間（年）", min_value=1, max_value=50, value=33, step=1, key="years_num", on_change=update_years_slider)
    st.slider("（スライダー調整）", 1, 50, value=33, key="years_slider", on_change=update_years_num, label_visibility="collapsed")

    st.divider()
    scale_type = st.radio("グラフ軸の表記", ["通常表記（線形）", "対数表記（2の累乗）"])
    if 'seed' not in st.session_state: st.session_state.seed = 42
    if st.button("🎲 ランダム（再計算）"): st.session_state.seed = np.random.randint(0, 1000000)

# --- 高速計算セクション ---
np.random.seed(st.session_state.seed)
mu = annual_return_pct / 100
sigma = annual_volatility_pct / 100
n_sims = 10000
dt = 1

# 幾何ブラウン運動の一括計算
# 取り崩しがあるため、資産が途中で0になる可能性を考慮しつつループを最適化
t = np.arange(years + 1)
paths = np.zeros((years + 1, n_sims))
paths[0] = initial_asset

# 乱数を一括生成して高速化
shocks = np.random.standard_normal((years, n_sims))
drift = (mu - 0.5 * sigma**2) * dt
diffusion = sigma * np.sqrt(dt)

for s in range(years):
    # 前年の資産から取り崩しを引き、0以下にならないように調整
    prev_assets = np.maximum(paths[s] - annual_withdrawal, 0)
    # 成長率を一括適用
    paths[s+1] = prev_assets * np.exp(drift + diffusion * shocks[s])

paths_yen = paths * 10000
final_assets_yen = paths_yen[-1]
initial_asset_yen = initial_asset * 10000

# --- 代表値の計算（最頻値の計算も効率化） ---
mean_val = np.mean(final_assets_yen)
median_val = np.median(final_assets_yen)
mode_paths_yen = []
for s in range(len(t)):
    curr = paths_yen[s]
    if np.all(curr == curr[0]): mode_paths_yen.append(curr[0])
    else:
        kde = stats.gaussian_kde(curr)
        xr = np.linspace(0, np.percentile(curr, 99.5) + 1, 200) # サンプル数を調整して高速化
        mode_paths_yen.append(xr[np.argmax(kde(xr))])
mode_final_yen = mode_paths_yen[-1]

risk_probs = np.mean(paths_yen < initial_asset_yen, axis=1) * 100

# --- メイン画面レイアウト（既存のまま） ---
col1, col2 = st.columns([2.2, 1])

with col1:
    fig_ts = go.Figure()
    n_display = 100
    for i in range(n_display):
        color = 'royalblue' if paths_yen[-1, i] > median_val else 'crimson'
        fig_ts.add_trace(go.Scatter(x=t, y=paths_yen[:, i], mode='lines', line=dict(color=color, width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'))

    stats_lines = [(np.mean(paths_yen, axis=1), 'green', 'dash', '平均値'), (np.median(paths_yen, axis=1), 'black', 'solid', '中央値'), (np.array(mode_paths_yen), 'blue', 'dot', '最頻値')]
    for val, color, dash, name in stats_lines:
        fig_ts.add_trace(go.Scatter(x=t, y=val, name=name, line=dict(color=color, width=3, dash=dash), hovertemplate = "想定資産額: %{y:,.0f}円<br>年度: %{x:.2f}年<extra></extra>"))

    fig_ts.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="経過年数", yaxis_title="資産残高（円）", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis=dict(tickformat=",d", exponentformat="none"))
    if scale_type == "対数表記（2の累乗）": fig_ts.update_yaxes(type="log", dtick=1, tickformat=",d")
    else: fig_ts.update_yaxes(range=[-5000000, np.percentile(final_assets_yen, 95)])
    st.plotly_chart(fig_ts, use_container_width=True)

    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        st.write("### 最終資産の確率分布（山の形）")
        clean = final_assets_yen[final_assets_yen > 0]
        if len(clean) > 1:
            kde_final = stats.gaussian_kde(clean)
            x_limit = np.percentile(final_assets_yen, 95)
            x_dist = np.linspace(0, x_limit, 500)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=x_dist, y=kde_final(x_dist), fill='tozeroy', line=dict(color='gray', width=2), hovertemplate = "想定金額: %{x:,.0f}円<extra></extra>"))
            fig_dist.add_vline(x=mean_val, line=dict(color='green', width=2, dash='dash'))
            fig_dist.add_vline(x=median_val, line=dict(color='black', width=2))
            fig_dist.add_vline(x=mode_final_yen, line=dict(color='blue', width=2, dash='dot'))
            fig_dist.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="最終資産残高（円）", yaxis_title="確率密度", xaxis=dict(tickformat=",d", exponentformat="none"))
            st.plotly_chart(fig_dist, use_container_width=True)

    with sub_col2:
        st.write("### 元本割れ確率の推移")
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=t, y=risk_probs, line=dict(color='orange', width=3), fill='tozeroy', hovertemplate = "元本割れ確率: %{y:.1f}％<br>年度: %{x}年<extra></extra>"))
        fig_risk.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="経過年数", yaxis_title="元本割れ確率（％）", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    st.subheader("分析結果")
    st.markdown(f"""<div class="stat-box"><small>最終資産 最高金額</small><span class="stat-small-val" style="color: #1f77b4;">{int(np.max(final_assets_yen)):>15,} 円</span><small>最終資産 平均値</small><span class="stat-val" style="color: green;">{int(mean_val):>15,} 円</span><small>最終資産 中央値</small><span class="stat-val" style="color: black;">{int(median_val):>15,} 円</span><small>最終資産 最頻値</small><span class="stat-val" style="color: blue;">{int(mode_final_yen):>15,} 円</span><small>最終資産 最低金額</small><span class="stat-small-val" style="color: #d62728;">{int(np.min(final_assets_yen)):>15,} 円</span></div>""", unsafe_allow_html=True)
    st.divider()
    st.write(f"**元本毀損の数:** {np.sum(final_assets_yen < initial_asset_yen):,} 個")
    st.write(f"**資産0円（破産）の数:** {np.sum(final_assets_yen <= 0):,} 個")
    st.error(f"最終年 元本毀損リスク: {risk_probs[-1]:.1f} ％")
