import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- 日本語設定 ---
plt.rcParams['font.family'] = 'MS Gothic'

st.set_page_config(layout="wide", page_title="資産運用シミュレーター")

# --- CSS注入：文字の間隔を固定(等幅フォント)し、余計な動きを完全排除 ---
st.markdown("""
    <style>
    .reportview-container .main .block-container { font-family: 'MS Gothic', monospace; }
    .stat-box {
        font-family: 'Courier New', 'MS Gothic', monospace;
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4682b4;
        min-width: 350px;
    }
    .stat-val {
        font-size: 1.8em;
        font-weight: bold;
        text-align: right;
        display: block;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("資産運用シミュレーター（幾何ブラウン運動）")

# --- サイドバー設定 ---
with st.sidebar:
    st.header("パラメータ設定")
    initial_asset = st.number_input("初期投資元本（万円）", value=7500, step=100)
    annual_return_pct = st.slider("期待リターン（％）", 0.0, 30.0, 13.0)
    annual_volatility_pct = st.slider("ボラティリティ（％）", 0.0, 50.0, 22.0)
    annual_withdrawal = st.number_input("年間の取り崩し額（万円）", value=150, step=10)
    years = st.slider("シミュレーション期間（年）", 1, 50, 30)

# --- シミュレーション計算 ---
annual_return = annual_return_pct / 100
annual_volatility = annual_volatility_pct / 100
n_simulations = 10000
n_display_lines = 100

t = np.arange(years + 1)
paths = np.zeros((years + 1, n_simulations))
paths[0] = initial_asset

for s in range(1, years + 1):
    previous_asset = np.maximum(paths[s-1] - annual_withdrawal, 0)
    z = np.random.standard_normal(n_simulations)
    growth = np.exp((annual_return - 0.5 * annual_volatility**2) + annual_volatility * z)
    paths[s] = previous_asset * growth

final_assets = paths[-1]
median_val = np.median(final_assets)
mean_val = np.mean(final_assets)

# --- メイン画面レイアウト ---
col1, col2 = st.columns([2.5, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_display_lines):
        color = 'royalblue' if paths[-1, i] > median_val else 'crimson'
        ax.plot(t, paths[:, i], color=color, alpha=0.3, linewidth=0.8)
    
    ax.plot(t, np.median(paths, axis=1), color='black', linewidth=3, label='中央値')
    ax.plot(t, np.mean(paths, axis=1), color='green', linewidth=3, linestyle='--', label='平均値')
    
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylabel("資産残高（万円）")
    ax.set_ylim(-500, np.percentile(final_assets, 95))
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("分析結果")
    
    # --- 震えを完全に止めるためのHTML/CSS表示 ---
    st.markdown(f"""
        <div class="stat-box">
            <small>最終資産 中央値</small>
            <span class="stat-val">{int(median_val * 10000):>15,} 円</span>
            <br>
            <small>最終資産 平均値</small>
            <span class="stat-val">{int(mean_val * 10000):>15,} 円</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # 統計個数
    st.write(f"**平均値超の数:** {np.sum(final_assets > mean_val):,} 個")
    st.write(f"**中央値超の数:** {np.sum(final_assets > median_val):,} 個")
    st.write(f"**中央値未満の数:** {np.sum(final_assets < median_val):,} 個")
    
    # リスク
    loss_risk = np.mean(final_assets < initial_asset) * 100
    st.error(f"元本毀損リスク: {loss_risk:.1f} ％")