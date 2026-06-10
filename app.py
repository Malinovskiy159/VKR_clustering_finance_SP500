import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json, os

st.set_page_config(
    page_title="MarketLens",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  #MainMenu,header,footer{display:none!important}
  .block-container{padding:0!important;max-width:100%!important}
  .stTabs [data-baseweb="tab-list"]{
    background:#fff;border-bottom:1px solid #e6e4de;
    padding:0 20px;gap:0;margin-bottom:0
  }
  .stTabs [data-baseweb="tab"]{
    padding:12px 16px;font-size:20px;font-weight:500;
    color:#6b6a66;border-bottom:2px solid transparent
  }
  .stTabs [aria-selected="true"]{
    color:#18181a!important;
    border-bottom-color:#185FA5!important
  }
  iframe{border:none!important;width:100%!important}
  div[data-testid="stVerticalBlockBorderWrapper"]{padding:0!important}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    files = {
        "exp1": "data/results_exp1.parquet",
        "exp2": "data/results_exp2.parquet",
        "exp3": "data/results_exp3.parquet",
        "exp4": "data/results_exp4.parquet",
        "feat": "data/features.parquet",
        "oos":  "data/validation_oos_2025.parquet",
        "meta": "data/sp500_meta.csv",
    }
    d = {}
    for k, p in files.items():
        if not os.path.exists(p):
            st.error(f"Не найден: {p}")
            st.stop()
        d[k] = pd.read_parquet(p) if p.endswith(".parquet") \
               else pd.read_csv(p, index_col="ticker")

    df = d["exp1"][["cluster_exp1_name", "uncertainty_idx"]].copy()
    df.columns = ["risk", "cui"]

    if "cluster_exp2_name" in d["exp2"].columns:
        df["ret"] = d["exp2"]["cluster_exp2_name"]
    if "cluster_exp3_name" in d["exp3"].columns:
        df["exp3"] = d["exp3"]["cluster_exp3_name"]
    else:
        df["exp3"] = "—"
    if "cluster_exp4_name" in d["exp4"].columns:
        df["mkt"] = d["exp4"]["cluster_exp4_name"]

    df = df.join(d["meta"][["name", "sector"]], how="left")

    for col in ["sigma", "beta", "max_dd", "sharpe",
                "mom_12m", "corr_sp500", "corr_calm"]:
        if col in d["feat"].columns:
            df[col] = d["feat"][col]

    for col in ["sigma", "beta", "max_dd"]:
        if col in d["oos"].columns:
            df[f"{col}_oos"] = d["oos"][col]

    return df


@st.cache_data(ttl=3600)
def load_sparklines(tickers, period="6mo"):
    """Загружает реальные цены для спарклайнов."""
    try:
        import yfinance as yf
        batch = " ".join(tickers[:50])  # берём первые 50
        raw = yf.download(
            batch, period=period,
            auto_adjust=True, progress=False
        )["Close"]
        result = {}
        for tk in tickers[:50]:
            if tk in raw.columns:
                prices = raw[tk].dropna()
                if len(prices) >= 5:
                    # Нормализуем к 0..100 для SVG
                    mn, mx = prices.min(), prices.max()
                    rng = mx - mn if mx != mn else 1
                    pts = prices.values[-20:]  # последние 20 точек
                    norm = [(float(p) - float(mn)) / float(rng) * 100
                            for p in pts]
                    result[tk] = norm
        return result
    except Exception:
        return {}


@st.cache_data
def compute_drift(df):
    """Считает процент акций с опасным дрейфом."""
    drifted = 0
    for ticker, row in df.iterrows():
        for feat in ["sigma", "beta"]:
            tr = row.get(feat)
            oo = row.get(f"{feat}_oos")
            try:
                if not (np.isnan(float(tr)) or
                        np.isnan(float(oo))):
                    if float(oo) > float(tr) * 1.30:
                        drifted += 1
                        break
            except Exception:
                pass
    return round(drifted / max(len(df), 1) * 100, 1)


def safe_float(x, default=0.0):
    try:
        v = float(x)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default



def make_meta_json(df):
    rows = []
    for tk, r in df.iterrows():
        rows.append({
            "tick"      : str(tk),
            "name"      : str(r.get("name", tk) or tk),
            "sector"    : str(r.get("sector","—") or "—"),
            "risk"      : str(r.get("risk","—") or "—"),
            "ret"       : str(r.get("ret","—") or "—"),
            "exp3"      : str(r.get("exp3","—") or "—"),
            "mkt"       : str(r.get("mkt","—") or "—"),
            "cui"       : round(safe_float(r.get("cui"),0.5),1),
            "sigma_tr"  : round(safe_float(r.get("sigma")),3),
            "sigma_oos" : round(safe_float(r.get("sigma_oos")),3),
            "beta_tr"   : round(safe_float(r.get("beta")),3),
            "beta_oos"  : round(safe_float(r.get("beta_oos")),3),
            "sharpe"    : round(safe_float(r.get("sharpe")),3),
            "corr"      : round(safe_float(r.get("corr_sp500")),3),
            "corr_calm" : round(safe_float(r.get("corr_calm")),3),
            "chg"       : round(safe_float(r.get("mom_12m"))*100,1),
            "mdd"       : round(safe_float(r.get("max_dd"))*100,1),
            "win"       : round(safe_float(r.get("win_rate")),3),
            "drift_warns": [],
        })
    import json
    return json.dumps(rows, ensure_ascii=False)

def make_stocks_json(df, sparklines):
    rows = []
    for tk, r in df.iterrows():
        spark = sparklines.get(str(tk), [])
        chg = safe_float(r.get("mom_12m")) * 100
        rows.append({
            "tick":   str(tk),
            "name":   str(r.get("name", tk) or tk)[:40],
            "sector": str(r.get("sector", "—") or "—"),
            "sigma":  round(safe_float(r.get("sigma")), 3),
            "beta":   round(safe_float(r.get("beta")), 3),
            "sharpe": round(safe_float(r.get("sharpe")), 3),
            "chg":    round(chg, 1),
            "cui":    round(safe_float(r.get("cui"), 0.5), 1),
            "risk":   str(r.get("risk", "—") or "—"),
            "ret":    str(r.get("ret", "—") or "—"),
            "exp3":   str(r.get("exp3", "—") or "—"),
            "mkt":    str(r.get("mkt", "—") or "—"),
            "spark":  spark,           # реальные нормализованные цены
            "trend":  "up" if chg >= 0 else "down",
            "corr":   round(safe_float(r.get("corr_sp500")), 3),
        })
    return json.dumps(rows, ensure_ascii=False)


# ── Загрузка ─────────────────────────────────────────────────
df        = load_data()
drift_pct = compute_drift(df)
tickers   = df.index.tolist()
sparklines = load_sparklines(tickers, period="6mo")
STOCKS_JSON = make_stocks_json(df, sparklines)
META_JSON   = make_meta_json(df)
TOTAL       = len(df)


def build_screener(stocks_json, total, drift_pct):
    ok  = drift_pct <= 20
    pill_bg  = "#EAF3DE" if ok else "#FCEBEB"
    pill_col = "#27500A" if ok else "#791F1F"
    pill_txt = f"{'✓' if ok else '⚠'} Дрейф: {drift_pct}%"

    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f5f4f1;--sf:#fff;--br:#e6e4de;--br2:#ccc9be;
  --tx:#18181a;--t2:#6b6a66;--t3:#9e9c97;
  --bl:#185FA5;--bl-bg:#E6F1FB;--bl-t:#0C447C;
  --gr:#3B6D11;--gr-bg:#EAF3DE;--gr-t:#27500A;
  --re:#A32D2D;--re-bg:#FCEBEB;--re-t:#791F1F;
  --am:#854F0B;--am-bg:#FAEEDA;--am-t:#633806;
  --r:7px;--rl:11px
}}
html,body{{height:100%;overflow:hidden;
  font-family:'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--tx);font-size:20px}}

/* Шапка */
.topbar{{
  background:var(--sf);border-bottom:1px solid var(--br);
  height:46px;display:flex;align-items:center;
  padding:0 16px;gap:10px;flex-shrink:0
}}
.logo{{font-size:22px;font-weight:700;letter-spacing:-.4px;white-space:nowrap}}
.logo em{{color:var(--bl);font-style:normal}}
.logo-sub{{font-size:16px;color:var(--t3);font-weight:400;margin-left:4px}}
.drift-pill{{
  margin-left:auto;padding:3px 10px;border-radius:20px;
  font-size:16px;font-weight:700;
  background:{pill_bg};color:{pill_col};white-space:nowrap
}}
.data-date{{font-size:16px;color:var(--t3);white-space:nowrap}}

/* Основной layout */
.wrap{{
  display:grid;grid-template-columns:200px 1fr;
  gap:10px;padding:10px;
  height:calc(100vh - 46px);overflow:hidden
}}
.sb{{
  overflow-y:auto;display:flex;flex-direction:column;
  gap:8px;height:100%
}}
.main{{
  display:flex;flex-direction:column;
  gap:8px;height:100%;overflow:hidden
}}

/* Карточки сайдбара */
.card{{
  background:var(--sf);border:1px solid var(--br);
  border-radius:var(--rl);padding:10px;flex-shrink:0
}}
.ct{{
  font-size:15px;font-weight:700;color:var(--t3);
  text-transform:uppercase;letter-spacing:.5px;margin-bottom:7px
}}

/* Риск-чекбоксы */
.rck{{
  display:flex;align-items:center;gap:6px;padding:5px 7px;
  border-radius:var(--r);cursor:pointer;font-size:18px;
  font-weight:500;margin-bottom:2px;user-select:none
}}
.rck:hover{{background:var(--bg)}}
.rck-box{{
  width:14px;height:14px;border-radius:3px;
  border:2px solid var(--br2);flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  font-size:14px;color:#fff
}}
.rck.g .rck-box{{border-color:var(--gr)}}
.rck.b .rck-box{{border-color:var(--bl)}}
.rck.a .rck-box{{border-color:var(--am)}}
.rck.r .rck-box{{border-color:var(--re)}}
.rck.on.g .rck-box{{background:var(--gr);border-color:var(--gr)}}
.rck.on.b .rck-box{{background:var(--bl);border-color:var(--bl)}}
.rck.on.a .rck-box{{background:var(--am);border-color:var(--am)}}
.rck.on.r .rck-box{{background:var(--re);border-color:var(--re)}}
.rdot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.rcnt{{margin-left:auto;font-size:15px;color:var(--t3)}}

/* Обычные чекбоксы */
.fr{{display:flex;align-items:center;gap:6px;padding:3px 0;cursor:pointer}}
.fr input{{accent-color:var(--bl);width:12px;height:12px;cursor:pointer}}
.fr label{{font-size:18px;flex:1;color:var(--t2);cursor:pointer}}

/* Тулбар */
.tb{{display:flex;align-items:center;gap:7px;flex-wrap:wrap;flex-shrink:0}}
.si{{position:relative;flex:1;min-width:130px}}
.si input{{
  width:100%;padding:7px 10px 7px 27px;border-radius:var(--r);
  border:1px solid var(--br2);background:var(--sf);
  font-size:18px;color:var(--tx);outline:none
}}
.si input:focus{{border-color:var(--bl)}}
.sic{{
  position:absolute;left:8px;top:50%;transform:translateY(-50%);
  color:var(--t3);font-size:20px;pointer-events:none
}}
.sel{{
  padding:7px 7px;border-radius:var(--r);
  border:1px solid var(--br2);background:var(--sf);
  font-size:18px;color:var(--tx)
}}
.tg{{
  display:flex;border:1px solid var(--br2);
  border-radius:var(--r);overflow:hidden
}}
.tgb{{
  padding:6px 9px;font-size:16px;border:none;
  background:transparent;color:var(--t2);cursor:pointer;font-weight:500
}}
.tgb.on{{background:var(--bg);color:var(--tx)}}
.clbl{{font-size:18px;color:var(--t3);white-space:nowrap}}

/* Таблица */
.tw{{
  background:var(--sf);border:1px solid var(--br);
  border-radius:var(--rl);overflow:auto;flex:1;min-height:0
}}
table{{width:100%;border-collapse:collapse}}
th{{
  padding:7px 8px;font-size:15px;font-weight:700;
  color:var(--t3);text-align:left;border-bottom:1px solid var(--br);
  white-space:nowrap;text-transform:uppercase;letter-spacing:.3px;
  position:sticky;top:0;background:var(--sf);z-index:1
}}
th.r{{text-align:right}}
td{{
  padding:8px 8px;font-size:18px;
  border-bottom:1px solid var(--br);vertical-align:middle
}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--bg);cursor:pointer}}
td.r{{text-align:right;font-variant-numeric:tabular-nums}}

/* Ячейка акции */
.tc{{display:flex;align-items:center;gap:7px}}
.tic{{
  width:28px;height:28px;border-radius:var(--r);
  background:var(--bg);border:1px solid var(--br);
  display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:700;color:var(--t2);flex-shrink:0
}}
.tt{{font-weight:600;font-size:18px;line-height:1.2}}
.tn{{font-size:15px;color:var(--t2)}}

/* Бейджи */
.b{{
  display:inline-flex;padding:2px 6px;border-radius:20px;
  font-size:15px;font-weight:600;white-space:nowrap
}}
.bgr{{background:var(--gr-bg);color:var(--gr-t)}}
.bre{{background:var(--re-bg);color:var(--re-t)}}
.bbl{{background:var(--bl-bg);color:var(--bl-t)}}
.bam{{background:var(--am-bg);color:var(--am-t)}}
.bgy{{background:#F1EFE8;color:#444441}}
.pos{{color:var(--gr);font-weight:600}}
.neg{{color:var(--re);font-weight:600}}
.sec-p{{
  padding:2px 6px;border-radius:20px;font-size:15px;
  background:var(--bg);color:var(--t2);border:1px solid var(--br)
}}

/* Тултипы */
.tip-w{{position:relative;display:inline-flex;align-items:center;gap:2px}}
.tip-i{{
  width:12px;height:12px;border-radius:50%;
  background:var(--bg);border:1px solid var(--br2);
  display:inline-flex;align-items:center;justify-content:center;
  font-size:12px;color:var(--t3);cursor:help
}}
.tip-i .tip{{
  display:none;position:absolute;top:16px;left:50%;
  transform:translateX(-50%);background:#1a1a18;color:#fff;
  font-size:16px;padding:6px 9px;border-radius:var(--r);
  width:180px;line-height:1.5;z-index:999;font-weight:400;
  text-transform:none;letter-spacing:0;white-space:normal;pointer-events:none
}}
.tip-i:hover .tip{{display:block}}
</style></head><body>

<!-- Шапка -->
<div class="topbar">
  <div class="logo">
    Market<em>Lens</em>
    <span class="logo-sub">S&P 500</span>
  </div>
  <span class="data-date">Данные: декабрь 2024</span>
  <span class="drift-pill">{pill_txt}</span>
</div>

<div class="wrap">
<!-- ═══ SIDEBAR ═══ -->
<div class="sb">
  <div class="card">
    <div class="ct">Риск (Эксп.1)</div>
    <div class="rck g on" onclick="tgl(this,'risk','Защитные')">
      <div class="rck-box">✓</div>
      <div class="rdot" style="background:var(--gr)"></div>
      Защитные<span class="rcnt" id="rk0">—</span>
    </div>
    <div class="rck b on" onclick="tgl(this,'risk','Рыночные')">
      <div class="rck-box">✓</div>
      <div class="rdot" style="background:var(--bl)"></div>
      Рыночные<span class="rcnt" id="rk1">—</span>
    </div>
    <div class="rck a on" onclick="tgl(this,'risk','Умеренно-агрессивные')">
      <div class="rck-box">✓</div>
      <div class="rdot" style="background:var(--am)"></div>
      <span style="font-size:16px">Умеренно-агрес.</span>
      <span class="rcnt" id="rk2">—</span>
    </div>
    <div class="rck r on" onclick="tgl(this,'risk','Агрессивные')">
      <div class="rck-box">✓</div>
      <div class="rdot" style="background:var(--re)"></div>
      Агрессивные<span class="rcnt" id="rk3">—</span>
    </div>
  </div>

  <div class="card">
    <div class="ct">Доходность (Эксп.2)</div>
    <div class="fr"><input type="checkbox" id="r1" checked onchange="filt()">
      <label for="r1">Опережающие рынок</label></div>
    <div class="fr"><input type="checkbox" id="r2" checked onchange="filt()">
      <label for="r2">Отстающие</label></div>
    <div class="fr"><input type="checkbox" id="r3" checked onchange="filt()">
      <label for="r3">Не определён</label></div>
  </div>

  <div class="card">
    <div class="ct">Сект. независ. (Эксп.3)</div>
    <div class="fr"><input type="checkbox" id="e1" checked onchange="filt()">
      <label for="e1">Секторально незав.</label></div>
    <div class="fr"><input type="checkbox" id="e2" checked onchange="filt()">
      <label for="e2">Умеренно связанные</label></div>
    <div class="fr"><input type="checkbox" id="e3" checked onchange="filt()">
      <label for="e3">Зависимые</label></div>
      <div class="fr"><input type="checkbox" id="e5" checked onchange="filt()">
      <label for="e5">β-усилители</label></div>
    <div class="fr"><input type="checkbox" id="e4" checked onchange="filt()">
      <label for="e4">Не определён</label></div>
  </div>

  <div class="card">
    <div class="ct">Межрыночные (Эксп.4)</div>
    <div class="fr"><input type="checkbox" id="m1" checked onchange="filt()">
      <label for="m1">Независимые</label></div>
    <div class="fr"><input type="checkbox" id="m2" checked onchange="filt()">
      <label for="m2">Умеренно связанные</label></div>
    <div class="fr"><input type="checkbox" id="m3" checked onchange="filt()">
      <label for="m3">Рыночные</label></div>
    <div class="fr"><input type="checkbox" id="m4" checked onchange="filt()">
      <label for="m4">Не определён</label></div>
  </div>

  <div class="card">
    <div class="ct">Надёжность (CUI)</div>
    <div class="fr"><input type="checkbox" id="c1" checked onchange="filt()">
      <label for="c1">Высокая (CUI=0)</label></div>
    <div class="fr"><input type="checkbox" id="c2" checked onchange="filt()">
      <label for="c2">Умеренная (CUI=0.5)</label></div>
    <div class="fr"><input type="checkbox" id="c3" checked onchange="filt()">
      <label for="c3">Низкая (CUI=1)</label></div>
  </div>
</div>

<!-- ═══ MAIN ═══ -->
<div class="main">
  <div class="tb">
    <div class="si">
      <span class="sic">⌕</span>
      <input type="text" placeholder="Поиск по тикеру или названию..."
             id="sq" oninput="filt()">
    </div>
    <select class="sel" onchange="srt(this.value)">
      <option value="">По умолчанию</option>
      <option value="sharpe">Шарп ↓</option>
      <option value="sigma">σ ↓</option>
      <option value="beta">β ↓</option>
      <option value="chg">Доходность ↓</option>
    </select>
    <div class="tg">
      <button class="tgb on" onclick="sv(this,'tick')">Тикер</button>
      <button class="tgb" onclick="sv(this,'name')">Название</button>
    </div>
    <span class="clbl" id="clbl">загрузка...</span>
  </div>

  <div class="tw">
    <table>
      <thead><tr>
        <th style="min-width:150px">Акция</th>
        <th style="width:72px">6 мес.</th>
        <th style="width:110px">Сектор</th>
        <th class="r" style="width:52px">
          <div class="tip-w">σ
            <div class="tip-i">?<div class="tip">Волатильность. σ=0.30 → цена колеблется ±30% в год.</div></div>
          </div>
        </th>
        <th class="r" style="width:46px">
          <div class="tip-w">β
            <div class="tip-i">?<div class="tip">Бета. β=1.5: рынок −10% → акция −15%.</div></div>
          </div>
        </th>
        <th class="r" style="width:58px">
          <div class="tip-w">Шарп
            <div class="tip-i">?<div class="tip">Доходность на единицу риска. Выше — лучше. >1 — хороший результат.</div></div>
          </div>
        </th>
        <th class="r" style="width:60px">
          <div class="tip-w">Изм. 1г
            <div class="tip-i">?<div class="tip">Изменение цены за 12 месяцев (моментум).</div></div>
          </div>
        </th>
        <th style="width:85px">
          <div class="tip-w">Сект. незав.
            <div class="tip-i">?<div class="tip">Независимость от ETF сектора. «Незав.» лучше диверсифицируют портфель.</div></div>
          </div>
        </th>
        <th style="width:90px">
          <div class="tip-w">Надёжность
            <div class="tip-i">?<div class="tip">CUI — согласованность алгоритмов. «Высокая» — все три согласны.</div></div>
          </div>
        </th>
        <th style="width:80px">Доходность</th>
      </tr></thead>
      <tbody id="stb"></tbody>
    </table>
  </div>
</div>
</div>

<script>
const SD = {stocks_json};
const TOTAL = {total};
let vm = 'tick', sk = '';
const F = {{
  risk: new Set(['Защитные','Рыночные','Умеренно-агрессивные','Агрессивные'])
}};

// Счётчики по кластерам
const rk = ['Защитные','Рыночные','Умеренно-агрессивные','Агрессивные'];
const cnts = {{}};
SD.forEach(s => {{ cnts[s.risk] = (cnts[s.risk]||0)+1; }});
['rk0','rk1','rk2','rk3'].forEach((id,i) => {{
  const el=document.getElementById(id);
  if(el) el.textContent = cnts[rk[i]]||0;
}});

// Предикаты фильтров
const CB = {{
  r1: s => s.ret && s.ret.includes('Опережа'),
  r2: s => s.ret && (s.ret.includes('Отстаю')||s.ret.includes('Отстающ')),
  r3: s => !s.ret || s.ret==='—',
  e1: s => s.exp3 && (s.exp3.toLowerCase().includes('незав')||s.exp3.toLowerCase().includes('независ')),
  e2: s => s.exp3 && (s.exp3.toLowerCase().includes('умер')||s.exp3.toLowerCase().includes('связ')),
  e3: s => s.exp3 && (s.exp3.toLowerCase().includes('зав')&&!s.exp3.toLowerCase().includes('незав')&&!s.exp3.toLowerCase().includes('усил')),
  e5: s => s.exp3 && (s.exp3.toLowerCase().includes('усил')||s.exp3.toLowerCase().includes('β')),
  e4: s => !s.exp3 || s.exp3==='—',
  m1: s => s.mkt && s.mkt.includes('Независим'),
  m2: s => s.mkt && (s.mkt.includes('Умер')||s.mkt.includes('связ')),
  m3: s => s.mkt && s.mkt.includes('Рыноч'),
  m4: s => !s.mkt || s.mkt==='—',
  c1: s => parseFloat(s.cui)===0,
  c2: s => parseFloat(s.cui)===0.5,
  c3: s => parseFloat(s.cui)===1||parseFloat(s.cui)===1.0,
}};

// Группы чекбоксов
const CB_GROUPS = {{
  ret : ['r1','r2','r3'],
  exp3: ['e1','e2','e3','e4','e5'],
  mkt : ['m1','m2','m3','m4'],
  cui : ['c1','c2','c3'],
}};

function spk(spark, trend, w=68, h=28) {{
  const col = trend==='up' ? '#3B6D11' : '#A32D2D';
  const fc  = trend==='up' ? '#EAF3DE' : '#FCEBEB';
  let pts;
  if(spark && spark.length >= 2) {{
    const mn=Math.min(...spark), mx=Math.max(...spark), rng=mx-mn||1;
    pts = spark.map((v,i) => {{
      const x = 2 + i*(w-4)/(spark.length-1);
      const y = (h-2) - ((v-mn)/rng)*(h-4);
      return x.toFixed(1)+','+y.toFixed(1);
    }});
  }} else {{
    // fallback — прямая
    pts = trend==='up'
      ? ['2,'+(h-2), w/2+','+(h/2), (w-2)+',2']
      : ['2,2', w/2+','+(h/2), (w-2)+','+(h-2)];
  }}
  const poly = pts.join(' ')+' '+(w-2)+','+(h-2)+' 2,'+(h-2);
  return '<svg width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'">'+
    '<polygon points="'+poly+'" fill="'+fc+'" opacity=".7"/>'+
    '<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+col+
    '" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'+
    '</svg>';
}}

function cuib(c) {{
  const n = parseFloat(c);
  if(n===0)   return '<span class="b bgr">✓ Высокая</span>';
  if(n===0.5) return '<span class="b bam">~ Умеренная</span>';
  return '<span class="b bre">⚠ Низкая</span>';
}}
function retb(r) {{
  if(!r||r==='—') return '<span class="b bgy">—</span>';
  return r.includes('Опережа')
    ? '<span class="b bgr">Опережает</span>'
    : '<span class="b bgy">Отстаёт</span>';
}}
function exp3b(e) {{
  if(!e||e==='—') return '<span class="b bgy">—</span>';
  const el = e.toLowerCase();
  if(el.includes('незав')||el.includes('независ'))
    return '<span class="b bgr">Незав.</span>';
  if(el.includes('умер')||el.includes('связ'))
    return '<span class="b bam">Умерен.</span>';
  if(el.includes('зав')&&!el.includes('незав'))
    return '<span class="b bre">Завис.</span>';
  return '<span class="b bgy">'+e.slice(0,8)+'</span>';
}}

function filt() {{
  const q = document.getElementById('sq').value.toLowerCase();
  let res = SD.filter(s => {{
    if(!F.risk.has(s.risk)) return false;
    if(q&&!s.tick.toLowerCase().includes(q)&&
         !s.name.toLowerCase().includes(q)) return false;
    for(const [grp, ids] of Object.entries(CB_GROUPS)) {{
      const active = ids.filter(id => document.getElementById(id)?.checked);
      if(active.length===0) return false;
      if(!active.some(id => CB[id](s))) return false;
    }}
    return true;
  }});
  if(sk==='sharpe') res.sort((a,b)=>b.sharpe-a.sharpe);
  else if(sk==='sigma') res.sort((a,b)=>b.sigma-a.sigma);
  else if(sk==='beta')  res.sort((a,b)=>b.beta-a.beta);
  else if(sk==='chg')   res.sort((a,b)=>b.chg-a.chg);

  document.getElementById('stb').innerHTML = res.map(s => {{
    const cc=s.chg>=0?'pos':'neg';
    const cs=(s.chg>=0?'+':'')+s.chg.toFixed(1)+'%';
    const lbl = vm==='tick'
      ? '<div class="tt">'+s.tick+'</div><div class="tn">'+(s.name.length>20?s.name.slice(0,20)+'…':s.name)+'</div>'
      : '<div class="tt">'+(s.name.length>20?s.name.slice(0,20)+'…':s.name)+'</div><div class="tn">'+s.tick+'</div>';
    const sec = s.sector
      .replace('Consumer Discretionary','Con.Disc.')
      .replace('Consumer Staples','Con.Staples')
      .replace('Information Technology','IT')
      .replace('Communication Services','Comm.');
    return '<tr>'+
      '<td><div class="tc"><div class="tic">'+s.tick.slice(0,3)+'</div><div>'+lbl+'</div></div></td>'+
      '<td>'+spk(s.spark,s.trend)+'</td>'+
      '<td><span class="sec-p">'+sec+'</span></td>'+
      '<td class="r">'+s.sigma.toFixed(3)+'</td>'+
      '<td class="r">'+s.beta.toFixed(2)+'</td>'+
      '<td class="r '+(s.sharpe>=0?'pos':'neg')+'">'+s.sharpe.toFixed(2)+'</td>'+
      '<td class="r '+cc+'">'+cs+'</td>'+
      '<td>'+exp3b(s.exp3)+'</td>'+
      '<td>'+cuib(s.cui)+'</td>'+
      '<td>'+retb(s.ret)+'</td>'+
      '</tr>';
  }}).join('');
  document.getElementById('clbl').textContent =
    'показано '+res.length+' из '+TOTAL;
}}

function tgl(el,field,val) {{
  if(el.classList.contains('on')) {{
    if(F[field]?.size===1) return;
    F[field]?.delete(val);
    el.classList.remove('on');
    el.querySelector('.rck-box').textContent='';
  }} else {{
    F[field]?.add(val);
    el.classList.add('on');
    el.querySelector('.rck-box').textContent='✓';
  }}
  filt();
}}
function sv(btn,m){{
  document.querySelectorAll('.tgb').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on'); vm=m; filt();
}}
function srt(v){{ sk=v; filt(); }}
filt();
</script>




<script>
(function(){{
  function sendH(){{
    var h=document.documentElement.scrollHeight||document.body.scrollHeight;
    window.parent.postMessage({{type:'streamlit:setFrameHeight',height:h}},'*');
  }}
  window.addEventListener('load',function(){{setTimeout(sendH,300);}});
  setTimeout(sendH,700);
}})();
</script>

</body></html>"""




# ════════════════════════════════════════════════════════════
# СТРАНИЦА 2 — ПАСПОРТ АКЦИИ
# ════════════════════════════════════════════════════════════
def build_passport(stocks_json, meta_json):
    css = """
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#f5f4f1;--sf:#fff;--br:#e6e4de;--br2:#ccc9be;
  --tx:#18181a;--t2:#6b6a66;--t3:#9e9c97;
  --bl:#185FA5;--bl-bg:#E6F1FB;--gr:#3B6D11;--gr-bg:#EAF3DE;--gr-t:#27500A;
  --re:#A32D2D;--re-bg:#FCEBEB;--re-t:#791F1F;
  --am:#854F0B;--am-bg:#FAEEDA;--am-t:#633806;--r:8px;--rl:12px}
html,body{min-height:100vh;overflow-y:auto;margin:0;
  font-family:"Segoe UI",system-ui,sans-serif;
  background:var(--bg);color:var(--tx);font-size:15px}
.b{display:inline-flex;padding:4px 11px;border-radius:20px;font-size:14px;font-weight:600}
.bgr{background:var(--gr-bg);color:var(--gr-t)}.bre{background:var(--re-bg);color:var(--re-t)}
.bbl{background:var(--bl-bg);color:#0C447C}.bam{background:var(--am-bg);color:var(--am-t)}
.bgy{background:#F1EFE8;color:#444441}
.pos{color:var(--gr);font-weight:600}.neg{color:var(--re);font-weight:600}
.wrap{padding:18px;display:flex;flex-direction:column;gap:14px}
.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.tit{font-size:24px;font-weight:700}
.sw{position:relative;flex:1;max-width:420px}
.sw input{width:100%;padding:11px 14px 11px 36px;border-radius:var(--r);
  border:1px solid var(--br2);background:var(--sf);font-size:16px;color:var(--tx);outline:none}
.sw input:focus{border-color:var(--bl)}
.sic{position:absolute;left:12px;top:50%;transform:translateY(-50%);
  color:var(--t3);font-size:17px;pointer-events:none}
.dd{position:absolute;top:46px;left:0;right:0;background:var(--sf);
  border:1px solid var(--br2);border-radius:var(--r);z-index:200;
  max-height:260px;overflow-y:auto;display:none;
  box-shadow:0 4px 16px rgba(0,0,0,.12)}
.di{padding:12px 16px;cursor:pointer;font-size:15px;display:flex;
  align-items:center;gap:12px;border-bottom:1px solid var(--br)}
.di:last-child{border-bottom:none}.di:hover{background:var(--bg)}
.dt{font-weight:700;min-width:50px;font-size:16px}.dn{color:var(--t2);font-size:13px}
.ph{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);padding:20px;display:flex;gap:16px}
.pi{width:60px;height:60px;border-radius:var(--rl);background:var(--bl-bg);
  display:flex;align-items:center;justify-content:center;
  font-size:20px;font-weight:700;color:#0C447C;flex-shrink:0}
.pn{font-size:22px;font-weight:700;margin-bottom:5px}
.ps{font-size:15px;color:var(--t2)}
.brow{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.eg{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);padding:16px}
.en{font-size:12px;color:var(--t3);font-weight:700;text-transform:uppercase;
  letter-spacing:.4px;margin-bottom:5px}
.etit{font-size:17px;font-weight:600;margin-bottom:13px}
.mr{display:flex;justify-content:space-between;align-items:center;
  padding:7px 0;border-bottom:1px solid var(--br)}
.mr:last-child{border-bottom:none}
.mn{font-size:14px;color:var(--t2)}.mv{font-size:15px;font-weight:600}
.ib{background:var(--bg);border-radius:var(--r);padding:11px;
  margin-top:10px;font-size:14px;color:var(--t2);line-height:1.6}
.cui-b{border:1px solid var(--br);border-radius:var(--r);padding:11px 14px;margin-top:11px}
.cui-ok{border-color:var(--gr);background:var(--gr-bg)}
.cui-mid{border-color:#EF9F27;background:var(--am-bg)}
.cui-bad{border-color:var(--re);background:var(--re-bg)}
.cui-t{font-size:14px;font-weight:700;margin-bottom:6px}
.cui-ok .cui-t{color:var(--gr-t)}.cui-mid .cui-t{color:var(--am-t)}.cui-bad .cui-t{color:var(--re-t)}
.cui-d{font-size:13px;color:var(--t2);line-height:1.5}
.empty{display:flex;align-items:center;justify-content:center;
  height:55vh;color:var(--t3);font-size:17px;flex-direction:column;gap:12px}
"""
    js = r"""
var SD_ALL  = JSON.parse(document.getElementById('sd').textContent);
var META_ALL = JSON.parse(document.getElementById('md').textContent);
var IDX = {};
META_ALL.forEach(function(m){ IDX[m.tick] = m; });

document.addEventListener('click', function(e){
  if(!e.target.closest('.sw'))
    document.getElementById('dd').style.display='none';
});

function doSearch(){
  var q = document.getElementById('psq').value.toLowerCase().trim();
  var dd = document.getElementById('dd');
  if(!q){ dd.style.display='none'; return; }
  var res = [];
  for(var i=0;i<SD_ALL.length && res.length<8;i++){
    var s=SD_ALL[i];
    if(s.tick.toLowerCase().indexOf(q)>=0||s.name.toLowerCase().indexOf(q)>=0)
      res.push(s);
  }
  if(!res.length){ dd.style.display='none'; return; }
  var html='';
  for(var i=0;i<res.length;i++){
    html+='<div class="di" onclick="loadTick(this.dataset.tick)" data-tick="'
      +res[i].tick+'">'
      +'<span class="dt">'+res[i].tick+'</span>'
      +'<span class="dn">'+res[i].name.slice(0,32)+'</span>'
      +'</div>';
  }
  dd.innerHTML=html; dd.style.display='block';
}

function loadTick(tick){
  document.getElementById('psq').value=tick;
  document.getElementById('dd').style.display='none';
  var d=IDX[tick];
  if(!d){
    document.getElementById('pcon').innerHTML=
      '<div class="empty"><span>Нет данных для '+tick+'</span></div>';
    return;
  }
  renderPassport(d);
}

function rb(r){
  var m={'Защитные':'bgr','Агрессивные':'bre','Рыночные':'bbl','Умеренно-агрессивные':'bam'};
  return '<span class="b '+(m[r]||'bgy')+'">'+r+'</span>';
}

function e3badge(e){
  var el=(e||'').toLowerCase();
  if(el.indexOf('незав')>=0) return '<span class="b bgr">Незав.</span>';
  if(el.indexOf('умер')>=0||el.indexOf('связ')>=0) return '<span class="b bam">Умерен.</span>';
  if(el.indexOf('усил')>=0||el.indexOf('β')>=0) return '<span class="b bbl">β-усил.</span>';
  return '<span class="b bre">Завис.</span>';
}

function cuiBlock(d){
  var n=parseFloat(d.cui);
  if(n===0) return '<div class="cui-b cui-ok"><div class="cui-t">&#10003; Высокая надёжность</div><div class="cui-d">Все три алгоритма согласны: <b>'+d.risk+'</b>.</div></div>';
  if(n===0.5) return '<div class="cui-b cui-mid"><div class="cui-t">~ Умеренная надёжность</div><div class="cui-d">Два алгоритма согласны с <b>'+d.risk+'</b>, один расходится.</div></div>';
  return '<div class="cui-b cui-bad"><div class="cui-t">&#9888; Низкая надёжность</div><div class="cui-d">Алгоритмы расходятся. Пограничный случай — уменьшите позицию.</div></div>';
}

function renderPassport(d){
  var ml=d.mkt||'—';
  var mkl=ml.indexOf('Незав')>=0?'Независимые':ml.indexOf('Умер')>=0?'Умеренно связ.':'Рыночные';
  var mkc=ml.indexOf('Незав')>=0?'bgr':ml.indexOf('Умер')>=0?'bam':'bre';
  document.getElementById('pcon').innerHTML=
    '<div class="ph">'
    +'<div class="pi">'+d.tick.slice(0,2)+'</div>'
    +'<div><div class="pn">'+d.tick+' &#8212; '+d.name+'</div>'
    +'<div class="ps">'+d.sector+'</div>'
    +'<div class="brow">'+rb(d.risk)+'</div></div></div>'
    +'<div class="eg">'
    +'<div class="card"><div class="en">Эксперимент 1</div><div class="etit">Риск-профиль</div>'
    +'<div style="margin-bottom:10px">'+rb(d.risk)+'</div>'
    +'<div class="mr"><span class="mn">&#963; OOS</span><span class="mv">'+(d.sigma_oos||0).toFixed(3)+'</span></div>'
    +'<div class="mr"><span class="mn">&#946; OOS</span><span class="mv">'+(d.beta_oos||0).toFixed(3)+'</span></div>'
    +'<div class="mr"><span class="mn">Шарп OOS</span><span class="mv '+(d.sharpe>=0?'pos':'neg')+'">'+(d.sharpe||0).toFixed(2)+'</span></div>'
    +'<div class="mr"><span class="mn">Доходность OOS</span><span class="mv '+(d.chg>=0?'pos':'neg')+'">'+(d.chg>=0?'+':'')+d.chg.toFixed(1)+'%</span></div>'
    +cuiBlock(d)+'</div>'
    +'<div class="card"><div class="en">Эксперимент 2</div><div class="etit">Профиль доходности</div>'
    +'<div style="margin-bottom:10px">'+(d.ret&&d.ret.indexOf('Опережа')>=0?'<span class="b bgr">Опережает рынок</span>':'<span class="b bgy">Отстаёт</span>')+'</div>'
    +'<div class="mr"><span class="mn">Sharpe 2024</span><span class="mv">'+(d.sharpe||0).toFixed(2)+'</span></div>'
    +'<div class="mr"><span class="mn">mom_12m 2024</span><span class="mv">'+(d.chg>=0?'+':'')+d.chg.toFixed(1)+'%</span></div>'
    +'<div class="ib">'+(d.ret&&d.ret.indexOf('Опережа')>=0?'Лидер доходности 2024. Моментум сохраняется 6–12 мес.':'Аутсайдер 2024.')+'</div></div>'
    +'<div class="card"><div class="en">Эксперимент 3</div><div class="etit">Секторальная независимость</div>'
    +'<div style="margin-bottom:10px">'+e3badge(d.exp3||'—')+'</div>'
    +'<div class="mr"><span class="mn">Корр. с сектором</span><span class="mv">'+(d.corr||0).toFixed(3)+'</span></div>'
    +'<div class="ib">'+getE3interp(d.exp3||'')+'</div></div>'
    +'<div class="card"><div class="en">Эксперимент 4</div><div class="etit">Межрыночные связи</div>'
    +'<div style="margin-bottom:10px"><span class="b '+mkc+'">'+mkl+'</span></div>'
    +'<div class="mr"><span class="mn">Корр. S&P 500</span><span class="mv">'+(d.corr||0).toFixed(3)+'</span></div>'
    +'<div class="mr"><span class="mn">Liberation Day</span><span class="mv neg">'+(ml.indexOf('Незав')>=0?'&#8722;7.9%':ml.indexOf('Умер')>=0?'&#8722;12.1%':'&#8722;15.5%')+'</span></div>'
    +'</div></div>';
}

function getE3interp(e){
  var el=e.toLowerCase();
  if(el.indexOf('незав')>=0) return 'Независим от ETF сектора. Тарифный шок 2025: +2.4 п.п. vs ETF.';
  if(el.indexOf('усил')>=0||el.indexOf('β')>=0) return 'β-усилитель: beta_sector≈1.42. В кризис: −2.8 п.п. vs ETF.';
  if(el.indexOf('умер')>=0) return 'Умеренная зависимость от сектора.';
  return 'Сильно коррелирует с ETF. При секторальном шоке падает вместе с сектором.';
}
"""
    return (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='UTF-8'>"
        "<style>" + css + "</style></head><body>"
        "<script id='sd' type='application/json'>" + stocks_json + "</script>"
        "<script id='md' type='application/json'>" + meta_json + "</script>"
        "<div class='wrap'>"
        "<div class='top'><span class='tit'>Паспорт акции</span>"
        "<div class='sw'><span class='sic'>&#9906;</span>"
        "<input type='text' id='psq' placeholder='Введите тикер: AAPL, JNJ, NVDA...' oninput='doSearch()' autocomplete='off'>"
        "<div class='dd' id='dd'></div></div></div>"
        "<div id='pcon'><div class='empty'>"
        "<span style='font-size:40px'>&#128196;</span>"
        "<span>Введите тикер в строку поиска</span>"
        "</div></div></div>"
        "<script>" + js + "</script></body></html>"
    )


# ════════════════════════════════════════════════════════════
# СТРАНИЦА 3 — МОЙ ПОРТФЕЛЬ
# ════════════════════════════════════════════════════════════

def build_portfolio(stocks_json):
    css = """
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#f5f4f1;--sf:#fff;--br:#e6e4de;--br2:#ccc9be;
  --tx:#18181a;--t2:#6b6a66;--t3:#9e9c97;
  --bl:#185FA5;--bl-bg:#E6F1FB;--gr:#3B6D11;--gr-bg:#EAF3DE;--gr-t:#27500A;
  --re:#A32D2D;--re-bg:#FCEBEB;--re-t:#791F1F;
  --am:#854F0B;--am-bg:#FAEEDA;--am-t:#633806;--r:8px;--rl:12px}
html,body{min-height:100vh;overflow-y:auto;margin:0;
  font-family:"Segoe UI",system-ui,sans-serif;
  background:var(--bg);color:var(--tx);font-size:15px}
.b{display:inline-flex;padding:4px 11px;border-radius:20px;font-size:14px;font-weight:600}
.bgr{background:var(--gr-bg);color:var(--gr-t)} .bre{background:var(--re-bg);color:var(--re-t)}
.bbl{background:var(--bl-bg);color:#0C447C} .bam{background:var(--am-bg);color:var(--am-t)}
.bgy{background:#F1EFE8;color:#444441}
.pos{color:var(--gr);font-weight:600} .neg{color:var(--re);font-weight:600}
.wrap{padding:18px;display:flex;flex-direction:column;gap:14px}
.card{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);padding:16px}
.tit{font-size:24px;font-weight:700;margin-bottom:9px}
.sub{font-size:15px;color:var(--t2);margin-bottom:13px;line-height:1.6}
textarea{width:100%;border:1px solid var(--br2);border-radius:8px;
  padding:13px;font-size:15px;font-family:monospace;color:var(--tx);
  resize:vertical;min-height:100px;outline:none;background:var(--sf)}
textarea:focus{border-color:var(--bl)}
.btnp{padding:11px 22px;background:var(--bl);color:#fff;border:none;
  border-radius:8px;font-size:16px;font-weight:600;cursor:pointer}
.btnp:hover{background:#0C447C}
.btns{padding:11px 16px;background:transparent;color:var(--t2);
  border:1px solid var(--br2);border-radius:8px;font-size:15px;cursor:pointer}
.pstats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.sc{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);padding:15px}
.sl{font-size:12px;color:var(--t3);font-weight:700;text-transform:uppercase;
  letter-spacing:.3px;margin-bottom:5px}
.sv{font-size:26px;font-weight:700;line-height:1.1}
.ss{font-size:13px;color:var(--t3);margin-top:4px}
.tw{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);overflow:hidden}
table{width:100%;border-collapse:collapse}
th{padding:10px 12px;font-size:12px;font-weight:700;color:var(--t3);
  text-align:left;border-bottom:1px solid var(--br);
  text-transform:uppercase;letter-spacing:.3px}
th.r{text-align:right}
td{padding:10px 12px;font-size:14px;border-bottom:1px solid var(--br);vertical-align:middle}
tr:last-child td{border-bottom:none} td.r{text-align:right}
#results{display:none;flex-direction:column;gap:14px}
.alerts{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);padding:16px}
.ar{display:flex;align-items:flex-start;gap:12px;padding:10px 0;
  border-bottom:1px solid var(--br)}
.ar:last-child{border-bottom:none}
.ai{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:13px;flex-shrink:0;font-weight:700}
.aok{background:var(--gr-bg);color:var(--gr-t)}
.aw{background:var(--am-bg);color:var(--am-t)}
.ab{background:var(--re-bg);color:var(--re-t)}
.at{font-size:15px;font-weight:500}
.as{font-size:13px;color:var(--t3);margin-top:4px;line-height:1.4}
#errdiv{background:#FCEBEB;color:#A32D2D;padding:12px;border-radius:8px;
  margin-top:10px;font-size:14px;display:none}
"""
    # JS в r-строке — Python не трогает \n и другие escape
    js = r"""
var SD, IDX = {};
try {
  SD = JSON.parse(document.getElementById('sd').textContent);
  SD.forEach(function(s){ IDX[s.tick] = s; });
} catch(e) {
  var ed = document.getElementById('errdiv');
  ed.textContent = 'Ошибка загрузки данных: ' + e.message;
  ed.style.display = 'block';
}

function showErr(msg) {
  var d = document.getElementById('errdiv');
  d.textContent = msg; d.style.display = 'block';
}
function hideErr() {
  document.getElementById('errdiv').style.display = 'none';
}
function st(label, val, sub) {
  return '<div class="sc"><div class="sl">' + label +
    '</div><div class="sv">' + val +
    '</div><div class="ss">' + sub + '</div></div>';
}
function e3badge(e) {
  var el = (e || '').toLowerCase();
  if (el.indexOf('незав') >= 0) return '<span class="b bgr">Незав.</span>';
  if (el.indexOf('умер') >= 0 || el.indexOf('связ') >= 0) return '<span class="b bam">Умерен.</span>';
  if (el.indexOf('усил') >= 0) return '<span class="b bbl">&beta;-усил.</span>';
  return '<span class="b bre">Завис.</span>';
}

function demo() {
  var ta = document.getElementById('pi');
  ta.value = ['AAPL 25%','JNJ 20%','NVDA 15%','KO 20%','PG 20%'].join(String.fromCharCode(10));
  analyze();
}

function analyze() {
  hideErr();
  var raw = document.getElementById('pi').value.trim();
  if (!raw) { showErr('Введите хотя бы один тикер'); return; }

  var lines = raw.split(String.fromCharCode(10)).filter(function(l){ return l.trim(); });
  if (!lines.length) { showErr('Список пуст'); return; }

  var h = [];
  lines.forEach(function(l) {
    var parts = l.trim().split(' ');
    var tk = parts[0].toUpperCase().replace(/[^A-Z0-9.]/g, '');
    if (!tk) return;
    var w = null;
    if (parts[1]) {
      w = parseFloat(parts[1]);
      if (!isNaN(w) && parts[1].indexOf('%') >= 0) w /= 100;
      if (isNaN(w)) w = null;
    }
    h.push({ tick: tk, weight: w });
  });

  if (!h.length) { showErr('Не удалось распознать тикеры'); return; }

  var noW  = h.filter(function(x){ return x.weight === null; }).length;
  var sumW = h.filter(function(x){ return x.weight !== null; })
              .reduce(function(a, b){ return a + b.weight; }, 0);
  var eq   = noW > 0 ? (1 - sumW) / noW : 0;
  h.forEach(function(x){ if (x.weight === null) x.weight = eq; });

  var res = h.map(function(x) {
    var d = IDX[x.tick] || {
      tick: x.tick, name: x.tick, sector: '',
      risk: 'Неизвестно', ret: '', exp3: '', mkt: '',
      sigma: 0, beta: 0, sharpe: 0, chg: 0, cui: 0.5
    };
    return Object.assign({}, d, { weight: x.weight });
  });

  var aSig  = res.reduce(function(a, b){ return a + b.sigma * b.weight; }, 0);
  var aBeta = res.reduce(function(a, b){ return a + b.beta  * b.weight; }, 0);

  document.getElementById('pstats').innerHTML =
    st('Акций', res.length, 'в портфеле') +
    st('Взвеш. &sigma;', (aSig * 100).toFixed(1) + '%', 'волатильность') +
    st('Взвеш. &beta;', aBeta.toFixed(2),
       aBeta < 0.8 ? 'Защитный' : aBeta < 1.1 ? 'Умеренный' : 'Агрессивный') +
    st('Состав', res.length + ' акций', 'по 4 экспериментам');

  var rCls = {
    'Защитные':'bgr','Агрессивные':'bre',
    'Рыночные':'bbl','Умеренно-агрессивные':'bam','Неизвестно':'bgy'
  };

  document.getElementById('ptbl').innerHTML = res.map(function(item) {
    var ml  = item.mkt || '';
    var mkl = ml.indexOf('Незав') >= 0 ? 'Незав.' : ml.indexOf('Умер') >= 0 ? 'Умерен.' : 'Рыноч.';
    var mkc = ml.indexOf('Незав') >= 0 ? 'bgr' : ml.indexOf('Умер') >= 0 ? 'bam' : 'bre';
    var cui = parseFloat(item.cui);
    var cuil = cui === 0   ? '<span class="b bgr">Выс.</span>'  :
               cui === 0.5 ? '<span class="b bam">Умер.</span>' :
                             '<span class="b bre">Низк.</span>';
    var retl = item.ret && item.ret.indexOf('Опережа') >= 0
               ? '<span class="b bgr">Опережает</span>'
               : '<span class="b bgy">Отстаёт</span>';
    return '<tr>' +
      '<td><b>' + item.tick + '</b></td>' +
      '<td class="r">' + (item.weight * 100).toFixed(0) + '%</td>' +
      '<td><span class="b ' + (rCls[item.risk] || 'bgy') + '">' + item.risk + '</span></td>' +
      '<td>' + retl + '</td>' +
      '<td>' + e3badge(item.exp3) + '</td>' +
      '<td><span class="b ' + mkc + '">' + mkl + '</span></td>' +
      '<td class="r">' + item.sigma.toFixed(3) + '</td>' +
      '<td class="r">' + item.beta.toFixed(2)  + '</td>' +
      '<td>' + cuil + '</td>' +
      '</tr>';
  }).join('');

  var aW = res.filter(function(x){ return x.risk === 'Агрессивные'; })
              .reduce(function(a, b){ return a + b.weight; }, 0);
  var nW = res.filter(function(x){ return x.mkt && x.mkt.indexOf('Незав') >= 0; })
              .reduce(function(a, b){ return a + b.weight; }, 0);

  var alerts = [];
  if (aW > 0.5)
    alerts.push({t:'bad',
      h:'Высокая доля Агрессивных: ' + (aW*100).toFixed(0) + '%',
      s:'В кризис портфель может потерять >25%.'});
  else
    alerts.push({t:'ok',
      h:'Доля Агрессивных в норме: ' + (aW*100).toFixed(0) + '%',
      s:'Концентрация в высокорисковых активах умеренная.'});
  if (nW < 0.2)
    alerts.push({t:'warn',
      h:'Мало Независимых (Эксп.4): ' + (nW*100).toFixed(0) + '%',
      s:'Добавьте Consumer Staples или Utilities для защиты в кризис.'});

  document.getElementById('palerts').innerHTML = alerts.map(function(a) {
    var ic  = a.t === 'ok' ? 'aok' : a.t === 'warn' ? 'aw' : 'ab';
    var sym = a.t === 'ok' ? '&#10003;' : a.t === 'warn' ? '!' : '&#10007;';
    return '<div class="ar"><div class="ai ' + ic + '">' + sym +
           '</div><div><div class="at">' + a.h +
           '</div><div class="as">' + a.s + '</div></div></div>';
  }).join('');

  document.getElementById('results').style.display = 'flex';
  drawDonuts(res);
}

function drawDonut(canvasId, data, colors) {
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height, cx = W/2, cy = H/2;
  var R = Math.min(cx,cy)*0.85, r = R*0.52;
  var total = data.reduce(function(a,b){return a+b.v;},0);
  if (!total) return;
  ctx.clearRect(0,0,W,H);
  var angle = -Math.PI/2;
  data.forEach(function(d,i){
    var sweep = (d.v/total)*2*Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx,cy);
    ctx.arc(cx,cy,R,angle,angle+sweep);
    ctx.closePath();
    ctx.fillStyle = colors[i % colors.length];
    ctx.fill();
    angle += sweep;
  });
  ctx.beginPath();
  ctx.arc(cx,cy,r,0,2*Math.PI);
  ctx.fillStyle = '#fff';
  ctx.fill();
  ctx.fillStyle = '#18181a';
  ctx.font = 'bold 18px Segoe UI,system-ui,sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(total, cx, cy);
}
function renderLegend(divId, data, colors) {
  var div = document.getElementById(divId);
  if (!div) return;
  div.innerHTML = data.map(function(d,i){
    return '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
      +'<span style="width:10px;height:10px;border-radius:2px;background:'+colors[i]+';flex-shrink:0"></span>'
      +'<span style="font-size:12px;color:#6b6a66">'+d.label+' ('+d.v+')</span>'
      +'</div>';
  }).join('');
}
function drawDonuts(res) {
  var e1cnt={}, e2cnt={}, e3cnt={}, e4cnt={};
  res.forEach(function(s){
    var k1=s.risk||'Неизвестно'; e1cnt[k1]=(e1cnt[k1]||0)+1;
    var k2=s.ret&&s.ret.indexOf('Опережа')>=0?'Опережает':'Отстаёт'; e2cnt[k2]=(e2cnt[k2]||0)+1;
    var el=(s.exp3||'').toLowerCase();
    var k3=el.indexOf('незав')>=0?'Независимые':el.indexOf('умер')>=0||el.indexOf('связ')>=0?'Умерен.':el.indexOf('усил')>=0?'β-усил.':'Зависимые';
    e3cnt[k3]=(e3cnt[k3]||0)+1;
    var ml=s.mkt||'';
    var k4=ml.indexOf('Незав')>=0?'Независимые':ml.indexOf('Умер')>=0?'Умерен.':'Рыночные';
    e4cnt[k4]=(e4cnt[k4]||0)+1;
  });
  var c1={'Защитные':'#3B6D11','Рыночные':'#185FA5','Умеренно-агрессивные':'#854F0B','Агрессивные':'#A32D2D','Неизвестно':'#9e9c97'};
  var c2={'Опережает':'#3B6D11','Отстаёт':'#A32D2D'};
  var c3={'Независимые':'#3B6D11','Умерен.':'#854F0B','β-усил.':'#185FA5','Зависимые':'#A32D2D'};
  var c4={'Независимые':'#3B6D11','Умерен.':'#854F0B','Рыночные':'#A32D2D'};
  [[e1cnt,c1,'dn1','lg1'],[e2cnt,c2,'dn2','lg2'],[e3cnt,c3,'dn3','lg3'],[e4cnt,c4,'dn4','lg4']].forEach(function(x){
    var data=Object.keys(x[0]).map(function(k){return{label:k,v:x[0][k]};});
    var cols=data.map(function(d){return x[1][d.label]||'#ccc';});
    drawDonut(x[2],data,cols);
    renderLegend(x[3],data,cols);
  });
}
document.getElementById('abtn').addEventListener('click', analyze);
document.getElementById('dbtn').addEventListener('click', demo);
"""
    html  = "<!DOCTYPE html><html lang='ru'><head><meta charset='UTF-8'>"
    html += "<style>" + css + "</style></head><body>"
    html += "<script id='sd' type='application/json'>" + stocks_json + "</script>"
    html += "<div class='wrap'>"
    html += "<div class='card'>"
    html += "  <div class='tit'>Анализ портфеля</div>"
    html += "  <div class='sub'>Введите тикеры и веса, один в строке: <code>AAPL 25%</code></div>"
    html += "  <textarea id='pi' placeholder='AAPL 25%&#10;JNJ 20%&#10;NVDA 15%&#10;KO 20%&#10;PG 20%'></textarea>"
    html += "  <div style='display:flex;gap:10px;margin-top:13px;flex-wrap:wrap'>"
    html += "    <button class='btnp' id='abtn'>Анализировать</button>"
    html += "    <button class='btns' id='dbtn'>Демо</button>"
    html += "  </div>"
    html += "  <div id='errdiv'></div>"
    html += "</div>"
    html += "<div id='results'>"
    html += "  <div class='pstats' id='pstats'></div>"
    html += "  <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 12px'>"
    html += "    <div style='background:#fff;border:1px solid #e6e4de;border-radius:12px;padding:14px'>"
    html += "      <div style='font-size:11px;font-weight:700;color:#9e9c97;text-transform:uppercase;letter-spacing:.4px;margin-bottom:10px'>Риск (Э1)</div>"
    html += "      <canvas id='dn1' width='120' height='120' style='display:block;margin:0 auto'></canvas>"
    html += "      <div id='lg1' style='margin-top:10px'></div></div>"
    html += "    <div style='background:#fff;border:1px solid #e6e4de;border-radius:12px;padding:14px'>"
    html += "      <div style='font-size:11px;font-weight:700;color:#9e9c97;text-transform:uppercase;letter-spacing:.4px;margin-bottom:10px'>Доходность (Э2)</div>"
    html += "      <canvas id='dn2' width='120' height='120' style='display:block;margin:0 auto'></canvas>"
    html += "      <div id='lg2' style='margin-top:10px'></div></div>"
    html += "    <div style='background:#fff;border:1px solid #e6e4de;border-radius:12px;padding:14px'>"
    html += "      <div style='font-size:11px;font-weight:700;color:#9e9c97;text-transform:uppercase;letter-spacing:.4px;margin-bottom:10px'>Сект.незав. (Э3)</div>"
    html += "      <canvas id='dn3' width='120' height='120' style='display:block;margin:0 auto'></canvas>"
    html += "      <div id='lg3' style='margin-top:10px'></div></div>"
    html += "    <div style='background:#fff;border:1px solid #e6e4de;border-radius:12px;padding:14px'>"
    html += "      <div style='font-size:11px;font-weight:700;color:#9e9c97;text-transform:uppercase;letter-spacing:.4px;margin-bottom:10px'>Межрын. (Э4)</div>"
    html += "      <canvas id='dn4' width='120' height='120' style='display:block;margin:0 auto'></canvas>"
    html += "      <div id='lg4' style='margin-top:10px'></div></div>"
    html += "  </div>"
    html += "  <div class='alerts'>"
    html += "    <div style='font-size:16px;font-weight:600;margin-bottom:10px'>Диагностика</div>"
    html += "    <div id='palerts'></div>"
    html += "  </div>"
    html += "  <div class='tw'><table>"
    html += "    <thead><tr>"
    html += "      <th>Акция</th><th class='r'>Вес</th>"
    html += "      <th>Риск (Э1)</th><th>Доходн. (Э2)</th>"
    html += "      <th>Сект.нез. (Э3)</th><th>Межрын. (Э4)</th>"
    html += "      <th class='r'>&sigma;</th><th class='r'>&beta;</th><th>CUI</th>"
    html += "    </tr></thead>"
    html += "    <tbody id='ptbl'></tbody>"
    html += "  </table></div>"
    html += "</div></div>"
    html += "<script>" + js + "</script>"
    html += "</body></html>"
    return html


def build_sector(sj):
    return f"""<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#f5f4f1;--sf:#fff;--br:#e6e4de;--br2:#ccc9be;--tx:#18181a;--t2:#6b6a66;--t3:#9e9c97;
  --bl:#185FA5;--bl-bg:#E6F1FB;--gr:#3B6D11;--gr-bg:#EAF3DE;--gr-t:#27500A;
  --re:#A32D2D;--re-bg:#FCEBEB;--re-t:#791F1F;--am:#854F0B;--am-bg:#FAEEDA;--am-t:#633806;--r:8px;--rl:12px}}
html,body{{min-height:100vh;overflow-y:auto;margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--tx);font-size:15px}}
.b{{display:inline-flex;padding:4px 10px;border-radius:20px;font-size:14px;font-weight:600}}
.bgr{{background:var(--gr-bg);color:var(--gr-t)}} .bre{{background:var(--re-bg);color:var(--re-t)}}
.bbl{{background:var(--bl-bg);color:#0C447C}} .bam{{background:var(--am-bg);color:var(--am-t)}} .bgy{{background:#F1EFE8;color:#444441}}
.pos{{color:var(--gr);font-weight:600}} .neg{{color:var(--re);font-weight:600}}
.card{{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);padding:16px}}
table{{width:100%;border-collapse:collapse}}
th{{padding:10px 12px;font-size:12px;font-weight:700;color:var(--t3);text-align:left;border-bottom:1px solid var(--br);white-space:nowrap;text-transform:uppercase;letter-spacing:.3px;position:sticky;top:0;background:var(--sf)}}
th.r{{text-align:right}} td{{padding:10px 12px;font-size:14px;border-bottom:1px solid var(--br);vertical-align:middle}}
tr:last-child td{{border-bottom:none}} tr:hover td{{background:var(--bg)}} td.r{{text-align:right}}
.wrap{{padding:16px;display:flex;flex-direction:column;gap:14px}}
.top{{display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.tit{{font-size:24px;font-weight:700}}
.sel{{padding:10px 13px;border-radius:var(--r);border:1px solid var(--br2);background:var(--sf);font-size:15px;color:var(--tx);min-width:240px}}
.tg{{display:flex;border:1px solid var(--br2);border-radius:var(--r);overflow:hidden}}
.tgb{{padding:9px 14px;font-size:14px;border:none;background:transparent;color:var(--t2);cursor:pointer;font-weight:500}}
.tgb.on{{background:var(--bg);color:var(--tx)}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.sc{{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);padding:15px}}
.sl{{font-size:12px;color:var(--t3);font-weight:700;text-transform:uppercase;letter-spacing:.3px;margin-bottom:5px}}
.sv{{font-size:24px;font-weight:700}} .ss{{font-size:13px;color:var(--t3);margin-top:4px}}
.tw{{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);overflow:hidden}}
.tw-h{{padding:13px 16px;border-bottom:1px solid var(--br);font-size:15px;font-weight:600}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.empty{{display:flex;align-items:center;justify-content:center;height:50vh;color:var(--t3);font-size:17px;flex-direction:column;gap:12px}}
</style></head><body>
<div class="wrap">
  <div class="top">
    <span class="tit">Секторальный анализ</span>
    <select class="sel" id="ss" onchange="render()"><option value="">Выберите сектор...</option></select>
    <div class="tg">
      <button class="tgb on" onclick="se(this,1)">Риск</button>
      <button class="tgb" onclick="se(this,2)">Доходность</button>
      <button class="tgb" onclick="se(this,3)">Сект.незав.</button>
      <button class="tgb" onclick="se(this,4)">Межрыночные</button>
    </div>
  </div>
  <div id="con"><div class="empty"><span style="font-size:36px">🏭</span><span>Выберите сектор из списка</span></div></div>
</div>
<script>
var SD={sj};
var ae=1;
var secs=[...new Set(SD.map(function(s){{return s.sector;}}).filter(Boolean))].sort();
var sel=document.getElementById('ss');
secs.forEach(function(s){{var o=document.createElement('option');o.value=s;o.textContent=s;sel.appendChild(o);}});
function se(btn,n){{document.querySelectorAll('.tgb').forEach(function(b){{b.classList.remove('on');}});btn.classList.add('on');ae=n;render();}}
function e3b(e){{var el=(e||'').toLowerCase();
  if(el.indexOf('незав')>=0)return '<span class="b bgr">Незав.</span>';
  if(el.indexOf('умер')>=0||el.indexOf('связ')>=0)return '<span class="b bam">Умерен.</span>';
  if(el.indexOf('усил')>=0||el.indexOf('β')>=0)return '<span class="b bbl">β-усил.</span>';
  return '<span class="b bre">Завис.</span>';}}
function render(){{
  var sec=sel.value;if(!sec)return;
  var st=SD.filter(function(s){{return s.sector===sec;}});if(!st.length)return;
  var ef=['risk','ret','exp3','mkt'][ae-1];
  var cc={{'Защитные':'#3B6D11','Рыночные':'#185FA5','Умеренно-агрессивные':'#854F0B','Агрессивные':'#A32D2D',
    'Независимые':'#3B6D11','Умеренно связанные':'#854F0B','Опережающие рынок':'#3B6D11','Отстающие':'#A32D2D',
    'Секторально независимые':'#3B6D11','β-усилители':'#185FA5','Секторально зависимые':'#A32D2D'}};
  var as=st.reduce(function(a,b){{return a+b.sigma;}},0)/st.length;
  var ab=st.reduce(function(a,b){{return a+b.beta;}},0)/st.length;
  var ash=st.reduce(function(a,b){{return a+b.sharpe;}},0)/st.length;
  var ac=st.reduce(function(a,b){{return a+b.chg;}},0)/st.length;
  var dist={{}};st.forEach(function(s){{var v=s[ef]||'—';dist[v]=(dist[v]||0)+1;}});
  var de=Object.entries(dist).sort(function(a,b){{return b[1]-a[1];}});
  var sorted=[].concat(st).sort(function(a,b){{return b.sharpe-a.sharpe;}});
  var rCls={{'Защитные':'bgr','Агрессивные':'bre','Рыночные':'bbl','Умеренно-агрессивные':'bam'}};
  function mb(m){{return m&&m.indexOf('Незав')>=0?'<span class="b bgr">Незав.</span>':m&&m.indexOf('Умер')>=0?'<span class="b bam">Умерен.</span>':'<span class="b bre">Рыноч.</span>';}}
  document.getElementById('con').innerHTML='<div class="stats">'
    +'<div class="sc"><div class="sl">Акций в секторе</div><div class="sv">'+st.length+'</div><div class="ss">из 481</div></div>'
    +'<div class="sc"><div class="sl">Средняя σ</div><div class="sv">'+(as*100).toFixed(1)+'%</div><div class="ss">волатильность</div></div>'
    +'<div class="sc"><div class="sl">Средняя β</div><div class="sv">'+ab.toFixed(2)+'</div><div class="ss">к рынку</div></div>'
    +'<div class="sc"><div class="sl">Средний Шарп</div><div class="sv '+(ash>=0?'pos':'neg')+'">'+ash.toFixed(2)+'</div><div class="ss">риск/доходность</div></div>'
    +'<div class="sc"><div class="sl">Ср. моментум 1г</div><div class="sv '+(ac>=0?'pos':'neg')+'">'+(ac>=0?'+':'')+ac.toFixed(1)+'%</div><div class="ss">изм. цены</div></div>'
    +'<div class="sc"><div class="sl">Преобл. кластер</div><div class="sv" style="font-size:16px">'+(de[0]?de[0][0].slice(0,14):'—')+'</div><div class="ss">'+(de[0]?de[0][1]+' акций':'')+'</div></div>'
    +'</div>'
    +'<div class="g2">'
    +'<div class="tw"><div class="tw-h">🏆 Топ-5 по Шарпу</div><table><thead><tr><th>Тикер</th><th>Риск</th><th>Сект.нез.</th><th class="r">Шарп</th><th class="r">Изм.1г</th></tr></thead><tbody>'
    +sorted.slice(0,5).map(function(s){{return'<tr><td><b>'+s.tick+'</b></td><td><span class="b '+(rCls[s.risk]||'bgy')+'">'+s.risk+'</span></td><td>'+e3b(s.exp3)+'</td><td class="r pos">'+s.sharpe.toFixed(2)+'</td><td class="r '+(s.chg>=0?'pos':'neg')+'">'+(s.chg>=0?'+':'')+s.chg.toFixed(1)+'%</td></tr>';}}).join('')
    +'</tbody></table></div>'
    +'<div class="tw"><div class="tw-h">⚠ Аутсайдеры</div><table><thead><tr><th>Тикер</th><th>Риск</th><th>Сект.нез.</th><th class="r">Шарп</th><th class="r">Изм.1г</th></tr></thead><tbody>'
    +sorted.slice(-5).map(function(s){{return'<tr><td><b>'+s.tick+'</b></td><td><span class="b '+(rCls[s.risk]||'bgy')+'">'+s.risk+'</span></td><td>'+e3b(s.exp3)+'</td><td class="r neg">'+s.sharpe.toFixed(2)+'</td><td class="r '+(s.chg>=0?'pos':'neg')+'">'+(s.chg>=0?'+':'')+s.chg.toFixed(1)+'%</td></tr>';}}).join('')
    +'</tbody></table></div>'
    +'</div>'
    +'<div class="tw"><div class="tw-h">Все акции сектора</div><div style="max-height:320px;overflow-y:auto"><table><thead><tr><th>Тикер</th><th>Название</th><th>Риск (Э1)</th><th>Доходн. (Э2)</th><th>Сект.нез. (Э3)</th><th>Межрын. (Э4)</th><th class="r">σ</th><th class="r">β</th><th class="r">Шарп</th><th class="r">Изм.1г</th></tr></thead><tbody>'
    +st.map(function(s){{return'<tr><td><b>'+s.tick+'</b></td><td style="font-size:13px;color:var(--t2)">'+s.name.slice(0,16)+'</td><td><span class="b '+(rCls[s.risk]||'bgy')+'">'+s.risk+'</span></td><td>'+(s.ret&&s.ret.indexOf('Опережа')>=0?'<span class="b bgr">Опережает</span>':'<span class="b bgy">Отстаёт</span>')+'</td><td>'+e3b(s.exp3)+'</td><td>'+mb(s.mkt||'—')+'</td><td class="r">'+s.sigma.toFixed(3)+'</td><td class="r">'+s.beta.toFixed(2)+'</td><td class="r '+(s.sharpe>=0?'pos':'neg')+'">'+s.sharpe.toFixed(2)+'</td><td class="r '+(s.chg>=0?'pos':'neg')+'">'+(s.chg>=0?'+':'')+s.chg.toFixed(1)+'%</td></tr>';}}).join('')
    +'</tbody></table></div></div>';
}}
</script></body></html>"""


# ════════════════════════════════════════════════════════════
# СТРАНИЦА 5 — СРАВНЕНИЕ АКЦИЙ
# ════════════════════════════════════════════════════════════
def build_compare(stocks_json):
    js = r"""
var SD_ALL = JSON.parse(document.getElementById('sd').textContent);
var IDX = {}; SD_ALL.forEach(function(s){ IDX[s.tick]=s; });
var sel = []; var COLS=['#185FA5','#3B6D11','#A32D2D','#854F0B'];
document.addEventListener('click',function(e){if(!e.target.closest('.sw'))document.getElementById('cdd').style.display='none';});
function doSearch(){
  var q=document.getElementById('csq').value.toLowerCase().trim();
  var dd=document.getElementById('cdd');
  if(!q){dd.style.display='none';return;}
  var res=[];
  for(var i=0;i<SD_ALL.length&&res.length<7;i++){
    var s=SD_ALL[i];
    if(sel.indexOf(s.tick)<0&&(s.tick.toLowerCase().indexOf(q)>=0||s.name.toLowerCase().indexOf(q)>=0))res.push(s);
  }
  if(!res.length){dd.style.display='none';return;}
  dd.innerHTML=res.map(function(r){return'<div class="di" onclick="addTick(this.dataset.tick)" data-tick="'+r.tick+'"><span class="dt">'+r.tick+'</span><span style="font-size:13px;color:var(--t2)">'+r.name.slice(0,24)+'</span></div>';}).join('');
  dd.style.display='block';
}
function addTick(tick){if(sel.length>=4||sel.indexOf(tick)>=0)return;sel.push(tick);document.getElementById('csq').value='';document.getElementById('cdd').style.display='none';renderChips();renderCmp();}
function removeTick(tick){sel=sel.filter(function(t){return t!==tick;});renderChips();renderCmp();}
function renderChips(){document.getElementById('chips').innerHTML=sel.map(function(t,i){var col=COLS[i];return'<div class="chip" style="background:'+col+'22;color:'+col+';border:1.5px solid '+col+'" onclick="removeTick(this.dataset.tick)" data-tick="'+t+'">'+t+' <span style="font-size:16px">&#215;</span></div>';}).join('');}
function e3b(e){var el=(e||'').toLowerCase();
  if(el.indexOf('незав')>=0)return 'Незав.';
  if(el.indexOf('умер')>=0||el.indexOf('связ')>=0)return 'Умерен.';
  if(el.indexOf('усил')>=0||el.indexOf('β')>=0)return 'β-усил.';
  return 'Завис.';}
function renderCmp(){
  if(sel.length<2){document.getElementById('ccon').innerHTML='<div class="empty"><span style="font-size:36px">&#9878;</span><span>Добавьте 2–4 акции</span></div>';return;}
  var stocks=sel.map(function(t){return IDX[t];}).filter(Boolean);
  var n=stocks.length;
  function row(label,vals,fmt,hi){
    var fv=vals.map(function(v){return typeof fmt==='function'?fmt(v):String(v);});
    var nums=vals.map(function(v){return typeof v==='number'?v:(v!=null?parseFloat(String(v).replace(/[^0-9.-]/g,''))||0:0);});
    var mx=Math.max.apply(null,nums.map(Math.abs))||1;
    return '<div class="cmp-card"><div class="cmp-head">'+label+'</div>'
      +'<div class="cmp-cells" style="grid-template-columns:repeat('+n+',1fr)">'
      +stocks.map(function(s,i){
        var col=COLS[i],num=nums[i],pct=Math.abs(num)/mx*100;
        var best=hi==='h'?num===Math.max.apply(null,nums):hi==='l'?num===Math.min.apply(null,nums):false;
        return '<div class="cmp-cell">'
          +'<div style="font-size:15px;font-weight:700;color:'+col+'">'+s.tick+'</div>'
          +'<div style="font-size:13px;color:var(--t2);margin-bottom:5px">'+s.name.slice(0,14)+'</div>'
          +'<div class="cv '+(best?(hi==='h'?'pos':'neg'):'')+'" style="color:'+col+'">'+fv[i]+'</div>'
          +'<div class="cbar"><div class="cbf" style="width:'+pct+'%;background:'+col+'40"></div></div>'
          +'</div>';
      }).join('')
      +'</div></div>';
  }
  document.getElementById('ccon').innerHTML='<div style="display:flex;flex-direction:column;gap:12px">'
    +row('Риск-кластер (Эксп.1)',stocks.map(function(s){return s.risk;}),null,'')
    +row('Волатильность &#963;',stocks.map(function(s){return s.sigma||0;}),function(v){return v.toFixed(3);},'l')
    +row('Бета &#946;',stocks.map(function(s){return s.beta||0;}),function(v){return v.toFixed(2);},'l')
    +row('Шарп',stocks.map(function(s){return s.sharpe||0;}),function(v){return v.toFixed(2);},'h')
    +row('Изменение 1 год',stocks.map(function(s){return s.chg||0;}),function(v){return (v>=0?'+':'')+v.toFixed(1)+'%';},'h')
    +row('Доходность (Эксп.2)',stocks.map(function(s){return s.ret||'—';}),null,'')
    +row('Сект. независ. (Эксп.3)',stocks.map(function(s){return e3b(s.exp3||'—');}),null,'')
    +row('Корр. S&P 500',stocks.map(function(s){return s.corr||0;}),function(v){return v.toFixed(3);},'l')
    +row('Межрыночные (Эксп.4)',stocks.map(function(s){return s.mkt||'—';}),null,'')
    +'</div>';
}
"""
    css = """
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#f5f4f1;--sf:#fff;--br:#e6e4de;--br2:#ccc9be;
  --tx:#18181a;--t2:#6b6a66;--t3:#9e9c97;
  --bl:#185FA5;--bl-bg:#E6F1FB;--gr:#3B6D11;--gr-bg:#EAF3DE;--gr-t:#27500A;
  --re:#A32D2D;--re-bg:#FCEBEB;--re-t:#791F1F;
  --am:#854F0B;--am-bg:#FAEEDA;--am-t:#633806;--r:8px;--rl:12px}
html,body{min-height:100vh;overflow-y:auto;margin:0;
  font-family:"Segoe UI",system-ui,sans-serif;
  background:var(--bg);color:var(--tx);font-size:15px}
.pos{color:var(--gr);font-weight:600}.neg{color:var(--re);font-weight:600}
.wrap{padding:16px;display:flex;flex-direction:column;gap:14px}
.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.tit{font-size:24px;font-weight:700}
.sw{position:relative}
.sw input{padding:10px 13px 10px 34px;border-radius:var(--r);border:1px solid var(--br2);
  background:var(--sf);font-size:15px;color:var(--tx);outline:none;width:220px}
.sw input:focus{border-color:var(--bl)}
.sic{position:absolute;left:11px;top:50%;transform:translateY(-50%);color:var(--t3);font-size:16px;pointer-events:none}
.dd{position:absolute;top:42px;left:0;background:var(--sf);border:1px solid var(--br2);
  border-radius:var(--r);z-index:200;width:280px;max-height:230px;overflow-y:auto;
  display:none;box-shadow:0 4px 16px rgba(0,0,0,.12)}
.di{padding:10px 14px;cursor:pointer;font-size:14px;display:flex;gap:10px;border-bottom:1px solid var(--br)}
.di:last-child{border-bottom:none}.di:hover{background:var(--bg)}
.dt{font-weight:700;min-width:46px}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{display:flex;align-items:center;gap:7px;padding:6px 13px;border-radius:20px;font-size:14px;font-weight:600;cursor:pointer}
.cmp-card{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);overflow:hidden}
.cmp-head{padding:13px 16px;border-bottom:1px solid var(--br);font-size:12px;
  font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:.4px;background:var(--bg)}
.cmp-cells{display:grid;gap:1px;background:var(--br)}
.cmp-cell{background:var(--sf);padding:14px 17px}
.cv{font-size:18px;font-weight:700;margin-top:5px}
.cbar{height:7px;background:var(--bg);border-radius:3px;margin-top:8px;overflow:hidden}
.cbf{height:100%;border-radius:3px}
.empty{display:flex;align-items:center;justify-content:center;height:50vh;
  color:var(--t3);font-size:17px;flex-direction:column;gap:12px}
"""
    return (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='UTF-8'>"
        "<style>" + css + "</style></head><body>"
        "<script id='sd' type='application/json'>" + stocks_json + "</script>"
        "<div class='wrap'>"
        "<div class='top'><span class='tit'>Сравнение акций</span>"
        "<div class='sw'><span class='sic'>&#9906;</span>"
        "<input type='text' id='csq' placeholder='Добавить тикер...' oninput='doSearch()' autocomplete='off'>"
        "<div class='dd' id='cdd'></div></div>"
        "<span style='font-size:14px;color:var(--t3)'>до 4 акций</span></div>"
        "<div class='chips' id='chips'></div>"
        "<div id='ccon'><div class='empty'>"
        "<span style='font-size:36px'>&#9878;</span>"
        "<span>Добавьте 2–4 акции для сравнения</span>"
        "</div></div></div>"
        "<script>" + js + "</script></body></html>"
    )


# ════════════════════════════════════════════════════════════
# СТРАНИЦА 6 — ГЛОССАРИЙ
# ════════════════════════════════════════════════════════════
def build_glossary():
    terms = [
        ("σ — Волатильность","risk","Стандартное отклонение доходностей × √252. σ=0.30 → цена колеблется ±30% в год.","Защитные: σ≈0.19, Агрессивные: σ≈0.45"),
        ("β — Бета","risk","Чувствительность к S&P 500. β=1.5: рынок −10% → акция −15%.","Защитные: β≈0.55, Агрессивные: β≈1.50"),
        ("Max Drawdown","risk","Максимальное падение от пика до дна за период.","OOS: Защитные −8.2%, Агрессивные −33.0%"),
        ("Шарп (Sharpe Ratio)","return","Доходность на единицу риска. Шарп>1 — хорошо.","Шарп>1 — хорошо. >2 — отлично. <0 — хуже депозита"),
        ("Моментум (mom_12m)","return","Изменение цены за 12 месяцев.","Опережающие→OOS: +24.1% vs Отстающие +15.1%"),
        ("Win Rate","return","Доля торговых дней с положительной доходностью.","Типичный диапазон: 48–55%"),
        ("corr_sector","sector","Корреляция акции с ETF своего сектора GICS.","Незав.≈0.54, Зависимые≈0.80"),
        ("β-усилители","sector","Акции с beta_sector>1.2: усиливают движения ETF.","MU, AMAT, LRCX — beta_sector≈1.4"),
        ("Correlation Breakdown","market","Резкий рост корреляций в кризис. В норме диверсифицированы — в панике все падают.","Liberation Day 2025: Независимые 0.22→0.64"),
        ("CUI — индекс надёжности","method","Доля алгоритмов не согласившихся с основным кластером. 0=все согласны, 1.0=все расходятся.","Эксп.4: 91% акций CUI=0"),
    ]
    cat_cfg = {
        "risk": ("#A32D2D","#FCEBEB","Риск"),
        "return": ("#3B6D11","#EAF3DE","Доходность"),
        "sector": ("#854F0B","#FAEEDA","Сект.незав."),
        "market": ("#185FA5","#E6F1FB","Межрыночные"),
        "method": ("#534AB7","#EEEDFE","Методология"),
    }
    items = ""
    for term, cat, desc, ex in terms:
        col, bg, lbl = cat_cfg.get(cat, ("#9e9c96","#F1EFE8","—"))
        items += (
            f'<div class="gi" data-cat="{cat}" onclick="tog(this)">'
            f'<div class="gh">'
            f'<span class="gt">{term}</span>'
            f'<span class="gc" style="background:{bg};color:{col}">{lbl}</span>'
            f'<span class="ga">&#9656;</span></div>'
            f'<div class="gb">'
            f'<div class="gd">{desc}</div>'
            f'<div class="ge"><b>Пример:</b> {ex}</div>'
            f'</div></div>'
        )
    return (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='UTF-8'>"
        "<style>"
        "*{box-sizing:border-box;margin:0;padding:0}"
        ":root{--bg:#f5f4f1;--sf:#fff;--br:#e6e4de;--br2:#ccc9be;--tx:#18181a;--t2:#6b6a66;--t3:#9e9c97;--bl:#185FA5;--r:8px;--rl:12px}"
        "html,body{min-height:100vh;overflow-y:auto;margin:0;font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--tx);font-size:15px}"
        ".wrap{padding:16px;display:flex;flex-direction:column;gap:14px}"
        ".top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}"
        ".tit{font-size:24px;font-weight:700}"
        ".si{position:relative}"
        ".si input{padding:10px 13px 10px 34px;border-radius:var(--r);border:1px solid var(--br2);background:var(--sf);font-size:15px;color:var(--tx);outline:none;width:260px}"
        ".si input:focus{border-color:var(--bl)}"
        ".sic{position:absolute;left:11px;top:50%;transform:translateY(-50%);color:var(--t3);font-size:16px;pointer-events:none}"
        ".cats{display:flex;gap:8px;flex-wrap:wrap}"
        ".cb{padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600;cursor:pointer;border:1.5px solid transparent;background:var(--bg);color:var(--t2)}"
        ".cb.on{border-color:currentColor}"
        ".gl{display:flex;flex-direction:column;gap:9px}"
        ".gi{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);overflow:hidden;cursor:pointer}"
        ".gi:hover{border-color:var(--br2)}"
        ".gh{display:flex;align-items:center;gap:12px;padding:16px 18px}"
        ".gt{font-size:17px;font-weight:700;flex:1}"
        ".gc{padding:4px 11px;border-radius:20px;font-size:12px;font-weight:600;white-space:nowrap}"
        ".ga{font-size:15px;color:var(--t3);transition:.2s;flex-shrink:0}"
        ".gi.open .ga{transform:rotate(90deg)}"
        ".gb{display:none;padding:0 18px 16px;border-top:1px solid var(--br)}"
        ".gi.open .gb{display:block}"
        ".gd{font-size:15px;color:var(--t2);line-height:1.7;margin-top:13px;margin-bottom:11px}"
        ".ge{font-size:14px;background:var(--bg);border-radius:var(--r);padding:10px 14px;color:var(--t2);line-height:1.5}"
        "</style></head><body>"
        "<div class='wrap'>"
        "<div class='top'><span class='tit'>Глоссарий терминов</span>"
        "<div class='si'><span class='sic'>&#9906;</span>"
        "<input type='text' id='gsq' placeholder='Поиск термина...' oninput='filt()'></div></div>"
        "<div class='cats'>"
        "<button class='cb on' data-cat='' onclick='sc(this)' style='color:var(--t2)'>Все</button>"
        "<button class='cb' data-cat='risk' onclick='sc(this)' style='color:#A32D2D'>Риск</button>"
        "<button class='cb' data-cat='return' onclick='sc(this)' style='color:#3B6D11'>Доходность</button>"
        "<button class='cb' data-cat='sector' onclick='sc(this)' style='color:#854F0B'>Сект.незав.</button>"
        "<button class='cb' data-cat='market' onclick='sc(this)' style='color:#185FA5'>Межрыночные</button>"
        "<button class='cb' data-cat='method' onclick='sc(this)' style='color:#534AB7'>Методология</button>"
        "</div>"
        "<div class='gl' id='gl'>" + items + "</div>"
        "</div>"
        "<script>"
        "var ac='';"
        "function tog(el){el.classList.toggle('open');}"
        "function sc(btn){"
        "  document.querySelectorAll('.cb').forEach(function(b){b.classList.remove('on');});"
        "  btn.classList.add('on');ac=btn.dataset.cat||'';filt();"
        "}"
        "function filt(){"
        "  var q=document.getElementById('gsq').value.toLowerCase();"
        "  document.querySelectorAll('.gi').forEach(function(el){"
        "    var tm=el.querySelector('.gt').textContent.toLowerCase();"
        "    var ds=el.querySelector('.gd').textContent.toLowerCase();"
        "    var cm=!ac||el.dataset.cat===ac;"
        "    var qm=!q||tm.indexOf(q)>=0||ds.indexOf(q)>=0;"
        "    el.style.display=(cm&&qm)?'':'none';"
        "  });"
        "}"
        "</script></body></html>"
    )


# ════════════════════════════════════════════════════════════
# СТРАНИЦА 7 — МОНИТОРИНГ
# ════════════════════════════════════════════════════════════
def build_monitor(drift_pct, df):
    ok = drift_pct <= 20
    sc_col = "#27500A" if ok else "#A32D2D"
    status = ("✓ Модель актуальна" if ok else "⚠ Рекомендуется обновление") + " (порог 20%)"
    bar_w = min(drift_pct, 100)
    bar_col = "#3B6D11" if ok else "#A32D2D"
    drift_rows = ""
    cnt = 0
    for ticker, row in df.iterrows():
        if cnt >= 12: break
        warns = []
        for feat in ["sigma", "beta"]:
            tr_v = float(row.get(feat) or 0)
            oo_v = float(row.get(f"{feat}_oos") or 0)
            if tr_v > 0 and oo_v > tr_v * 1.30:
                chg = (oo_v / tr_v - 1) * 100
                warns.append(f"{feat}: {tr_v:.3f}→{oo_v:.3f} (+{chg:.0f}%)")
        if warns:
            risk = str(row.get("risk", "—"))
            rcls = {"Защитные": "bgr", "Агрессивные": "bre", "Рыночные": "bbl", "Умеренно-агрессивные": "bam"}.get(risk, "bgy")
            drift_rows += (
                f"<tr><td><b>{ticker}</b></td>"
                f"<td style='font-size:13px;color:#6b6a66'>{str(row.get('name', ticker))[:22]}</td>"
                f"<td><span class='b {rcls}'>{risk}</span></td>"
                f"<td style='font-size:13px;color:#A32D2D'>{'; '.join(warns)}</td></tr>"
            )
            cnt += 1
    if not drift_rows:
        drift_rows = "<tr><td colspan='4' style='text-align:center;color:#9e9c97;padding:18px'>Акций с опасным дрейфом не обнаружено ✓</td></tr>"
    return (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='UTF-8'>"
        "<style>"
        "*{box-sizing:border-box;margin:0;padding:0}"
        ":root{--bg:#f5f4f1;--sf:#fff;--br:#e6e4de;--br2:#ccc9be;--tx:#18181a;--t2:#6b6a66;--t3:#9e9c97;"
        "--bl:#185FA5;--bl-bg:#E6F1FB;--gr:#3B6D11;--gr-bg:#EAF3DE;--gr-t:#27500A;"
        "--re:#A32D2D;--re-bg:#FCEBEB;--re-t:#791F1F;--am:#854F0B;--am-bg:#FAEEDA;--am-t:#633806;--r:8px;--rl:12px}"
        "html,body{min-height:100vh;overflow-y:auto;margin:0;font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--tx);font-size:15px}"
        ".b{display:inline-flex;padding:4px 10px;border-radius:20px;font-size:14px;font-weight:600}"
        ".bgr{background:var(--gr-bg);color:var(--gr-t)}.bre{background:var(--re-bg);color:var(--re-t)}"
        ".bbl{background:var(--bl-bg);color:#0C447C}.bam{background:var(--am-bg);color:var(--am-t)}.bgy{background:#F1EFE8;color:#444441}"
        ".wrap{padding:16px;display:flex;flex-direction:column;gap:14px}"
        ".mg{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}"
        ".mbc{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);padding:17px}"
        ".mbl{font-size:12px;color:var(--t3);font-weight:700;text-transform:uppercase;letter-spacing:.4px;margin-bottom:9px}"
        ".mbv{font-size:28px;font-weight:700;line-height:1}.mbs{font-size:13px;color:var(--t3);margin-top:5px}"
        ".mbbar{height:6px;border-radius:3px;background:var(--bg);margin-top:11px;overflow:hidden}"
        ".mbfill{height:100%;border-radius:3px}.mbst{margin-top:9px;font-size:14px}"
        ".tw{background:var(--sf);border:1px solid var(--br);border-radius:var(--rl);overflow:hidden}"
        ".tw-h{padding:13px 16px;border-bottom:1px solid var(--br);font-size:15px;font-weight:600}"
        "table{width:100%;border-collapse:collapse}"
        "th{padding:10px 12px;font-size:12px;font-weight:700;color:var(--t3);text-align:center;border-bottom:1px solid var(--br);text-transform:uppercase;letter-spacing:.3px}"
        "th:first-child{text-align:left}"
        "td{padding:10px 12px;font-size:14px;border-bottom:1px solid var(--br);text-align:center}"
        "td:first-child{text-align:left;font-weight:600}"
        "tr:last-child td{border-bottom:none}"
        ".cp{color:var(--gr);font-weight:600}.cn{color:var(--re);font-weight:600}"
        "</style></head><body>"
        "<div class='wrap'>"
        "<div><div style='font-size:24px;font-weight:700;margin-bottom:5px'>Мониторинг модели</div>"
        "<div style='font-size:15px;color:#6b6a66'>Актуальность кластеров · Детектор дрейфа</div></div>"
        "<div class='mg'>"
        f"<div class='mbc'><div class='mbl'>Дрейф модели (Эксп.1)</div>"
        f"<div class='mbv' style='color:{bar_col}'>{drift_pct}%</div>"
        f"<div class='mbs'>акций с опасным ростом σ или β</div>"
        f"<div class='mbbar'><div class='mbfill' style='width:{bar_w:.0f}%;background:{bar_col}'></div></div>"
        f"<div class='mbst' style='color:{sc_col}'>{status}</div></div>"
        "<div class='mbc'><div class='mbl'>Данные модели</div>"
        "<div class='mbv'>Дек 2024</div>"
        "<div class='mbs'>обучение 2019–2024 · скользящее окно 5 лет</div></div>"
        f"<div class='mbc'><div class='mbl'>Акций с дрейфом</div>"
        f"<div class='mbv' style='color:var(--am)'>{cnt}</div>"
        f"<div class='mbs'>из {len(df)} акций</div></div>"
        "</div>"
        "<div class='tw'><div class='tw-h'>Доходность кластеров по горизонтам (OOS 2025–2026)</div>"
        "<table><thead><tr><th>Горизонт</th><th>Защитные</th><th>Рыночные</th><th>Умер.-агрес.</th><th>Агрессивные</th><th>S&amp;P 500</th></tr></thead>"
        "<tbody>"
        "<tr><td>1 неделя</td><td class='cn'>−1.2%</td><td class='cn'>−0.6%</td><td class='cn'>−2.5%</td><td class='cn'>−0.6%</td><td class='cn'>−0.7%</td></tr>"
        "<tr><td>2 недели</td><td class='cp'>+2.4%</td><td class='cp'>+3.6%</td><td class='cp'>+2.8%</td><td class='cp'>+5.0%</td><td class='cp'>+2.2%</td></tr>"
        "<tr><td>1 месяц</td><td class='cp'>+6.5%</td><td class='cp'>+4.0%</td><td class='cp'>+2.7%</td><td class='cp'>+3.0%</td><td class='cp'>+2.9%</td></tr>"
        "<tr><td>3 месяца</td><td class='cp'>+5.9%</td><td class='cn'>−12.5%</td><td class='cn'>−19.2%</td><td class='cn'>−26.0%</td><td class='cn'>−13.5%</td></tr>"
        "<tr><td>6 месяцев</td><td class='cp'>+7.4%</td><td class='cp'>+3.8%</td><td class='cp'>+5.7%</td><td class='cp'>+4.0%</td><td class='cp'>+6.1%</td></tr>"
        "<tr><td>1 год</td><td class='cp'>+10.9%</td><td class='cp'>+18.3%</td><td class='cp'>+18.8%</td><td class='cp'>+24.0%</td><td class='cp'>+18.3%</td></tr>"
        "</tbody></table></div>"
        "<div class='tw'><div class='tw-h'>Акции с опасным дрейфом (σ или β выросли >30%)</div>"
        "<table><thead><tr><th>Тикер</th><th style='text-align:left'>Компания</th><th>Кластер</th><th style='text-align:left'>Изменение</th></tr></thead>"
        "<tbody>" + drift_rows + "</tbody></table></div>"
        "</div></body></html>"
    )



H = 800

tabs = st.tabs([
    "📋 Скринер",
    "📄 Паспорт акции",
    "💼 Мой портфель",
    "🏭 Секторальный анализ",
    "⚖️ Сравнение акций",
    "📚 Глоссарий",
    "📡 Мониторинг",
])

with tabs[0]:
    st.iframe(build_screener(STOCKS_JSON, TOTAL, drift_pct), height=1300)

with tabs[1]:
    st.iframe(build_passport(STOCKS_JSON, META_JSON), height=1300)

with tabs[2]:
    st.iframe(build_portfolio(STOCKS_JSON), height=1300)

with tabs[3]:
    st.iframe(build_sector(STOCKS_JSON), height=1300)

with tabs[4]:
    st.iframe(build_compare(STOCKS_JSON), height=1300)

with tabs[5]:
    st.iframe(build_glossary(), height=H)

with tabs[6]:
    st.iframe(build_monitor(drift_pct, df), height=H)
