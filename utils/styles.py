"""Visual system: palette, global CSS, decorative assets and small HTML components.

Layout model
------------
Every page is a single-screen dashboard on desktop/laptop screens: a slim header row,
one KPI strip, then a grid of fixed-height cards sized from the viewport height
(``card1_*`` = full remaining height, ``card2_*`` = half). Charts stretch to fill
the space their card leaves free, and each card's interpretation box sits at the
bottom of the card so boxes line up across a row. Below 1100 px wide (tablets,
phones) or on very short windows the grid falls back to natural heights and the
page scrolls, because one screen can no longer hold everything legibly.

Only decorative PNGs (background, header art, leaves, books, sparkles…) are used.
KPI / chart PNGs with baked-in numbers are never rendered.
"""

from __future__ import annotations

import base64
import html
import io
import re
from pathlib import Path

import numpy as np
import streamlit as st

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
FOREST = "#0B3D2C"
FOREST_2 = "#12583E"
EMERALD = "#0F8A5F"
EMERALD_SOFT = "#5FBF95"
MINT = "#A7E3C9"
MINT_LIGHT = "#E6F6EE"
BLUE = "#3E78B2"
BLUE_SOFT = "#9DBEDF"
GOLD = "#C9962B"
GOLD_SOFT = "#F3DFA8"
TERRA = "#B9573E"
INK = "#17261F"
INK_2 = "#3C4E45"
MUTED = "#6B7B73"
GRID = "rgba(11,61,44,0.08)"
NEUTRAL = "#C9D3CE"

# Callouts and notices: one flat colour, no gradient.
INSIGHT_BG = "#EEF6F1"
INSIGHT_BORDER = "#D3E8DC"
WARN_BG = "#FBEFEB"
WARN_BORDER = "#EFCFC4"

FONT = "'Plus Jakarta Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"


# ----------------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------------
@st.cache_data
def asset_b64(name: str) -> str:
    """Base64 of an asset PNG, untouched."""
    p = ASSET_DIR / name
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode()


@st.cache_data
def deco_b64(name: str, top: float = 0.10, bottom: float = 0.74) -> str:
    """Decorative element cut out of the asset sheet crop.

    The supplied decorative PNGs carry a near-white backdrop and a caption under
    the drawing; this crops the caption off and keys the white backdrop to
    transparency so the element floats over cards and backgrounds.
    """
    from PIL import Image

    p = ASSET_DIR / name
    if not p.exists():
        return ""
    im = np.asarray(Image.open(p).convert("RGB")).astype(float)
    h = im.shape[0]
    im = im[int(h * top): int(h * bottom)]
    mn = im.min(axis=2)
    alpha = np.clip((255 - mn) * 3.2, 0, 255)
    alpha[(255 - mn) < 14] = 0
    rgba = np.dstack([im, alpha]).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data
def wordmark_b64() -> str:
    """RASIO 10.0 wordmark with its dark backdrop keyed out (and the yellow corner mark cropped)."""
    from PIL import Image

    p = ASSET_DIR / "branding" / "rasio_wordmark_reference.png"
    if not p.exists():
        return ""
    im = np.asarray(Image.open(p).convert("RGB")).astype(float)[14:]
    dist = np.abs(im - np.array([26, 57, 25])).sum(axis=2)
    alpha = np.clip((dist - 18) * 6, 0, 255)
    rgba = np.dstack([im, alpha]).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data
def logo_strip_b64() -> str:
    """Partner-logo pill cut from the supplied logo strip."""
    from PIL import Image

    p = ASSET_DIR / "branding" / "logo_strip_reference.png"
    if not p.exists():
        return ""
    im = Image.open(p).convert("RGB").crop((60, 34, 1000, 176))
    im.thumbnail((520, 80))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def img_uri(b64: str) -> str:
    return f"data:image/png;base64,{b64}" if b64 else ""


# ----------------------------------------------------------------------------
# Global CSS
# ----------------------------------------------------------------------------
_INTERP_COUNTER = [0]


def inject_css(kpi_rows: int = 1, tabs: bool = False) -> None:
    """Global CSS. kpi_rows / tabs size the card grid for pages with 2 KPI rows or page-level tabs."""
    _INTERP_COUNTER[0] = 0
    bg = img_uri(asset_b64("01_background_main.png"))
    header_art = img_uri(asset_b64("02_header_banner.png"))
    side_leaf = img_uri(deco_b64("26c_leaf_bottom_left.png"))
    leaf_br = img_uri(deco_b64("26d_leaf_bottom_right.png"))
    leaf_tl = img_uri(deco_b64("26a_leaf_top_left.png"))

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400..600,0..1,0&display=swap');
:root {{
  --forest: {FOREST}; --forest-2: {FOREST_2}; --emerald: {EMERALD}; --mint: {MINT};
  --ink: {INK}; --ink-2: {INK_2}; --muted: {MUTED};
  --card-bg: rgba(255,255,255,0.86);
  --card-border: rgba(15,138,95,0.16);
  --shadow: 0 8px 24px -16px rgba(11,61,44,0.40), 0 1px 4px -1px rgba(11,61,44,0.08);
  /* layout */
  --pad-top: .55rem; --pad-bottom: .55rem; --gap: .6rem;
  --hero-h: 58px; --kpi-h: 60px;
  --kpi-rows: {kpi_rows}; --tabs-h: {46 if tabs else 0}px;
  --top: calc(var(--pad-top) + var(--pad-bottom) + var(--hero-h) + var(--kpi-rows) * var(--kpi-h)
         + (var(--kpi-rows) - 1) * var(--gap) + var(--tabs-h) + 2 * var(--gap) + 12px);
  --h1: calc(100vh - var(--top));
  --h2: calc((100vh - var(--top) - var(--gap)) / 2);
  /* cards on a tab without its own KPI strip */
  --h1n: calc(var(--h1) + var(--kpi-h) + var(--gap));
  --h2n: calc((100vh - var(--top) + var(--kpi-h)) / 2);
}}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {{ font-family: {FONT}; }}
.stApp {{
  background-color: #F3F8F5;
  background-image: linear-gradient(rgba(246,250,248,0.93), rgba(246,250,248,0.93)), url("{bg}");
  background-size: cover; background-attachment: fixed; background-position: center top; color: var(--ink);
}}

/* ---------- Streamlit chrome ---------- */
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stDecoration"], .stDeployButton,
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; min-height: 0; }}
[data-testid="stMainBlockContainer"], .block-container {{
  padding: var(--pad-top) 1.1rem var(--pad-bottom) 1.1rem !important; max-width: 100% !important;
}}
.block-container [data-testid="stVerticalBlock"] {{ gap: var(--gap); }}
[data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(.hero, .kpi-grid, .card-head, .insight, .narr-title, .diag-table, .status-line, .ctl-label) {{ margin-bottom: 0 !important; }}
.block-container [data-testid="stHorizontalBlock"] {{ gap: var(--gap) !important; }}
h1, h2, h3, h4 {{ font-family: {FONT}; color: var(--ink); }}
a {{ color: var(--emerald); }}

/* ---------- Sidebar (compact, never scrolls on laptop screens) ---------- */
section[data-testid="stSidebar"] {{ background: #0B3F2D !important; border-right: 0; }}
section[data-testid="stSidebar"]::after {{ content:""; position:absolute; left:-8px; bottom:-6px; width:120px; height:120px; pointer-events:none;
  background: url("{side_leaf}") no-repeat left bottom / contain; opacity:.30; }}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{ height: 2rem; min-height: 2rem; padding: .3rem .8rem 0 .8rem; margin: 0; }}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: 0; }}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding: 0 .8rem .6rem .8rem; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .4rem; }}
section[data-testid="stSidebar"] * {{ color: #E9F5EF; }}
section[data-testid="stSidebar"] .stMarkdown p {{ color: #CFE5DA; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.14); margin: .25rem 0; }}
.brand {{ display:flex; align-items:center; gap:.6rem; padding: .1rem 0; }}
.brand-mark {{ width:42px; height:42px; border-radius: 13px; background: rgba(255,255,255,0.08);
  display:flex; align-items:center; justify-content:center; border:1px solid rgba(255,255,255,0.15); }}
.brand-mark .material-symbols-rounded {{ font-size: 25px; color: #FFFFFF; }}
.brand-title {{ font-weight: 800; font-size: .92rem; line-height:1.15; color:#FFFFFF; letter-spacing:.02em; text-transform: uppercase; }}
.brand-sub {{ font-weight: 500; font-size: .78rem; color:#BFE3D1; }}
.side-label {{ font-size:.66rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#8FC9AE !important; margin:.1rem 0 0 .1rem; display:flex; align-items:center; gap:.35rem; }}
.side-footer {{ padding: .55rem .7rem; border-radius: 12px; background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12); font-size: .74rem; line-height: 1.5; color:#CFE5DA; }}
.side-footer b {{ color:#FFFFFF; }}
.side-note {{ font-size:.74rem; line-height:1.45; padding:.5rem .65rem; border-radius:10px; background: rgba(255,255,255,0.07);
  border:1px solid rgba(255,255,255,0.16); color:#E3F2EA !important; }}
/* navigation radio as menu: new (react-aria) DOM */
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {{ width: 100%; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {{
  width: 100%; margin: 0 0 1px 0; padding: .36rem .6rem; border-radius: 10px; cursor: pointer;
  border: 1px solid transparent; box-sizing: border-box; transition: background .15s ease;
}}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size: .80rem; font-weight: 500; color: #D7EBE1; line-height: 1.3; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p span[role="img"] {{ margin-right: .4rem; font-size: 1.05rem; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: rgba(255,255,255,0.07); }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: {EMERALD}; border-color: rgba(255,255,255,0.18); }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: #FFFFFF; font-weight: 700; }}
/* navigation radio: legacy (baseweb) DOM */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{ width: 100%; margin: 0; padding: .48rem .65rem; border-radius: 10px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{ background: {EMERALD}; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{ color:#FFFFFF; font-weight:700; }}
/* sidebar selects and buttons */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color:#CFE5DA; font-weight:600; font-size:.78rem; }}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"],
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
  background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.22) !important; border-radius: 9px; min-height: 32px; height: 32px;
}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input {{ color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF; background: transparent !important; font-size: .84rem; }}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] button svg, section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{ color: #CFE5DA; fill: #CFE5DA; }}
section[data-testid="stSidebar"] .stButton button, section[data-testid="stSidebar"] .stDownloadButton button {{
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.25); border-radius: 9px; min-height: 32px; padding: .2rem .6rem; }}
section[data-testid="stSidebar"] .stButton button:hover, section[data-testid="stSidebar"] .stDownloadButton button:hover {{ border-color: {EMERALD_SOFT}; background: rgba(95,191,149,0.15); }}

/* ---------- Header row ---------- */
.hero {{
  position: relative; overflow: hidden; border-radius: 16px; height: var(--hero-h); box-sizing: border-box;
  padding: 0 1.1rem; display:flex; align-items:center; gap: 1rem;
  background: linear-gradient(100deg, rgba(10,70,48,0.97) 0%, rgba(15,138,95,0.93) 60%, rgba(63,174,128,0.88) 100%);
  box-shadow: var(--shadow);
}}
.hero::after {{ content:""; position:absolute; inset:0; background: url("{header_art}") right center / auto 100% no-repeat; opacity:.45; mix-blend-mode: screen; pointer-events:none; }}
.hero-text {{ position:relative; z-index:2; min-width:0; flex: 1 1 auto; }}
.hero h1 {{ color:#FFFFFF !important; font-size: 1.28rem; font-weight: 800; margin: 0; padding:0; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.hero p {{ color:#E3F6EC; font-size: .80rem; margin:.1rem 0 0 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.hero-chips {{ display:flex; gap:.35rem; flex-wrap:nowrap; position:relative; z-index:2; flex: 0 1 auto; overflow:hidden; }}
.hero-chip {{ background: rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.28); color:#FFFFFF; font-size:.70rem; font-weight:600;
  padding:.2rem .6rem; border-radius:999px; white-space:nowrap; }}
.hero-chip.gold {{ background: rgba(201,150,43,0.40); border-color: rgba(243,223,168,0.6); }}
.hero-deco {{ position:relative; z-index:2; height: 46px; flex: 0 0 auto; }}
/* header action column (popovers, selectors) */
div[class*="st-key-hdr_actions"] {{ height: var(--hero-h); justify-content: center; gap: .35rem !important; }}
div[class*="st-key-hdr_actions"] [data-testid="stWidgetLabel"] {{ display:none; }}
div[class*="st-key-hdr_actions"] button {{ min-height: 30px; padding: .15rem .6rem; border-radius: 9px; font-size: .80rem; }}
div[class*="st-key-hdr_actions"] [data-testid="stSelectbox"] [role="group"], div[class*="st-key-hdr_actions"] div[data-baseweb="select"] > div {{ min-height: 36px; border-radius: 10px; border-color: rgba(15,138,95,0.35); background: #FFFFFF; }}

div[class*="st-key-hdr_actions"] [data-testid="stMultiSelect"] [role="group"] {{ min-height: 38px; max-height: 44px; overflow-y: auto;
  border-radius: 10px; border: 1px solid rgba(15,138,95,0.35); background: #FFFFFF; }}
div[class*="st-key-hdr_actions"] [data-testid="stMultiSelect"] [role="group"] {{ max-height: 62px; }}
div[class*="st-key-hdr_actions"] [data-testid="stMultiSelectTagsContainer"] {{ flex-wrap: wrap; gap: 3px; padding: 3px 0; }}
div[class*="st-key-hdr_actions"] [data-testid="stMultiSelectTagsContainer"] [data-tag] {{ margin: 0; padding: 0 .3rem 0 .45rem; height: 22px; font-size: .72rem; }}
div[class*="st-key-hdr_actions"] [data-testid="stMultiSelectTagsContainer"] [data-tag] button {{ margin-left: .15rem; }}
div[class*="st-key-hdr_actions"] [data-testid="stMultiSelectTagsContainer"] input {{ height: 22px; min-width: 2rem; }}

/* ---------- KPI strip ---------- */
.kpi-grid {{ display:grid; gap: var(--gap); grid-template-columns: repeat(var(--n), minmax(0, 1fr)); grid-auto-rows: var(--kpi-h); }}
.kpi-grid.multi .kpi-value, .kpi-grid.dense.multi .kpi-value {{ font-size: .78rem; letter-spacing: -.01em; }}
.kpi-grid.multi .kpi {{ gap: .45rem; }}
.kpi-grid.multi .kpi-icon {{ width: 30px; height: 30px; min-width: 30px; }}
.kpi {{
  position:relative; background: var(--card-bg); border:1px solid var(--card-border); border-radius:14px; box-sizing: border-box;
  padding:.45rem .6rem; display:flex; align-items:center; gap:.55rem; box-shadow: var(--shadow); min-width:0; height: var(--kpi-h); overflow:hidden;
}}
.kpi-icon {{ flex: 0 0 36px; width:36px; height:36px; border-radius:11px; display:flex; align-items:center; justify-content:center; }}
.kpi-icon .material-symbols-rounded {{ font-size: 21px; color:#FFFFFF; }}
.kpi-body {{ min-width:0; flex: 1 1 auto; }}
.kpi-value {{ font-size: 1.12rem; font-weight: 800; color: var(--ink); line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.kpi-label {{ font-size: .68rem; color: var(--muted); font-weight:600; margin-top:.1rem; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.kpi-sub {{ font-size: .62rem; color: var(--muted); line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.kpi .help {{ position:absolute; top:.3rem; right:.4rem; font-size: 13px; color: #9AB0A5; cursor: help; }}
.kpi-grid.dense .kpi {{ padding: .4rem .45rem; gap: .4rem; }}
.kpi-grid.dense .kpi-icon {{ flex-basis: 28px; width: 28px; height: 28px; border-radius: 9px; }}
.kpi-grid.dense .kpi-icon .material-symbols-rounded {{ font-size: 17px; }}
.kpi-grid.dense .kpi-value {{ font-size: 1.0rem; }}
.kpi-grid.dense .kpi-label {{ font-size: .64rem; }}
.tone-forest {{ background: #0F5A3F; }} .tone-emerald {{ background: #0F8A5F; }} .tone-teal {{ background: #10717A; }}
.tone-blue {{ background: #3E78B2; }} .tone-gold {{ background: #C9962B; }} .tone-terra {{ background: #B9573E; }}
.tone-mint {{ background: #3FAE80; }} .tone-slate {{ background: #5E6E66; }}

/* ---------- Cards: fixed height from the viewport ---------- */
div.stVerticalBlock[class*="st-key-card"] {{
  position: relative; background: var(--card-bg); border: 1px solid var(--card-border);
  border-radius: 16px; box-shadow: var(--shadow); box-sizing: border-box;
  padding: .6rem .75rem .65rem .75rem; overflow: hidden; gap: .35rem !important; flex-wrap: nowrap;
}}
div.stVerticalBlock[class*="st-key-card1_"] {{ height: var(--h1) !important; min-height: var(--h1) !important; max-height: var(--h1) !important; }}
div.stVerticalBlock[class*="st-key-card2_"] {{ height: var(--h2) !important; min-height: var(--h2) !important; max-height: var(--h2) !important; }}
div.stVerticalBlock[class*="st-key-card1n_"] {{ height: var(--h1n) !important; min-height: var(--h1n) !important; max-height: var(--h1n) !important; }}
div.stVerticalBlock[class*="st-key-card2n_"] {{ height: var(--h2n) !important; min-height: var(--h2n) !important; max-height: var(--h2n) !important; }}
div.stVerticalBlock[class*="st-key-card"]::after {{
  content: ""; position:absolute; right:-6px; top:-8px; width: 46px; height: 46px; pointer-events:none;
  background: url("{leaf_br}") no-repeat right top / contain; opacity: .35; transform: scaleY(-1);
}}
div.stVerticalBlock[class*="_narr"] {{ background: #F4FAF6; }}
div.stVerticalBlock[class*="_narr"]::before {{
  content: ""; position:absolute; left:-8px; bottom:-10px; width: 64px; height: 64px; pointer-events:none;
  background: url("{leaf_tl}") no-repeat left bottom / contain; opacity: .40;
}}
/* the chart takes all free space; the insight box is pushed to the bottom */
div.stVerticalBlock[class*="st-key-card"] > div:has([data-testid="stPlotlyChart"]) {{ flex: 1 1 0 !important; min-height: 0; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stPlotlyChart"] {{ height: 100% !important; }}
div.stVerticalBlock[class*="st-key-card"] > div:has(.insight) {{ margin-top: auto; flex: 0 0 auto; }}
div.stVerticalBlock[class*="st-key-card"] > div:has(.scroll-area) {{ flex: 1 1 0 !important; min-height: 0; overflow: auto; }}
/* tabs inside cards */
div.stVerticalBlock[class*="st-key-card"] > div:has([data-testid="stTabs"]) {{ flex: 1 1 0 !important; min-height: 0; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] {{ height: 100%; display:flex; flex-direction:column; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [role="tablist"] {{ flex: 0 0 auto; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] > div {{ flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [data-testid="stTabPanel"] {{ flex: 1 1 0; min-height: 0; overflow: auto; padding-top: .3rem; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [data-testid="stTabPanel"] > [data-testid="stVerticalBlock"] {{ height: 100%; flex-wrap: nowrap; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [data-testid="stTabPanel"] [data-testid="stVerticalBlock"] > div:has(.insight) {{ margin-top: auto; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [role="tabpanel"] {{ flex: 1 1 0; min-height: 0; overflow: auto; padding-top: .3rem; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [role="tabpanel"] > div > [data-testid="stVerticalBlock"] {{ height: 100%; flex-wrap: nowrap; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [role="tabpanel"] [data-testid="stVerticalBlock"] > div:has([data-testid="stPlotlyChart"]) {{ flex: 1 1 0 !important; min-height: 120px; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] button[role="tab"] p {{ font-size: .78rem; font-weight: 600; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [role="tablist"] {{ gap: .9rem; }}

.card-head {{ position:relative; z-index:2; }}
.ch-row {{ display:flex; align-items:center; gap:.5rem; min-width:0; }}
.pill {{ display:inline-flex; align-items:center; gap:.25rem; font-size:.60rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;
  padding:.18rem .5rem; border-radius:999px; color:#FFFFFF; background: var(--forest-2); white-space:nowrap; flex: 0 0 auto; }}
.pill .material-symbols-rounded {{ font-size: 13px; }}
.pill.emerald {{ background: #0F8A5F; }} .pill.blue {{ background: #3E78B2; }} .pill.gold {{ background: #B9862A; }}
.pill.mint {{ background: #2F9A6E; }} .pill.slate {{ background: #5F7A8A; }} .pill.terra {{ background: #B9573E; }}
.card-title {{ font-size: .95rem; font-weight: 800; color: var(--ink); line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }}
.card-sub {{ font-size: .72rem; color: var(--muted); margin: .12rem 0 0 0; line-height:1.3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.narr-title {{ font-size: 1.12rem; font-weight: 800; color: var(--forest); line-height:1.2; margin:.35rem 60px .3rem 0; }}
.narr-body {{ font-size: .80rem; color: var(--ink-2); line-height:1.5; }}
.narr-books {{ position:absolute; right: 0; top: 1.4rem; height: 52px; }}

/* ---------- Interpretation box: one flat colour ---------- */
.insight {{ position:relative; z-index:2; display:flex; gap:.5rem; align-items:flex-start; padding:.45rem .6rem; border-radius:10px;
  background: {INSIGHT_BG}; border:1px solid {INSIGHT_BORDER}; }}
.insight .material-symbols-rounded {{ font-size: 18px; color: {EMERALD}; flex: 0 0 auto; margin-top:1px; }}
.insight-body {{ font-size:.74rem; color: var(--ink-2); line-height:1.42;
  display:-webkit-box; -webkit-line-clamp: var(--clamp, 3); -webkit-box-orient: vertical; overflow:hidden; }}
.insight-body b {{ color: var(--ink); }}
.insight-body .it {{ font-weight:800; color: var(--forest); margin-right:.25rem; }}
.insight.warn {{ background: {WARN_BG}; border-color: {WARN_BORDER}; }}
.insight.warn .material-symbols-rounded {{ color: {TERRA}; }}

/* ---------- Badges / tables / misc ---------- */
.badge {{ display:inline-flex; align-items:center; gap:.25rem; padding:.12rem .45rem; border-radius:999px; font-size:.66rem; font-weight:700; white-space:nowrap; }}
.badge.ok {{ background: rgba(15,138,95,0.12); color:#0B6B48; }}
.badge.bad {{ background: rgba(185,87,62,0.14); color:#9A3F28; }}
.badge.warn {{ background: rgba(201,150,43,0.18); color:#8A6413; }}
.badge.na {{ background: rgba(107,123,115,0.14); color:#4F5E57; }}
.badge .material-symbols-rounded {{ font-size: 13px; }}
.diag-table {{ width:100%; border-collapse: collapse; font-size:.72rem; }}
.diag-table th {{ text-align:left; font-size:.62rem; letter-spacing:.06em; text-transform:uppercase; color: var(--muted); font-weight:700;
  padding:.3rem .4rem; border-bottom: 1px solid rgba(11,61,44,0.12); position: sticky; top: 0; background: #FFFFFF; }}
.diag-table td {{ padding:.28rem .4rem; border-bottom: 1px solid rgba(11,61,44,0.07); color: var(--ink-2); vertical-align: middle; line-height:1.3; }}
.diag-table tr:last-child td {{ border-bottom: 0; }}
.diag-table td.num {{ font-variant-numeric: tabular-nums; font-weight:600; color: var(--ink); white-space:nowrap; }}
.eq {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.70rem; }}
.status-line {{ display:flex; align-items:flex-start; gap:.4rem; padding:.4rem .55rem; border-radius:10px; font-size:.72rem; line-height:1.4; }}
.status-line.ok {{ background: {INSIGHT_BG}; border:1px solid {INSIGHT_BORDER}; color:#0B5A3D; }}
.status-line.bad {{ background: {WARN_BG}; border:1px solid {WARN_BORDER}; color:#8E3A24; }}
.status-line .material-symbols-rounded {{ font-size: 17px; }}
.status-line .eq {{ display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }}
.ctl-label {{ font-size:.70rem; font-weight:800; color: var(--forest); letter-spacing:.04em; text-transform:uppercase; margin-top:.15rem; }}

/* ---------- Widgets inside cards: compact ---------- */
div.stVerticalBlock[class*="st-key-card"] [data-testid="stWidgetLabel"] p {{ font-size: .72rem; font-weight: 600; color: var(--ink-2); }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stWidgetLabel"] {{ min-height: 0; margin-bottom: .1rem; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stSelectbox"] [role="group"],
div.stVerticalBlock[class*="st-key-card"] div[data-baseweb="select"] > div {{ min-height: 32px; border-radius: 9px; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stSelectbox"] input {{ font-size: .80rem; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stButtonGroup"] button,
div.stVerticalBlock[class*="st-key-card"] div[data-testid="stSegmentedControl"] button {{ min-height: 26px; padding: .1rem .6rem; font-size: .74rem; }}
div.stVerticalBlock[class*="st-key-card"] [data-testid="stButtonGroup"] button p {{ font-size: .74rem; }}
div.stVerticalBlock[class*="st-key-card"] .stButton button {{ min-height: 30px; padding: .15rem .55rem; border-radius: 9px; }}
div.stVerticalBlock[class*="st-key-card"] .stButton button p {{ font-size: .76rem; }}
[data-testid="stMain"] div[data-baseweb="select"] > div {{ border-color: rgba(15,138,95,0.25); }}
[data-testid="stPlotlyChart"] {{ position: relative; z-index: 2; }}
[data-testid="stPopoverBody"] {{ max-height: 80vh; overflow: auto; }}
.material-symbols-rounded {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
  letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; direction: ltr; -webkit-font-smoothing: antialiased; }}

/* ---------- Sidebar branding (RASIO / team) ---------- */
.rasio-brand {{ position:relative; text-align:center; padding:.15rem .2rem .45rem .2rem; margin-top:-.2rem; }}
.rasio-brand::before {{ content:"∑  π  √x  ∫  Δ  σ²  ≈  β  ∞  x̄  ∑  π  √x  ∫  Δ  σ²  ≈  β"; position:absolute; inset:-4px 0 0 0;
  font-size: 15px; line-height: 22px; letter-spacing: 6px; color:#FFFFFF; opacity:.05; overflow:hidden; pointer-events:none; word-break: break-all; }}
.rasio-brand img.wordmark {{ position:relative; height: 24px; max-width: 100%; display:block; margin: 0 auto; }}
.rasio-brand img.logos {{ position:relative; height: 30px; max-width: 100%; display:block; margin:.35rem auto 0 auto; border-radius: 999px;
  box-shadow: 0 2px 8px -4px rgba(0,0,0,0.5); }}
.rasio-team {{ position:relative; font-size:.70rem; font-weight:700; letter-spacing:.06em; color:#BFE3D1; margin-top:.3rem; }}
/* overview characteristic controls: let the two button groups sit side by side and wrap only when needed */
div[class*="st-key-card"][class*="_ov_chars"] [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; row-gap: .3rem; }}
div[class*="st-key-card"][class*="_ov_chars"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ flex: 0 0 auto !important; width: auto !important; min-width: 0; }}
/* sidebar multiselect and segmented control */
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [role="group"] {{
  background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.22) !important; border-radius: 9px;
  min-height: 32px; max-height: 66px; overflow-y: auto; }}
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] input {{ color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF; font-size:.82rem; background: transparent !important; }}
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] input::placeholder {{ color:#BFE3D1 !important; -webkit-text-fill-color:#BFE3D1; opacity: .9; }}
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] svg {{ color:#CFE5DA; stroke:#CFE5DA; }}
section[data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] > *:not(input) {{ background: rgba(95,191,149,0.32) !important; border-color: transparent !important; }}
section[data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] span {{ font-size: .72rem; color:#FFFFFF; }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] {{ width: 100%; }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [role="radiogroup"] {{ width: 100%; display:flex; }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [role="radiogroup"] button {{ flex: 1 1 0; min-width: 0; min-height: 30px; padding: .1rem .3rem;
  background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.22); }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [role="radiogroup"] button[aria-checked="true"] {{
  background: {EMERALD}; border-color: rgba(255,255,255,0.3); }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [role="radiogroup"] button p {{ font-size: .78rem; color:#FFFFFF; }}
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] button {{ background: transparent !important; border: 0 !important; }}
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]) {{ width: 100% !important; }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [role="radiogroup"] {{ display:flex !important; flex-wrap: nowrap; width: 100% !important; max-width: none !important; }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [role="radiogroup"] button {{ flex: 1 1 0 !important; max-width: none !important; }}
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [role="group"] button {{ background: transparent !important; border: 0 !important; }}
.active-view {{ font-size:.72rem; line-height:1.4; padding:.4rem .6rem; border-radius:10px; background: rgba(255,255,255,0.07);
  border:1px solid rgba(255,255,255,0.16); color:#E3F2EA !important; }}
.active-view b {{ color:#FFFFFF; }}

/* ---------- Page-level tabs (internal views) ---------- */
div[class*="st-key-ptabs"] [role="tablist"]:not(div[class*="st-key-card"] [role="tablist"]) {{
  gap: .35rem; border-bottom: 0; padding: 0; }}
div[class*="st-key-ptabs"] [role="tab"]:not(div[class*="st-key-card"] [role="tab"]) {{
  background: rgba(255,255,255,0.80); border: 1px solid var(--card-border); border-radius: 10px; padding: .2rem .9rem; height: 32px;
  display:flex; align-items:center; box-shadow: var(--shadow); }}
div[class*="st-key-ptabs"] [role="tab"]:not(div[class*="st-key-card"] [role="tab"]) p {{ font-size: .80rem; font-weight: 700; color: var(--ink-2); }}
div[class*="st-key-ptabs"] [role="tab"][aria-selected="true"]:not(div[class*="st-key-card"] [role="tab"]) {{ background: {EMERALD}; border-color: {EMERALD}; }}
div[class*="st-key-ptabs"] [role="tab"][aria-selected="true"]:not(div[class*="st-key-card"] [role="tab"]) p {{ color: #FFFFFF; }}
div[class*="st-key-ptabs"] [role="tab"]:not(div[class*="st-key-card"] [role="tab"]) .react-aria-SelectionIndicator {{ display: none; }}
div[class*="st-key-ptabs"] [role="tablist"]:not(div[class*="st-key-card"] [role="tablist"])::after {{ display:none; }}

/* ---------- Interpretation popover: floats at the card's top-right ---------- */
div.stVerticalBlock[class*="st-key-card"] > div:has(> div[class*="st-key-interp_"]),
div.stVerticalBlock[class*="st-key-card"] > div[class*="st-key-interp_"] {{
  position: absolute !important; top: .5rem; right: .7rem; width: auto !important; z-index: 6; margin: 0 !important; }}
div[class*="st-key-interp_"] {{ width: auto !important; }}
/* popovers inside a card's own tabs: anchor to the card corner as well */
div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabPanel"] div[class*="st-key-interp_"] {{
  position: absolute !important; top: .5rem; right: .7rem; width: auto !important; z-index: 6; margin: 0 !important; }}
div[class*="st-key-card1_lab_controls"] [data-testid="stSelectbox"] [role="group"] * {{ font-size: .76rem !important; }}
div[class*="st-key-interp_"] button {{ min-height: 24px !important; height: 24px; padding: 0 .5rem !important; border-radius: 999px !important;
  background: #FFFFFF; border: 1px solid rgba(15,138,95,0.35) !important; gap: .2rem; }}
div[class*="st-key-interp_"] button p {{ font-size: .68rem !important; font-weight: 700; color: {FOREST_2}; }}
div[class*="st-key-interp_"] button [data-testid="stIconMaterial"], div[class*="st-key-interp_"] button > div:last-child:has(svg) {{ display: none; }}
div[class*="st-key-interp_"] button span[data-testid="stIconEmoji"], div[class*="st-key-interp_"] button span {{ font-size: .78rem; }}
div.stVerticalBlock[class*="st-key-card"]:has(div[class*="st-key-interp_"]) .card-head {{ padding-right: 112px; }}
.interp-pop {{ font-size: .82rem; line-height: 1.55; color: var(--ink-2); max-width: 420px; }}
.interp-pop b {{ color: var(--ink); }}
[data-testid="stPopoverBody"]:has(.interp-pop) {{ max-width: 460px; }}

/* ---------- Stat callouts (r, slope, n, p-values, conclusions) ---------- */
.stat-row {{ display:flex; flex-wrap: wrap; gap:.3rem; position:relative; z-index:2; }}
.stat {{ display:inline-flex; align-items:center; gap:.3rem; font-size:.70rem; line-height:1.2; padding:.18rem .5rem; border-radius:999px;
  background: {INSIGHT_BG}; border:1px solid {INSIGHT_BORDER}; color: var(--ink-2); white-space: nowrap; }}
.stat b {{ color: var(--ink); font-weight: 800; }}
.stat.gold {{ background: rgba(201,150,43,0.14); border-color: rgba(201,150,43,0.35); }}
.stat.blue {{ background: rgba(62,120,178,0.10); border-color: rgba(62,120,178,0.28); }}
.stat.terra {{ background: {WARN_BG}; border-color: {WARN_BORDER}; }}
.stat .material-symbols-rounded {{ font-size: 14px; }}
.head-help {{ font-size: 14px; color: #9AB0A5; cursor: help; margin-left: .1rem; flex: 0 0 auto; }}
.model-id {{ border-bottom: 1px dotted currentColor; cursor: help; }}
.spec-box {{ font-size:.74rem; line-height:1.45; padding:.4rem .55rem; border-radius:10px; background: {INSIGHT_BG}; border:1px solid {INSIGHT_BORDER}; color: var(--ink-2); }}
.spec-box b {{ color: var(--ink); }}
.spec-box .eq {{ display:block; margin-top:.15rem; color: var(--forest); }}

/* ---------- Responsive fallback: tablets, phones, very short windows ---------- */
/* short laptop screens: keep the sidebar inside the viewport */
@media (max-height: 720px) {{
  section[data-testid="stSidebar"] .side-note, section[data-testid="stSidebar"] .side-footer, .rasio-brand img.logos {{ display: none; }}
  section[data-testid="stSidebar"] [data-testid="stRadioOption"] {{ padding-top: .26rem !important; padding-bottom: .26rem !important; }}
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .3rem; }}
  .insight-body {{ -webkit-line-clamp: 2; }}
}}
@media (max-width: 1100px), (max-height: 600px) {{
  [data-testid="stHeader"] {{ height: 2.5rem; }}
  [data-testid="stMainBlockContainer"], .block-container {{ padding-top: 2.6rem !important; }}
  .hero {{ height: auto; min-height: var(--hero-h); padding: .6rem .9rem; flex-wrap: wrap; }}
  .hero h1, .hero p {{ white-space: normal; }}
  .hero-chips {{ flex-wrap: wrap; }}
  .hero-deco {{ display:none; }}
  div[class*="st-key-hdr_actions"] {{ height: auto; }}
  .kpi-grid, .kpi-grid.dense {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important; height: auto; }}
  div.stVerticalBlock[class*="st-key-card"]:has(div[class*="st-key-interp_"]) .card-head {{ padding-right: 0; }}
  div.stVerticalBlock[class*="st-key-card"] > div:has(> div[class*="st-key-interp_"]),
  div.stVerticalBlock[class*="st-key-card"] > div[class*="st-key-interp_"] {{ position: static !important; }}
  div.stVerticalBlock[class*="st-key-card1"], div.stVerticalBlock[class*="st-key-card2"] {{
    height: auto !important; min-height: 0 !important; max-height: none !important; }}
  div.stVerticalBlock[class*="st-key-card"] > div:has([data-testid="stPlotlyChart"]),
  div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [role="tabpanel"] [data-testid="stVerticalBlock"] > div:has([data-testid="stPlotlyChart"]) {{
    flex: 0 0 auto !important; height: 320px; }}
  div.stVerticalBlock[class*="st-key-card"] > div:has(.scroll-area), div.stVerticalBlock[class*="st-key-card"] > div:has([data-testid="stTabs"]) {{ flex: 0 0 auto !important; }}
  div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [role="tabpanel"] {{ overflow: visible; }}
  div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] > div, div.stVerticalBlock[class*="st-key-card"] [data-testid="stTabs"] [data-testid="stTabPanel"] {{ flex: 0 0 auto; overflow: visible; }}
  .insight-body {{ -webkit-line-clamp: unset; display:block; }}
  .card-title, .card-sub {{ white-space: normal; }}
}}
@media (max-width: 640px) {{
  .hero h1 {{ font-size: 1.1rem; }}
  .kpi-grid, .kpi-grid.dense {{ grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }}
  div.stVerticalBlock[class*="st-key-card"] > div:has([data-testid="stPlotlyChart"]) {{ height: 280px; }}
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HTML components
# ----------------------------------------------------------------------------
def esc(s) -> str:
    """HTML-escape; newlines become &#10; so multi-line tooltips survive Streamlit's markdown parser."""
    return html.escape(str(s)).replace("\n", "&#10;")


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def icon(name: str) -> str:
    return f'<span class="material-symbols-rounded">{name}</span>'


def _chip(c, cls: str = "hero-chip") -> str:
    """A chip is plain text, or (text, tooltip) to explain it on hover (used for model IDs)."""
    if isinstance(c, tuple):
        text, tip = c
        return f'<span class="{cls}" title="{esc(tip)}">{esc(text)} ⓘ</span>'
    return f'<span class="{cls}">{esc(c)}</span>'


def page_header(title: str, subtitle: str, kicker: str = "", chips: list | None = None,
                deco: str = "26h_books.png", gold_chip=None) -> None:
    deco_uri = img_uri(deco_b64(deco))
    chip_html = "".join(_chip(c) for c in (chips or []) if c)
    if gold_chip:
        chip_html += _chip(gold_chip, "hero-chip gold")
    st.markdown(
        f"""<div class="hero"><div class="hero-text"><h1 title="{esc(title)}">{esc(title)}</h1>
<p title="{esc(subtitle)}">{esc(subtitle)}</p></div>
<div class="hero-chips">{chip_html}</div>
{'<img class="hero-deco" src="' + deco_uri + '"/>' if deco_uri else ''}</div>""",
        unsafe_allow_html=True,
    )


def header_row(title: str, subtitle: str, chips: list[str] | None = None, deco: str = "26h_books.png",
               gold_chip: str | None = None, actions_width: float = 0):
    """Hero banner; when actions_width > 0 also returns a container to its right for popovers/selectors."""
    if actions_width <= 0:
        page_header(title, subtitle, chips=chips, deco=deco, gold_chip=gold_chip)
        return None
    left, right = st.columns([12 - actions_width, actions_width], vertical_alignment="center")
    with left:
        page_header(title, subtitle, chips=chips, deco=deco, gold_chip=gold_chip)
    return right.container(key="hdr_actions")


def kpi_strip(items: list[dict], cols: int | None = None, multi: bool = False) -> None:
    """items: dict(label, value, icon, tone, sub=None, help=None). cols < len(items) wraps into rows."""
    n = cols or len(items)
    cards = []
    for it in items:
        help_html = (f'<span class="material-symbols-rounded help" title="{esc(it["help"])}">info</span>'
                     if it.get("help") else "")
        sub = f'<div class="kpi-sub" title="{esc(it["sub"])}">{esc(it["sub"])}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="kpi" title="{esc(it["label"])}: {esc(it["value"])}">{help_html}'
            f'<div class="kpi-icon tone-{it.get("tone", "emerald")}">{icon(it.get("icon", "insights"))}</div>'
            f'<div class="kpi-body"><div class="kpi-value">{esc(it["value"])}</div>'
            f'<div class="kpi-label">{esc(it["label"])}</div>{sub}</div></div>'
        )
    dense = (" dense" if n >= 7 else "") + (" multi" if multi else "")
    st.markdown(f'<div class="kpi-grid{dense}" style="--n:{n};">' + "".join(cards) + "</div>",
                unsafe_allow_html=True)


def card_head(pill: str, title: str, subtitle: str = "", tone: str = "emerald", pill_icon: str | None = None,
              help: str | None = None) -> None:
    ic = icon(pill_icon) if pill_icon else ""
    sub_plain = strip_tags(subtitle)
    sub = f'<div class="card-sub" title="{esc(sub_plain)}">{subtitle}</div>' if subtitle else ""
    hlp = f'<span class="material-symbols-rounded head-help" title="{esc(help)}">info</span>' if help else ""
    st.markdown(
        f'<div class="card-head"><div class="ch-row"><span class="pill {tone}">{ic}{esc(pill)}</span>'
        f'<span class="card-title" title="{esc(title)}">{esc(title)}</span>{hlp}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def stats_row(items: list) -> None:
    """Small inline callouts: each item is (label, value) or (label, value, tone, tooltip)."""
    out = []
    for it in items:
        if not it:
            continue
        label, value, *rest = it
        tone = rest[0] if rest else ""
        tip = rest[1] if len(rest) > 1 else ""
        t = f' title="{esc(tip)}"' if tip else ""
        lab = f"{esc(label)} " if label else ""
        out.append(f'<span class="stat {tone}"{t}>{lab}<b>{value}</b></span>')
    st.markdown('<div class="stat-row">' + "".join(out) + "</div>", unsafe_allow_html=True)


def model_tag(model_id: str) -> str:
    """Model ID with its specification on hover (an ID is never shown unexplained)."""
    from . import data as D
    return f'<span class="model-id" title="{esc(D.model_tooltip(model_id))}">{esc(model_id)}</span>'


def spec_box(model_id: str, show_formula: bool = True) -> None:
    """Visible specification: plain functional form plus the technical formula."""
    from . import data as D
    sp = D.model_spec(model_id)
    f = f'<span class="eq">{esc(sp["formula"])}</span>' if show_formula else ""
    st.markdown(f'<div class="spec-box"><b>{esc(model_id)}</b> · {esc(sp["plain"])}{f}</div>', unsafe_allow_html=True)


def insight(body_html: str, title: str = "Interpretation", kind: str = "info", lines: int = 3) -> None:
    """Interpretation behind a small 💡 button at the card's top-right; it opens as a floating popover,
    so it never takes chart space. kind="warn" keeps an inline notice (e.g. not enough data)."""
    if kind != "warn":
        _INTERP_COUNTER[0] += 1
        with st.container(key=f"interp_{_INTERP_COUNTER[0]}"):
            with st.popover(title, icon="💡"):
                st.markdown(f'<div class="interp-pop">{body_html}</div>', unsafe_allow_html=True)
        return
    ic = "warning" if kind == "warn" else "lightbulb"
    cls = "warn" if kind == "warn" else ""
    full = esc(strip_tags(f"{title}: {body_html}"))
    st.markdown(
        f'<div class="insight {cls}" title="{full}">{icon(ic)}'
        f'<div class="insight-body" style="--clamp:{lines}"><span class="it">{esc(title)}</span>{body_html}</div></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "ok") -> str:
    ic = {"ok": "check_circle", "bad": "error", "warn": "warning", "na": "help"}.get(kind, "info")
    return f'<span class="badge {kind}">{icon(ic)}{esc(text)}</span>'


def card(key: str, rows: int = 2, kpi: bool = True):
    """Glass card with a fixed viewport-based height: rows=1 full height, rows=2 half height.
    kpi=False sizes the card for a tab that has no KPI strip above its cards."""
    return st.container(key=f"card{rows}{'' if kpi else 'n'}_{key}")
