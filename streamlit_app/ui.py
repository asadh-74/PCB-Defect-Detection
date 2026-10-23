"""Presentation components for PCB Inspection Studio; no inference logic."""
from html import escape

import streamlit as st


def apply_theme():
    st.markdown('''<style>
    :root {color-scheme:dark}
    .stApp {background:#0b1215;color:#edf4f2}
    [data-testid="stHeader"] {background:rgba(11,18,21,.94)}
    .block-container {max-width:1400px;padding-top:2rem;padding-bottom:3rem}
    [data-testid="stSidebar"] {background:#101c20;border-right:1px solid #283b40}
    [data-testid="stSidebar"] .block-container {padding-top:1.5rem}
    h1,h2,h3 {letter-spacing:-.035em!important;color:#edf4f2!important}
    h2 {font-size:1.5rem!important} h3 {font-size:1.15rem!important}
    p,li {line-height:1.65}
    .pcb-top {display:flex;align-items:center;justify-content:space-between;gap:16px;margin:8px 0 24px}
    .pcb-brand {font-size:15px;font-weight:750;letter-spacing:.04em;color:#f0faf6}
    .pcb-brand b {display:inline-grid;place-items:center;background:#63e6be;color:#09251d;border-radius:8px;width:34px;height:34px;margin-right:10px}
    .pcb-tag {display:inline-block;font-size:11px;letter-spacing:.08em;border:1px solid #30464c;border-radius:999px;padding:6px 11px;color:#a7c3bc;text-transform:uppercase}
    .pcb-hero {position:relative;overflow:hidden;padding:36px;border:1px solid #2a4446;border-radius:22px;background:#122a2b;margin-bottom:22px}
    .pcb-hero:after {content:'';position:absolute;right:-45px;top:-60px;width:330px;height:330px;border:1px solid #356b60;border-radius:50%;box-shadow:0 0 0 42px #173432,0 0 0 43px #28534c,0 0 0 86px #15302e;opacity:.55;pointer-events:none}
    .pcb-hero>* {position:relative;z-index:1}
    .pcb-eyebrow {color:#72e6be;font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase}
    .pcb-hero h1 {font-size:clamp(32px,4.5vw,56px);line-height:1.08;font-weight:750;margin:15px 0!important;max-width:680px}
    .pcb-hero h1 span {color:#72e6be}
    .pcb-hero p {color:#bad0c9;max-width:590px;font-size:15px;margin-bottom:22px}
    .pcb-pills {display:flex;gap:8px;flex-wrap:wrap}
    .pcb-pills span {background:#102323;border:1px solid #36534f;color:#c4d8d1;font-size:11px;padding:6px 10px;border-radius:6px}
    .pcb-stats {display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
    .pcb-stat {background:#142125;border:1px solid #293d42;padding:20px;border-radius:14px}
    .pcb-stat small {display:block;font-size:11px;color:#9cb7af;letter-spacing:.04em}
    .pcb-stat strong {display:block;color:#e9f5ef;font-size:28px;line-height:1.45;font-weight:650}
    .pcb-stat span {font-size:11px;color:#91a7a2}
    .pcb-steps {display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0 24px}
    .pcb-step {border-bottom:1px solid #30454a;padding:12px 0;color:#c8d8d3;font-size:12px}
    .pcb-step b {color:#75dfbd;margin-right:10px;font-size:11px}
    .pcb-section {font-size:11px;letter-spacing:.14em;color:#82a89b;text-transform:uppercase;margin:24px 0 6px;font-weight:700}
    .pcb-empty {text-align:center;border:1px dashed #36534e;border-radius:16px;padding:38px 20px;margin:14px 0 24px;background:#101e21}
    .pcb-empty .pcb-glyph {font-size:34px;color:#75e7c0;line-height:1}
    .pcb-empty h3 {margin:14px 0 8px;font-size:20px!important}
    .pcb-empty p {max-width:540px;margin:0 auto;color:#9db8ae;font-size:13px}
    .pcb-guide {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
    .pcb-defect {border:1px solid #2b4246;border-radius:12px;padding:20px;background:#142125}
    .pcb-defect b {font-size:14px;display:block;color:#e3f0ea;margin:8px 0}
    .pcb-defect p {font-size:12px;color:#a0b9b0;margin:0}
    .pcb-defect small {font-family:monospace;color:#77ddba;letter-spacing:.15em}
    .pcb-note {border-left:3px solid #66d5b0;background:#132623;color:#a6c6b9;padding:14px 18px;font-size:12px;border-radius:0 8px 8px 0;margin:18px 0}
    .pcb-footer {margin-top:32px;padding-top:20px;border-top:1px solid #2a3b3e;color:#8ca89d;font-size:11px;display:flex;justify-content:space-between;gap:18px;flex-wrap:wrap}
    .pcb-footer a {color:#75dfbd}
    [data-testid="stMetric"] {background:#142125;border:1px solid #2b4144;border-radius:12px;padding:15px}
    [data-testid="stMetricValue"] {font-size:1.6rem;color:#81e8c3}
    [data-testid="stMetricLabel"] {color:#a6bfb5;font-size:12px}
    [data-testid="stFileUploaderDropzone"] {border:1px dashed #477368;background:#152a29;border-radius:14px;padding:24px}
    [data-baseweb="tab-list"] {gap:24px;border-bottom:1px solid #29403f}
    [data-baseweb="tab"] {padding:12px 0;font-weight:600}
    [data-baseweb="tab"][aria-selected="true"] {color:#75e7c0}
    [data-testid="stImage"] img {border-radius:12px}
    [data-testid="stDownloadButton"] button {width:100%;border-radius:9px;border-color:#3a6257;background:#14372c;color:#b9f3d9}
    button:focus-visible,a:focus-visible {outline:2px solid #86f5cf!important;outline-offset:3px}
    @media(max-width:800px) {.pcb-hero {padding:24px}.pcb-hero:after {opacity:.16}.pcb-stats {grid-template-columns:repeat(2,1fr)}.pcb-guide {grid-template-columns:repeat(2,1fr)}.pcb-top {align-items:flex-start}.block-container {padding-left:1rem;padding-right:1rem}.pcb-stat {padding:15px}}
    @media(max-width:480px) {.pcb-guide {grid-template-columns:1fr}.pcb-tag {font-size:9px}.pcb-step {font-size:10px}.pcb-stat strong {font-size:24px}}
    @media(prefers-reduced-motion:reduce) {* {scroll-behavior:auto!important;animation:none!important;transition:none!important}}
    </style>''', unsafe_allow_html=True)


def render_header(metrics, model_name, ready):
    status = 'Checkpoint available' if ready else 'Checkpoint needed'
    st.markdown(f'''<div class="pcb-top"><div class="pcb-brand"><b>◈</b> PCB / INSPECTION STUDIO</div><span class="pcb-tag">{status}</span></div>
    <div class="pcb-hero"><div class="pcb-eyebrow">Computer vision · Bare-board inspection</div>
    <h1>Find the fault.<br><span>See the detail.</span></h1>
    <p>Turn a board image into a focused defect review. Inspect marked regions, compare the original, and export the evidence.</p>
    <div class="pcb-pills"><span>{escape(model_name)} detector</span><span>6 defect classes</span><span>PNG + CSV exports</span><span>Human review</span></div></div>''', unsafe_allow_html=True)
    def percent(key):
        value = metrics.get(key)
        return f'{value * 100:.1f}%' if isinstance(value, (int, float)) else '—'
    scope = 'Subset benchmark' if metrics.get('subset_evaluation') else 'Recorded test benchmark'
    cards = [('mAP @ 0.5', percent('map50'), scope), ('Precision', percent('precision'), scope),
             ('Recall', percent('recall'), scope), ('Defect categories', '06', 'Bare boards only')]
    st.markdown('<div class="pcb-stats">' + ''.join(f'<div class="pcb-stat"><small>{label}</small><strong>{value}</strong><span>{note}</span></div>' for label, value, note in cards) + '</div>', unsafe_allow_html=True)


def render_empty():
    st.markdown('''<div class="pcb-empty"><div class="pcb-glyph">⌖</div><h3>Your inspection starts here</h3>
    <p>Drop a bare-board image above. The workspace will show annotated findings, original-image comparison, defect counts and exports. No sample predictions are shown.</p></div>''', unsafe_allow_html=True)


def render_guide(defect_info):
    st.markdown('<div class="pcb-guide">' + ''.join(f'<div class="pcb-defect"><small>CLASS {i+1:02d}</small><b>{escape(name.replace("_", " ").title())}</b><p>{escape(description)}</p></div>' for i, (name, description) in enumerate(defect_info.items())) + '</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown('''<div class="pcb-footer"><span>Built by Asad Hussain · Computer vision portfolio</span>
    <span><a href="https://github.com/asadh-74/PCB-Defect-Detection" target="_blank" rel="noopener noreferrer">View source ↗</a> &nbsp; / &nbsp; Inspection aid · not a certified AOI system</span></div>''', unsafe_allow_html=True)
