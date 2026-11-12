"""Redraws Figures 1-3 and 7-10 of the SCA-QF-MSGT manuscript.

Numbers that are not computed from predictions (Figure 2 class counts, the 6 h
target, delta) are set in the DATA block below. Everything that comes from model
output (AUPRC per test set, precision-recall curves, confusion matrix, lead times,
single-patient trajectories, HRV densities) is written by figures/from_predictions.py
into digit.json and data_override.json, which this script reads.

Usage:  python figures/from_predictions.py ...   then   python figures/make_figures.py [output folder]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
_DGP = os.path.join(HERE, 'digit.json')
DG = json.load(open(_DGP)) if os.path.exists(_DGP) else {}

# ------------------------------------------------------------------ DATA
DELTA = 0.42                      # operating threshold, Table 4
TARGET_H = 6                      # clinical lead-time target, Section 3.12
MODELS = ['ResNet-Attention', 'Multi-Scale GCN', 'CNN-LSTM', 'SCA-QF-MSGT']
SHORT = ['ResNet-Att.', 'MS-GCN', 'CNN-LSTM', 'SCA-QF-MSGT']
# AUPRC on each 12-lead test set (SCAOS, MIMIC-IV)
AUPRC_SRC = {'ResNet-Attention': (0.82, 0.68), 'Multi-Scale GCN': (0.81, 0.72),
             'CNN-LSTM': (0.84, 0.65), 'SCA-QF-MSGT': (0.96, 0.90)}
# lead time of true alarms: mean (Table 5), median, Q1, Q3 (Table S7)
LEAD = {'ResNet-Attention': (3.2, 3.0, 1.9, 4.3), 'Multi-Scale GCN': (4.1, 3.9, 2.4, 5.6),
        'CNN-LSTM': (3.8, 3.5, 2.2, 5.1), 'SCA-QF-MSGT': (6.3, 6.4, 4.1, 8.6)}
# ResNet-Attention confusion matrix on MIMIC-IV, rows = true (normal, pre-arrest)
CM_RESNET = [[450, 50], [70, 30]]
# ECG-level class counts (Table S1): (y = 0, y = 1)
CLASS = {'SCAOS': (13492, 1430), 'MIMIC-IV-ECG': (84534, 1813)}
# Values computed by from_predictions.py replace the ones above and the traced curves
OVR = os.path.join(HERE, 'data_override.json')
REAL = os.path.exists(OVR)
if REAL:
    _o = json.load(open(OVR))
    AUPRC_SRC.update({k: tuple(v) for k, v in _o.get('AUPRC_SRC', {}).items()})
    LEAD.update({k: tuple(v) for k, v in _o.get('LEAD', {}).items()})
    CM_RESNET = _o.get('CM_RESNET', CM_RESNET)
    DELTA = _o.get('DELTA', DELTA)
HRV_LABEL = {'sdnn_ms': 'SDNN (ms)', 'rmssd_ms': 'RMSSD (ms)', 'mean_hr': 'Mean heart rate (bpm)'}.get(
    (_o if REAL else {}).get('HRV_METRIC'), 'HRV metric')
# ------------------------------------------------------------------ STYLE
COL = {'ResNet-Attention': '#7d8a8c', 'Multi-Scale GCN': '#2a9d8f',
       'CNN-LSTM': '#8e5ea2', 'SCA-QF-MSGT': '#d35400'}
SRC_COL = {'SCAOS': '#3b75af', 'MIMIC-IV': '#c0392b'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 11.5,
                     'axes.titleweight': 'bold', 'axes.labelsize': 10.5, 'legend.fontsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.7,
                     'savefig.dpi': 300, 'axes.axisbelow': True, 'savefig.bbox': 'tight'})


def smooth(y, k=7, mode='edge'):
    y = np.asarray(y, float)
    if len(y) < k: return y
    pad = np.pad(y, k // 2, mode=mode)
    return np.array([np.median(pad[i:i + k]) for i in range(len(y))])


def curve(key, xmin=None, xmax=None, ymax=None, k=9):
    x, y = (np.array(v, float) for v in DG[key])
    if REAL:
        o = np.argsort(x); return x[o], y[o]
    keep = np.ones_like(x, bool)
    if xmin is not None: keep &= x >= xmin
    if xmax is not None: keep &= x <= xmax
    if ymax is not None: keep &= y <= ymax
    x, y = x[keep], y[keep]
    o = np.argsort(x); x, y = x[o], y[o]
    ok = np.abs(y - smooth(y, 21, 'reflect')) < 0.06        # drop stray pixels (labels, legend)
    return x[ok], smooth(y[ok], k)


def pr(key):
    x, y = (np.array(v, float) for v in DG[key])
    if REAL: return x, y
    # drop legend pixels in the upper right of the original panel
    keep = ~((x > 0.7) & (y > 0.45))
    x, y = x[keep], y[keep]
    o = np.argsort(x)
    return x[o], np.clip(smooth(y[o], 11), 0, 1)


def tag(ax, s):
    ax.set_title(s, loc='left')


def auprc_panel(ax, models, title):
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        v = AUPRC_SRC[m]
        xs = np.arange(2) + (j - (len(models) - 1) / 2) * w
        b = ax.bar(xs, v, w * 0.92, color=COL[m], label=m, edgecolor='white', linewidth=0.6)
        for xx, vv in zip(xs, v):
            ax.text(xx, vv + 0.008, '%.2f' % vv, ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['SCAOS test', 'MIMIC-IV test'])
    ax.set_ylim(0.5, 1.02); ax.set_ylabel('AUPRC'); ax.grid(axis='x', visible=False)
    tag(ax, title)
    if len(models) > 1: ax.legend(loc='lower left', frameon=False, ncol=2)


def pr_panel(ax, prefix, model, title):
    for src, key in (('SCAOS', 'scaos'), ('MIMIC-IV', 'mimic')):
        x, y = pr(prefix + key)
        ax.plot(x, y, color=SRC_COL[src], lw=1.8, label='%s, %s test' % (model, src))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); tag(ax, title)
    ax.legend(frameon=False, loc='upper right')


def lead_box(ax, models, title):
    stats = []
    for m in models:
        mean, med, q1, q3 = LEAD[m]
        stats.append(dict(med=med, q1=q1, q3=q3, whislo=q1, whishi=q3, mean=mean, label=''))
    bp = ax.bxp(stats, showmeans=True, showfliers=False, showcaps=False, patch_artist=True, widths=0.55,
                meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                medianprops=dict(color='black', lw=1.6))
    for p, m in zip(bp['boxes'], models):
        p.set_facecolor(COL[m]); p.set_alpha(0.85); p.set_edgecolor('black')
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([SHORT[MODELS.index(m)].replace('-', '-\n', 1) if len(models) > 2 else SHORT[MODELS.index(m)]
                        for m in models])
    ax.axhline(TARGET_H, color='#c0392b', ls='--', lw=1.3)
    ax.text(0.02, TARGET_H / 10 + 0.012, '6 h target', color='#c0392b', ha='left', va='bottom', fontsize=8.5,
            transform=ax.transAxes)
    ax.set_ylim(0, 10); ax.set_ylabel('Lead time of true alarms (h)'); ax.grid(axis='x', visible=False)
    ax.legend(handles=[Line2D([], [], color='black', lw=1.6, label='Median'),
                       Line2D([], [], marker='D', ls='', mfc='white', mec='black', label='Mean'),
                       plt.Rectangle((0, 0), 1, 1, fc='#bbbbbb', ec='black', label='IQR')],
              frameon=False, loc='upper left', fontsize=8.5)
    tag(ax, title)


def crossing(x, y, thr):
    """hours before arrest at which the curve first exceeds thr (x decreasing in time)."""
    o = np.argsort(-x); x, y = x[o], y[o]
    idx = np.where(y > thr)[0]
    if not len(idx): return None
    i = idx[0]
    if i == 0: return x[0]
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return x0 + (thr - y0) * (x1 - x0) / (y1 - y0)


def traj_axes(ax, title):
    ax.axhline(DELTA, color='black', ls=':', lw=1.2)
    ax.set_xlim(24, 0); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Time before cardiac arrest (h)'); ax.set_ylabel('Predicted risk score')
    tag(ax, title)


def three(figsize=(10, 7.6), hspace=0.42):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=hspace, wspace=0.65)
    return fig, [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)


REPORT = {}

# ================================================================ FIGURE 1
def box(ax, x, y, w, h, text, fc, ec, num=None, sub=None, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=fc, ec=ec, lw=1.2))
    if num is not None:
        ax.add_patch(Circle((x - w / 2 + 0.03, y + h / 2 - 0.03), 0.105, fc=ec, ec='white', lw=1, zorder=5))
        ax.text(x - w / 2 + 0.03, y + h / 2 - 0.03, str(num), ha='center', va='center', color='white',
                fontsize=8, fontweight='bold', zorder=6)
    if sub:
        ax.text(x, y + 0.11, text, ha='center', va='center', fontsize=fs, fontweight='bold', color='#222')
        ax.text(x, y - 0.13, sub, ha='center', va='center', fontsize=fs - 1.2, color='#333')
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color='#222')


def arrow(ax, x0, y0, x1, y1, **kw):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=kw.get('c', '#444'), lw=1.1,
                                connectionstyle=kw.get('cs', 'arc3'), shrinkA=0, shrinkB=0))


def figure1():
    fig = plt.figure(figsize=(10, 13.5))
    # ---- (a) pipeline
    ax = fig.add_axes([0.0, 0.0, 0.47, 1.0]); ax.set_xlim(0, 4.6); ax.set_ylim(0.2, 13.2); ax.axis('off')
    P = ('#ecebe4', '#6b6b6b'); G = ('#dff2ea', '#1d7a5f'); C = ('#e1edf8', '#24609a')
    Q = ('#e9e6fb', '#5a4aa8'); F = ('#f9e4dc', '#9c4221')
    stages = [
        (P, 'Input 12-lead ECG', 'resampled to 510 Hz (Sec. 3.5); shape 12 × T'),
        (P, 'Band-pass filter + z-score', 'Eqs. (1)-(2); shape 12 × T'),
        (P, 'R-peaks on lead II, 0.6 s beats', 'Eq. (3); 12 × 306 samples per beat'),
        (G, 'Node features', r'Eq. (4); $X^{(k)} \in \mathbb{R}^{12\times4}$'),
        (G, 'Pearson edges, τ = 0.5', r'Eqs. (5)-(6); $A^{(k)} \in \mathbb{R}^{12\times12}$'),
        (C, '3 edge-weighted GAT layers + readout', r'Eqs. (7)-(12), M heads; $z_G \in \mathbb{R}^{64}$'),
        (Q, 'Angles + 4-qubit circuit', r'Eqs. (13)-(18); $q \in [-1,1]^4 \rightarrow z_Q \in \mathbb{R}^{64}$'),
        (F, 'Sigmoid gate and fusion', r'Eqs. (19)-(21); $z_F \in \mathbb{R}^{64}$'),
        (F, 'Classifier + threshold δ', r'Eqs. (22)-(23); one risk score $\hat{y}$ per beat'),
        (F, 'Sequence alarm and lead time', r'Sec. 3.12, Eqs. (25a)-(25c); alarm, $\hat{t}_{lead}$'),
    ]
    ys = np.linspace(12.4, 0.9, 10)
    for k, ((fc, ec), t, s) in enumerate(stages):
        box(ax, 2.3, ys[k], 3.7, 0.78, t, fc, ec, num=k + 1, sub=s, fs=9.5)
        if k: arrow(ax, 2.3, ys[k - 1] - 0.39, 2.3, ys[k] + 0.39)
    # skip connection z_G -> fusion
    ax.annotate('', xy=(4.15, ys[7]), xytext=(4.15, ys[5]),
                arrowprops=dict(arrowstyle='-|>', color=C[1], lw=1.2, connectionstyle='bar,fraction=-0.25'))
    ax.text(4.5, (ys[5] + ys[7]) / 2, r'$z_G$', color=C[1], fontsize=9, ha='center', va='center', rotation=90)
    ax.text(2.3, 13.0, '(a) Pipeline, stages (1)-(10) of Section 3.2', ha='center', fontsize=11, fontweight='bold')

    # ---- (b) circuit
    ax = fig.add_axes([0.49, 0.745, 0.51, 0.227]); ax.set_xlim(-0.6, 11.2); ax.set_ylim(-2.8, 4.3); ax.axis('off')
    ax.text(5.3, 4.1, '(b) 4-qubit circuit, Eqs. (15)-(17)', ha='center', fontsize=11, fontweight='bold')
    qy = [3, 2, 1, 0]
    for i, y in enumerate(qy):
        ax.text(-0.35, y, 'q%d  |0⟩' % (i + 1), ha='right', va='center', fontsize=9.5)
        ax.plot([-0.25, 10.6], [y, y], color='#555', lw=1, zorder=0)

    def gate(x, y, t, fc, ec, w=0.95):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.3), w, 0.6, boxstyle='round,pad=0.01,rounding_size=0.08',
                                    fc=fc, ec=ec, lw=1.1, zorder=3))
        ax.text(x, y, t, ha='center', va='center', fontsize=7.5, zorder=4)

    for i, y in enumerate(qy):
        gate(0.6, y, r'$R_Y(\theta_%d)$' % (i + 1), '#fff3d6', '#b7791f', 1.05)

    def ring(x0):
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for j, (c, t) in enumerate(pairs):
            x = x0 + j * 0.42
            yc, yt = qy[c], qy[t]
            ax.plot([x, x], [yc, yt], color='#333', lw=1, zorder=2)
            ax.add_patch(Circle((x, yc), 0.07, color='#333', zorder=4))
            ax.add_patch(Circle((x, yt), 0.16, fc='white', ec='#333', lw=1, zorder=4))
            ax.plot([x, x], [yt - 0.16, yt + 0.16], color='#333', lw=1, zorder=5)
            ax.plot([x - 0.16, x + 0.16], [yt, yt], color='#333', lw=1, zorder=5)

    for d, xs in enumerate((2.05, 5.75)):
        ax.add_patch(FancyBboxPatch((xs - 0.65, -0.55), 3.3, 4.1, boxstyle='round,pad=0.02,rounding_size=0.1',
                                    fc='#f3f1fd', ec='#5a4aa8', lw=1, ls='--', zorder=-1))
        ax.text(xs + 1.0, 3.75, 'layer d = %d' % (d + 1), ha='center', fontsize=8.5, color='#5a4aa8')
        for i, y in enumerate(qy):
            gate(xs, y, r'$R_Y(\phi_%d^{(%d)})$' % (i + 1, d + 1), '#e9e6fb', '#5a4aa8', 1.15)
        ring(xs + 0.95)
    for i, y in enumerate(qy):
        gate(9.8, y, r'$\langle Z_%d \rangle$' % (i + 1), '#f9e4dc', '#9c4221', 0.8)
    ax.text(0.6, -0.8, 'Encoding, Eqs. (13)-(15)\n' + r'$\theta = \pi\,\tanh(W_q z_G + b_q)$', ha='center', va='top', fontsize=8)
    ax.text(5.3, -1.75, 'D = 2 variational layers, Eq. (16): trainable ' + r'$R_Y$' + ' rotations\nfollowed by a CNOT ring (q1→q2→q3→q4→q1)',
            ha='center', va='top', fontsize=8)
    ax.text(9.8, -0.8, 'Readout, Eq. (17)\n' + r'$q_i = \langle Z_i \rangle$', ha='center', va='top', fontsize=8)

    # ---- (c) graph attention
    ax = fig.add_axes([0.55, 0.43, 0.4, 0.29]); ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.75, 1.6)
    ax.set_aspect('equal'); ax.axis('off')
    ax.text(0, 1.55, '(c) Edge-weighted graph attention', ha='center', fontsize=11, fontweight='bold')
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    ang = np.pi / 2 - np.arange(12) * 2 * np.pi / 12
    pos = {l: (np.cos(a), np.sin(a)) for l, a in zip(leads, ang)}
    wts = {'I': .75, 'III': .8, 'aVR': .45, 'aVL': .5, 'aVF': .85, 'V1': .3, 'V2': .25, 'V3': .35,
           'V4': .5, 'V5': .6, 'V6': .65}
    for l, wv in wts.items():
        (x0, y0), (x1, y1) = pos['II'], pos[l]
        ax.plot([x0, x1], [y0, y1], color='#d9822b', lw=0.6 + 4 * wv, alpha=0.25 + 0.7 * wv, zorder=1)
    for l, (x, y) in pos.items():
        hub = l == 'II'
        ax.add_patch(Circle((x, y), 0.19, fc='#f9e4dc' if hub else '#ecebe4', ec='#9c4221' if hub else '#777',
                            lw=1.4 if hub else 0.9, zorder=3))
        ax.text(x, y, l, ha='center', va='center', fontsize=8.5, zorder=4, fontweight='bold' if hub else None)
    ax.text(0, -1.3, 'Edges from lead II; width and opacity show\n' + r'$\alpha_{ij}$' + ' of Eq. (9), which uses the edge weight ' + r'$A_{ij}$'
            '\n(illustrative weights)', ha='center', va='top', fontsize=8.3)

    # ---- (d) fusion
    ax = fig.add_axes([0.51, 0.1, 0.48, 0.3]); ax.set_xlim(0, 3); ax.set_ylim(-0.1, 4.75); ax.axis('off')
    ax.text(1.5, 4.62, '(d) Adaptive fusion and head', ha='center', fontsize=11, fontweight='bold')
    box(ax, 0.72, 3.8, 1.3, 0.55, r'$z_G \in \mathbb{R}^{64}$', *C)
    box(ax, 2.28, 3.8, 1.3, 0.55, r'$z_Q \in \mathbb{R}^{64}$', *Q)
    box(ax, 1.5, 2.75, 2.8, 0.7, 'Gate, Eqs. (19)-(20)', *F, sub=r'$g = \sigma(W_g[z_G \,\|\, z_Q] + b_g)$', fs=9)
    box(ax, 1.5, 1.65, 2.8, 0.7, 'Fusion, Eq. (21)', *F, sub=r'$z_F = g \odot z_Q + (1-g) \odot z_G$', fs=9)
    box(ax, 1.5, 0.45, 2.8, 0.7, 'Head, Eqs. (22)-(23)', *P, sub=r'$\hat{y} = \sigma(W_c z_F + b_c)$, alarm if $\hat{y} > \delta$', fs=9)
    arrow(ax, 0.72, 3.52, 1.2, 3.1); arrow(ax, 2.28, 3.52, 1.8, 3.1)
    arrow(ax, 1.5, 2.4, 1.5, 2.0); arrow(ax, 1.5, 1.3, 1.5, 0.8)

    # legend
    ax = fig.add_axes([0.51, 0.0, 0.49, 0.07]); ax.axis('off')
    items = [(P, 'Signal processing / output head'), (G, 'Graph construction'), (C, 'Graph encoder'),
             (Q, 'Quantum circuit (simulated)'), (F, 'Fusion and alarm')]
    for k, ((fc, ec), t) in enumerate(items):
        x = (k % 2) * 0.5; y = 0.75 - (k // 2) * 0.33
        ax.add_patch(FancyBboxPatch((x, y), 0.04, 0.25, boxstyle='round,pad=0.005', fc=fc, ec=ec,
                                    transform=ax.transAxes))
        ax.text(x + 0.055, y + 0.12, t, transform=ax.transAxes, va='center', fontsize=9)
    save(fig, 'Figure1.png')


# ================================================================ FIGURE 2
def figure2():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    srcs = list(CLASS)
    x = np.arange(len(srcs)); w = 0.36
    n0 = [CLASS[s][0] for s in srcs]; n1 = [CLASS[s][1] for s in srcs]
    b0 = ax.bar(x - w / 2, n0, w, color='#3b75af', label='No arrest within 12 h (y = 0)')
    b1 = ax.bar(x + w / 2, n1, w, color='#c0392b', label='Pre-arrest, ≤ 12 h (y = 1)')
    for bars, vals in ((b0, n0), (b1, n1)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.12, '{:,}'.format(v), ha='center', fontsize=9)
    for i, s in enumerate(srcs):
        prev = n1[i] / (n0[i] + n1[i])
        ax.text(x[i], 2.6e5, 'y = 1: %.1f%%' % (100 * prev), ha='center', fontsize=9.5, fontweight='bold')
    ax.set_yscale('log'); ax.set_ylim(500, 5e5)
    ax.set_xticks(x); ax.set_xticklabels(srcs); ax.set_ylabel('Number of ECGs (log scale)')
    ax.grid(axis='x', visible=False); ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title('ECG-level class distribution, 12-lead sources')
    save(fig, 'Figure2.png')


# ================================================================ FIGURE 3
def figure3():
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    res = {}
    for k, xc, col, side in (('normal', 0, '#3b75af', -1), ('pre', 1, '#c0392b', 1)):
        a = np.array(DG['f3'][k], float); y, w = a[:, 0], a[:, 1]
        grid = np.arange(np.floor(y.min()), np.ceil(y.max()) + 0.5, 0.5)
        wi = np.interp(grid, y[::-1], w[::-1]) if y[0] > y[-1] else np.interp(grid, y, w)
        wi = np.convolve(np.pad(wi, 20, mode='edge'), np.ones(41) / 41, mode='same')[20:-20]
        wi = np.convolve(np.pad(wi, 10, mode='edge'), np.ones(21) / 21, mode='same')[10:-10]
        keep = grid >= 0
        grid, wi = grid[keep], wi[keep]
        wi = wi / wi.max() * 0.42
        ax.fill_betweenx(grid, xc, xc + side * wi, color=col, alpha=0.85, lw=0)
        ax.plot(xc + side * wi, grid, color='#333', lw=0.8)
        cdf = np.cumsum(wi); cdf /= cdf[-1]
        q = [np.interp(p, cdf, grid) for p in (0.25, 0.5, 0.75)]
        res[k] = q
        for qq, ls in zip(q, (':', '-', ':')):
            ww = np.interp(qq, grid, wi)
            ax.plot([xc, xc + side * ww], [qq, qq], color='#111', ls=ls, lw=1.2)
    ax.axvline(0.5, color='#999', lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal segments', 'Pre-arrest segments'])
    ax.set_xlim(-0.6, 1.6); ax.set_ylabel(HRV_LABEL)
    if not REAL:
        ax.set_ylim(0, 125)
    ax.grid(axis='x', visible=False)
    ax.legend(handles=[Line2D([], [], color='#111', lw=1.2, label='Median'),
                       Line2D([], [], color='#111', ls=':', lw=1.2, label='Quartiles')],
              frameon=False, loc='upper right')
    ax.set_title('HRV distribution, normal vs pre-arrest')
    REPORT['fig3_quartiles'] = {k: [round(v, 1) for v in q] for k, q in res.items()}
    save(fig, 'Figure3.png')


# ================================================================ FIGURE 7
def figure7():
    fig, axs = three()
    auprc_panel(axs[0], ['ResNet-Attention'], '(a) AUPRC by test set')
    pr_panel(axs[1], 'f7_pr_', 'ResNet-Att.', '(b) Precision-recall curves')
    ax = axs[2]; cm = np.array(CM_RESNET)
    ax.imshow(cm, cmap='Reds', vmin=0, vmax=cm.max() * 1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() * 0.6 else '#222')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'Pre-arrest']); ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Pre-arrest'], rotation=90, va='center'); ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
    ax.grid(False); tag(ax, '(c) Confusion matrix, MIMIC-IV test')
    ax.text(0, 1.33, 'missed pre-arrest events', ha='center', fontsize=8.5, color='#7b1d12')
    save(fig, 'Figure7.png')


# ================================================================ FIGURE 8
def figure8():
    fig, axs = three()
    auprc_panel(axs[0], ['Multi-Scale GCN'], '(a) AUPRC by test set')
    ax = axs[1]
    p = np.array(DG['f8_scatter'], float); p = p[p[:, 1] < 10]
    ax.scatter(np.arange(1, len(p) + 1), p[:, 1], s=26, color=COL['Multi-Scale GCN'], edgecolor='white',
               lw=0.5, label='True alarm (MS-GCN)')
    ax.axhline(TARGET_H, color='#c0392b', ls='--', lw=1.3, label='6 h target')
    ax.set_ylim(0, 10); ax.set_xlim(0, len(p) + 1)
    ax.set_xlabel('Detected arrest patient (MIMIC-IV test)'); ax.set_ylabel('Lead time (h)')
    ax.legend(frameon=False, loc='upper right'); tag(ax, '(b) Lead times of true alarms')
    REPORT['fig8_scatter'] = dict(n=len(p), median=round(float(np.median(p[:, 1])), 2),
                                  share_ge6=round(float((p[:, 1] >= 6).mean()), 3),
                                  q=[round(float(v), 2) for v in np.percentile(p[:, 1], [25, 75])])
    pr_panel(axs[2], 'f8_pr_', 'MS-GCN', '(c) Precision-recall curves')
    save(fig, 'Figure8.png')


# ================================================================ FIGURE 9
def figure9():
    fig, axs = three()
    ax = axs[0]
    xp, yp = curve('f9_prop', xmax=21.5); xc, yc = curve('f9_cnnlstm', xmax=21.5)
    if not REAL: xp, yp = np.r_[xp, 24], np.r_[yp, yp[-1]]; xc, yc = np.r_[xc, 24], np.r_[yc, yc[-1]]
    ax.plot(xc, yc, color=COL['CNN-LSTM'], lw=2, label='CNN-LSTM')
    ax.plot(xp, yp, color=COL['SCA-QF-MSGT'], lw=2, label='SCA-QF-MSGT')
    traj_axes(ax, '(a) Risk trajectory, one patient')
    ax.legend(handles=ax.get_lines()[:2] + [Line2D([], [], color='black', ls=':', label='Threshold δ = %.2f' % DELTA)],
              frameon=False, loc='upper left')
    REPORT['fig9_cross'] = dict(cnnlstm=crossing(xc, yc, DELTA), prop=crossing(xp, yp, DELTA),
                                cnnlstm_max=round(float(yc.max()), 3))
    lead_box(axs[1], ['CNN-LSTM', 'SCA-QF-MSGT'], '(b) Lead time of true alarms')
    auprc_panel(axs[2], ['CNN-LSTM'], '(c) AUPRC by test set')
    save(fig, 'Figure9.png')


# ================================================================ FIGURE 10
def figure10():
    fig, axs = three((10, 8.8), hspace=0.62)
    auprc_panel(axs[0], MODELS, '(a) AUPRC by test set')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    lead_box(axs[1], MODELS, '(b) Lead time of true alarms')
    ax = axs[2]; cr = {}
    for key, m in (('f10_resnet', 'ResNet-Attention'), ('f10_gcn', 'Multi-Scale GCN'), ('f10_cnnlstm', 'CNN-LSTM')):
        x, y = curve(key, xmax=12, ymax=0.97)
        if not REAL: x = np.r_[24, 12.5, x]; y = np.r_[0.1, 0.1, y]
        o = np.argsort(x); x, y = x[o], y[o]
        ax.plot(x, y, color=COL[m], lw=1.4, label=m); cr[m] = crossing(x, y, DELTA)
    x, y = curve('f10_prop', xmax=21.5)
    if not REAL: x = np.r_[24, x]; y = np.r_[0.11, y]
    o = np.argsort(x); x, y = x[o], y[o]
    ax.plot(x, y, color=COL['SCA-QF-MSGT'], lw=2.2, label='SCA-QF-MSGT'); cr['SCA-QF-MSGT'] = crossing(x, y, DELTA)
    traj_axes(ax, '(c) Risk trajectory, one patient')
    ax.legend(handles=ax.get_lines()[:4] + [Line2D([], [], color='black', ls=':', label='Threshold δ = %.2f' % DELTA)],
              frameon=False, loc='upper left', fontsize=8.5)
    REPORT['fig10_cross'] = cr
    save(fig, 'Figure10.png')


if __name__ == '__main__':
    for f in (figure1, figure2, figure3, figure7, figure8, figure9, figure10):
        try:
            f()
        except KeyError as ex:
            print('%s skipped: no data for %s (run figures/from_predictions.py first)' % (f.__name__, ex))
    json.dump(REPORT, open(os.path.join(OUT, 'report.json'), 'w'), indent=1, default=float)
    print(json.dumps(REPORT, indent=1, default=float))
