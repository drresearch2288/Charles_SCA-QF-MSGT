"""Writes Figure 1 as a draw.io (diagrams.net) file: black and white, Times New Roman bold 13."""
import math, sys
from xml.sax.saxutils import escape

import os
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figure1.drawio')
F = 'fontFamily=Times New Roman;fontStyle=1;fontSize=13;fontColor=#000000;'
cells = []; nid = [1]


def new_id():
    nid[0] += 1; return 'c%d' % nid[0]


def box(x, y, w, h, label, extra='rounded=1;arcSize=8;'):
    i = new_id()
    cells.append('<mxCell id="%s" value="%s" style="html=1;whiteSpace=wrap;fillColor=#FFFFFF;strokeColor=#000000;strokeWidth=1.5;%s%s" vertex="1" parent="1"><mxGeometry x="%g" y="%g" width="%g" height="%g" as="geometry"/></mxCell>'
                 % (i, escape(label, {'"': '&quot;'}), F, extra, x, y, w, h))
    return i


def text(x, y, w, h, label, align='center'):
    return box(x, y, w, h, label, 'text;rounded=0;strokeColor=none;fillColor=none;align=%s;verticalAlign=middle;' % align)


def dot(cx, cy, r=5):
    return box(cx - r, cy - r, 2 * r, 2 * r, '', 'shape=ellipse;perimeter=ellipsePerimeter;fillColor=#000000;')


def line(x0, y0, x1, y1, arrow=False, w=1.5, dash=False, pts=()):
    i = new_id()
    st = 'endArrow=%s;endFill=1;html=1;strokeColor=#000000;strokeWidth=%g;%s' % (
        'block' if arrow else 'none', w, 'dashed=1;' if dash else '')
    p = ''.join('<mxPoint x="%g" y="%g"/>' % q for q in pts)
    cells.append('<mxCell id="%s" value="" style="%s" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 '<mxPoint x="%g" y="%g" as="sourcePoint"/><mxPoint x="%g" y="%g" as="targetPoint"/>%s</mxGeometry></mxCell>'
                 % (i, st, x0, y0, x1, y1, '<Array as="points">%s</Array>' % p if p else ''))


def arrow(x0, y0, x1, y1, **k): line(x0, y0, x1, y1, arrow=True, **k)


R = lambda s: s  # html labels
# ---------------------------------------------------------------- (a) pipeline
text(20, 10, 420, 30, '(a) Pipeline, stages (1)-(10) of Section 3.2')
stages = [
    '(1) Input 12-lead ECG<br>resampled to 510 Hz (Sec. 3.5); 12 × T',
    '(2) Band-pass filter + z-score<br>Eqs. (1)-(2); 12 × T',
    '(3) R-peaks on lead II, 0.6 s beats<br>Eq. (3); 12 × 306 samples per beat',
    '(4) Node features<br>Eq. (4); <i>X</i><sup>(<i>k</i>)</sup> ∈ ℝ<sup>12×4</sup>',
    '(5) Pearson edges, τ = 0.5<br>Eqs. (5)-(6); <i>A</i><sup>(<i>k</i>)</sup> ∈ ℝ<sup>12×12</sup>',
    '(6) 3 edge-weighted GAT layers + readout<br>Eqs. (7)-(12), <i>M</i> heads; <i>z</i><sub>G</sub> ∈ ℝ<sup>64</sup>',
    '(7) Angles + 4-qubit circuit (simulated)<br>Eqs. (13)-(18); <i>q</i> ∈ [−1, 1]<sup>4</sup> → <i>z</i><sub>Q</sub> ∈ ℝ<sup>64</sup>',
    '(8) Sigmoid gate and fusion<br>Eqs. (19)-(21); <i>z</i><sub>F</sub> ∈ ℝ<sup>64</sup>',
    '(9) Classifier + threshold δ<br>Eqs. (22)-(23); one risk score <i>ŷ</i> per beat',
    '(10) Sequence alarm and lead time<br>Sec. 3.12, Eqs. (25a)-(25c); alarm, <i>t̂</i><sub>lead</sub>',
]
X, W, H, Y0, DY = 40, 380, 58, 50, 84
for k, s in enumerate(stages):
    y = Y0 + k * DY
    box(X, y, W, H, s, 'rounded=1;arcSize=8;' + ('dashed=1;' if k == 6 else ''))
    if k: arrow(X + W / 2, y - DY + H, X + W / 2, y)
# skip connection z_G from stage 6 to stage 8
y6 = Y0 + 5 * DY + H / 2; y8 = Y0 + 7 * DY + H / 2
arrow(X + W, y6, X + W, y8, pts=((X + W + 30, y6), (X + W + 30, y8)))
text(X + W + 32, (y6 + y8) / 2 - 12, 40, 24, '<i>z</i><sub>G</sub>', 'left')

# ---------------------------------------------------------------- (b) circuit
BX, BY = 520, 10
text(BX, BY, 760, 30, '(b) 4-qubit circuit, Eqs. (15)-(17)')
qy = [BY + 80 + i * 60 for i in range(4)]
x_wire0, x_wire1 = BX + 60, BX + 760
for i, y in enumerate(qy):
    text(BX - 10, y - 12, 70, 24, 'q%d |0⟩' % (i + 1), 'right')
    line(x_wire0, y, x_wire1, y, w=1.2)
GW, GH = 86, 34


def gate(cx, cy, label, dash=False):
    box(cx - GW / 2, cy - GH / 2, GW, GH, label, 'rounded=0;' + ('dashed=1;' if dash else ''))


for i, y in enumerate(qy):
    gate(BX + 115, y, '<i>R</i><sub>Y</sub>(θ<sub>%d</sub>)' % (i + 1))
for d, x0 in enumerate((BX + 190, BX + 440)):
    box(x0, BY + 45, 230, 255, '', 'rounded=1;arcSize=4;dashed=1;fillColor=none;')
    text(x0, BY + 22, 230, 22, 'layer <i>d</i> = %d' % (d + 1))
    for i, y in enumerate(qy):
        gate(x0 + 55, y, '<i>R</i><sub>Y</sub>(φ<sub>%d</sub><sup>(%d)</sup>)' % (i + 1, d + 1))
    for j, (c, t) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        x = x0 + 120 + j * 30
        line(x, qy[c], x, qy[t] + (11 if t < c else -11), w=1.2)
        dot(x, qy[c])
        box(x - 11, qy[t] - 11, 22, 22, '', 'shape=ellipse;perimeter=ellipsePerimeter;fillColor=#FFFFFF;')
        line(x, qy[t] - 11, x, qy[t] + 11, w=1.2); line(x - 11, qy[t], x + 11, qy[t], w=1.2)
for i, y in enumerate(qy):
    gate(BX + 715, y, '⟨<i>Z</i><sub>%d</sub>⟩' % (i + 1))
text(BX + 20, BY + 315, 200, 44, 'Encoding, Eqs. (13)-(15)<br>θ = π·tanh(<i>W</i><sub>q</sub><i>z</i><sub>G</sub> + <i>b</i><sub>q</sub>)')
text(BX + 240, BY + 315, 380, 44, '<i>D</i> = 2 variational layers, Eq. (16): trainable <i>R</i><sub>Y</sub><br>rotations, then a CNOT ring (q1→q2→q3→q4→q1)')
text(BX + 630, BY + 315, 160, 44, 'Readout, Eq. (17)<br><i>q</i><sub>i</sub> = ⟨<i>Z</i><sub>i</sub>⟩')

# ---------------------------------------------------------------- (c) graph attention
CX, CY, RR = 700, 600, 140
text(CX - 200, 400, 400, 30, '(c) Edge-weighted graph attention')
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
wts = {'I': .75, 'III': .8, 'aVR': .45, 'aVL': .5, 'aVF': .85, 'V1': .3, 'V2': .25, 'V3': .35, 'V4': .5, 'V5': .6, 'V6': .65}
pos = {}
for k, l in enumerate(leads):
    a = math.pi / 2 - k * 2 * math.pi / 12
    pos[l] = (CX + RR * math.cos(a), CY - RR * math.sin(a))
hx, hy = pos['II']
for l, wv in wts.items():
    x, y = pos[l]
    line(hx, hy, x, y, w=round(0.8 + 4.5 * wv, 1), dash=wv < 0.4)
for l, (x, y) in pos.items():
    box(x - 22, y - 22, 44, 44, l, 'shape=ellipse;perimeter=ellipsePerimeter;' + ('strokeWidth=3;' if l == 'II' else ''))
text(CX - 220, CY + 170, 440, 60, 'Edges from lead II; line width shows α<sub>ij</sub> of Eq. (9),<br>which uses the edge weight <i>A</i><sub>ij</sub> (illustrative weights)')

# ---------------------------------------------------------------- (d) fusion
DX = 960
text(DX, 400, 340, 30, '(d) Adaptive fusion and head')
box(DX, 445, 150, 40, '<i>z</i><sub>G</sub> ∈ ℝ<sup>64</sup>')
box(DX + 190, 445, 150, 40, '<i>z</i><sub>Q</sub> ∈ ℝ<sup>64</sup>', 'rounded=1;arcSize=8;dashed=1;')
box(DX, 525, 340, 54, 'Gate, Eqs. (19)-(20)<br><i>g</i> = σ(<i>W</i><sub>g</sub>[<i>z</i><sub>G</sub> ∥ <i>z</i><sub>Q</sub>] + <i>b</i><sub>g</sub>)')
box(DX, 619, 340, 54, 'Fusion, Eq. (21)<br><i>z</i><sub>F</sub> = <i>g</i> ⊙ <i>z</i><sub>Q</sub> + (1 − <i>g</i>) ⊙ <i>z</i><sub>G</sub>')
box(DX, 713, 340, 54, 'Head, Eqs. (22)-(23)<br><i>ŷ</i> = σ(<i>W</i><sub>c</sub><i>z</i><sub>F</sub> + <i>b</i><sub>c</sub>), alarm if <i>ŷ</i> > δ')
arrow(DX + 75, 485, DX + 120, 525); arrow(DX + 265, 485, DX + 220, 525)
arrow(DX + 170, 579, DX + 170, 619); arrow(DX + 170, 673, DX + 170, 713)
text(DX, 790, 340, 40, 'Dashed outline: simulated quantum stage')

xml = ('<mxfile host="app.diagrams.net"><diagram id="fig1" name="Figure 1"><mxGraphModel dx="1400" dy="900" grid="0" '
       'gridSize="10" guides="1" tooltips="1" connect="0" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1320" '
       'pageHeight="900" background="#FFFFFF" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>')
open(OUT, 'w', encoding='utf-8').write(xml)
print(OUT, len(cells), 'cells')
