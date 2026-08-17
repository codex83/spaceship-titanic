import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans"})

ACCENT = "#4C72B0"
ACCENT_FILL = "#EAF0F8"
TEXT_COLOR = "#1a1a1a"
ARROW_COLOR = "#666666"

# grid: 3 columns x 3 rows, snake order
# row1 (top): left-to-right    -> A B C
# row2 (mid): right-to-left    -> D E F  (physically D at col2, E col1, F col0)
# row3 (bot): left-to-right    -> G H I
BOX_W, BOX_H = 2.6, 0.85
COL_X = [0, 3.2, 6.4]
ROW_Y = [4.2, 2.1, 0]

nodes = {
    "A": ("Raw train/test\nCSVs", COL_X[0], ROW_Y[0]),
    "B": ("Feature\nengineering", COL_X[1], ROW_Y[0]),
    "C": ("Group & family\nimputation", COL_X[2], ROW_Y[0]),
    "D": ("Out-of-fold\ntarget encoding", COL_X[2], ROW_Y[1]),
    "E": ("Optuna\nhyperparameter search", COL_X[1], ROW_Y[1]),
    "F": ("Model training\n(LightGBM · XGBoost · CatBoost)", COL_X[0], ROW_Y[1]),
    "G": ("Group-aware\n5-fold CV", COL_X[0], ROW_Y[2]),
    "H": ("3-seed bagged\npredictions", COL_X[1], ROW_Y[2]),
    "I": ("submission.csv", COL_X[2], ROW_Y[2]),
}

edges_within = [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F"), ("G", "H"), ("H", "I")]
edges_between = [("C", "D"), ("F", "G")]

fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.set_xlim(COL_X[0] - BOX_W / 2 - 0.35, COL_X[2] + BOX_W / 2 + 0.35)
ax.set_ylim(-0.6, ROW_Y[0] + BOX_H / 2 + 0.6)
ax.axis("off")
ax.set_aspect("equal")

boxes = {}
for key, (label, x, y) in nodes.items():
    is_final = key == "I"
    box = FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6,
        edgecolor=ACCENT if not is_final else "#C44E52",
        facecolor=ACCENT_FILL if not is_final else "#FBEAEC",
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=10.3,
             color=TEXT_COLOR, zorder=4, linespacing=1.3, fontweight="medium")
    boxes[key] = (x, y)


def draw_arrow(n1, n2, style="-|>", lw=1.6, color=ARROW_COLOR, curve=None, zorder=2):
    x1, y1 = boxes[n1]
    x2, y2 = boxes[n2]
    if abs(y1 - y2) < 1e-6:
        # horizontal
        dx = BOX_W / 2 + 0.06
        start = (x1 + dx if x2 > x1 else x1 - dx, y1)
        end = (x2 - dx if x2 > x1 else x2 + dx, y2)
    else:
        # vertical
        dy = BOX_H / 2 + 0.06
        start = (x1, y1 - dy if y2 < y1 else y1 + dy)
        end = (x2, y2 + dy if y2 < y1 else y2 - dy)
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, zorder=zorder,
        connectionstyle=curve,
    )
    ax.add_patch(arrow)


for n1, n2 in edges_within:
    draw_arrow(n1, n2)
for n1, n2 in edges_between:
    draw_arrow(n1, n2, lw=2.0, color=ACCENT)

# row labels
for y, label in zip(ROW_Y, ["1. Data & features", "2. Tuning & training", "3. Evaluation & output"]):
    ax.text(COL_X[0] - BOX_W / 2, y + BOX_H / 2 + 0.28, label, ha="left", va="bottom",
             fontsize=9.5, color="#888888", fontweight="bold", style="italic")

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.tight_layout()
plt.savefig("assets/pipeline.png", dpi=180, bbox_inches="tight", transparent=True)
print("Saved assets/pipeline.png")
