import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "figure.titlesize": 18,
    }
)


def generate_sphere(radius=1.0, center=(0.0, 0.0, 0.0), nu=60, nv=40):
    """Create a smooth sphere surface."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)
    cx, cy, cz = center
    x = radius * np.cos(uu) * np.sin(vv) + cx
    y = radius * np.sin(uu) * np.sin(vv) + cy
    z = radius * np.cos(vv) + cz
    return x, y, z


def generate_spiky_sphere(radius=1.0, nu=72, nv=48):
    """Create a sphere with controlled radial spikes to mimic spectral irregularity."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)

    radial = np.full_like(uu, radius, dtype=float)
    centers = [
        (0.45 * np.pi, 0.25 * np.pi, 0.55),
        (1.10 * np.pi, 0.70 * np.pi, 0.40),
        (1.65 * np.pi, 0.45 * np.pi, 0.45),
        (0.20 * np.pi, 0.85 * np.pi, 0.35),
        (1.75 * np.pi, 0.92 * np.pi, 0.30),
    ]
    sigma = 0.07
    for cu, cv, amp in centers:
        radial += amp * np.exp(-(((uu - cu) ** 2) + ((vv - cv) ** 2)) / sigma)

    x = radial * np.cos(uu) * np.sin(vv)
    y = radial * np.sin(uu) * np.sin(vv)
    z = radial * np.cos(vv)
    return x, y, z


def wrapped_angle_diff(a, b):
    diff = np.abs(a - b)
    return np.minimum(diff, 2 * np.pi - diff)


def angular_gaussian(uu, vv, u0, v0, sigma_u=0.16, sigma_v=0.16):
    du = wrapped_angle_diff(uu, u0)
    dv = np.abs(vv - v0)
    return np.exp(-0.5 * ((du / sigma_u) ** 2 + (dv / sigma_v) ** 2))


def generate_janus_mesh(nu=120, nv=80):
    """Create an illustrative head-like mesh with a duplicated facial bump."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)

    front_face = angular_gaussian(uu, vv, 0.08, np.pi / 2, sigma_u=0.16, sigma_v=0.14)
    janus_face = angular_gaussian(uu, vv, np.pi - 0.10, np.pi / 2, sigma_u=0.19, sigma_v=0.16)
    cheek = angular_gaussian(uu, vv, 0.45, np.pi / 2 + 0.22, sigma_u=0.25, sigma_v=0.20)
    forehead = angular_gaussian(uu, vv, 0.05, np.pi / 2 - 0.55, sigma_u=0.30, sigma_v=0.25)

    radial = 1.0 + 0.18 * front_face + 0.14 * janus_face + 0.05 * cheek + 0.04 * forehead
    x = 0.96 * radial * np.cos(uu) * np.sin(vv)
    y = 0.82 * radial * np.sin(uu) * np.sin(vv)
    z = 1.10 * radial * np.cos(vv)

    lmm_projection = angular_gaussian(uu, vv, np.pi - 0.10, np.pi / 2, sigma_u=0.24, sigma_v=0.22)
    gcn_critic = 0.65 * janus_face + 0.25 * angular_gaussian(
        uu, vv, np.pi - 0.28, np.pi / 2 + 0.08, sigma_u=0.22, sigma_v=0.18
    )
    weights = 0.55 * lmm_projection + 0.45 * gcn_critic
    weights = weights / np.max(weights)
    return x, y, z, weights


def style_3d_axis(ax, elev=20, azim=35, lim=1.85):
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_axis_off()


def add_metric_box(ax, heading, lines, edge_color):
    ax.axis("off")
    box = FancyBboxPatch(
        (0.05, 0.10),
        0.90,
        0.78,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor="white",
        edgecolor=edge_color,
        linewidth=2,
    )
    ax.add_patch(box)
    ax.text(0.5, 0.72, heading, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.38, "\n".join(lines), ha="center", va="center", fontsize=12)


def draw_box(ax, x, y, w, h, text, facecolor, edgecolor="#1f2937", fontsize=10.5):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")


def draw_arrow(ax, start, end, color="#111827", lw=2.4, style="-|>", connection="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        linewidth=lw,
        color=color,
        connectionstyle=connection,
        mutation_scale=18,
    )
    ax.add_patch(arrow)


def add_label(ax, x, y, text, color="#111827", fontsize=10, bbox=False):
    kwargs = {}
    if bbox:
        kwargs["bbox"] = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, linewidth=1.5)
    ax.text(x, y, text, ha="center", va="center", color=color, fontsize=fontsize, fontweight="bold", **kwargs)


def create_geometry_grid():
    fig = plt.figure(figsize=(15, 6.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[4.0, 1.20], wspace=0.06, hspace=0.02)

    # Panel 1: clean sphere
    ax1 = fig.add_subplot(grid[0, 0], projection="3d")
    x, y, z = generate_sphere()
    ax1.plot_surface(x, y, z, color="#b7dceb", edgecolor="#111827", linewidth=0.35, alpha=0.96, shade=True)
    ax1.set_title("Clean Mesh", fontweight="bold", pad=4)
    style_3d_axis(ax1, elev=18, azim=35, lim=1.3)

    # Panel 2: disconnected duplicate component
    ax2 = fig.add_subplot(grid[0, 1], projection="3d")
    x_big, y_big, z_big = generate_sphere(radius=0.95, center=(-0.15, 0.0, 0.0))
    x_small, y_small, z_small = generate_sphere(radius=0.46, center=(1.15, 0.25, 0.10))
    ax2.plot_surface(x_big, y_big, z_big, color="#f5a0a0", edgecolor="#111827", linewidth=0.35, alpha=0.96, shade=True)
    ax2.plot_surface(x_small, y_small, z_small, color="#f5a0a0", edgecolor="#111827", linewidth=0.35, alpha=0.96, shade=True)
    ax2.set_title("Disconnected Duplication", fontweight="bold", pad=4)
    style_3d_axis(ax2, elev=18, azim=35, lim=1.8)

    # Panel 3: spiky geometry
    ax3 = fig.add_subplot(grid[0, 2], projection="3d")
    x_spike, y_spike, z_spike = generate_spiky_sphere()
    ax3.plot_surface(x_spike, y_spike, z_spike, color="#f5a0a0", edgecolor="#111827", linewidth=0.35, alpha=0.96, shade=True)
    ax3.set_title("Spiky Irregularity", fontweight="bold", pad=4)
    style_3d_axis(ax3, elev=18, azim=35, lim=1.9)

    # Metrics row
    box1 = fig.add_subplot(grid[1, 0])
    add_metric_box(
        box1,
        "Expected Behaviour",
        [r"Betti number $\beta_0 = 1$", r"Fiedler value $\lambda_2 > 0$", r"Eigenvalue variance: low"],
        edge_color="#2563eb",
    )

    box2 = fig.add_subplot(grid[1, 1])
    add_metric_box(
        box2,
        "Floating Hallucination",
        [r"Betti number $\beta_0 = 2$", r"Fiedler value $\lambda_2 = 0$", r"Eigenvalue variance: moderate"],
        edge_color="#dc2626",
    )

    box3 = fig.add_subplot(grid[1, 2])
    add_metric_box(
        box3,
        "Surface Irregularity",
        [r"Betti number $\beta_0 = 1$", r"Fiedler value $\lambda_2 > 0$", r"Eigenvalue variance: high"],
        edge_color="#dc2626",
    )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.06)
    fig.savefig("Figure_3_2_Geometry_Grid.png", dpi=300, bbox_inches="tight")
    fig.savefig("Figure_3_2_Geometry_Grid.svg", bbox_inches="tight")
    plt.close(fig)


def create_block_diagram():
    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Background panels
    left_panel = Rectangle((0.5, 0.65), 6.8, 8.35, facecolor="#f5f9ff", edgecolor="#9ca3af", linestyle="--", linewidth=2)
    right_panel = Rectangle((8.7, 0.65), 6.8, 8.35, facecolor="#fffaf3", edgecolor="#9ca3af", linestyle="--", linewidth=2)
    ax.add_patch(left_panel)
    ax.add_patch(right_panel)
    ax.text(3.9, 8.65, "Traditional Pipeline\n(Open-loop SDS)", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(12.1, 8.65, "Proposed Framework\n(Closed-loop AGD)", ha="center", va="center", fontsize=15, fontweight="bold")

    # Left column boxes
    left_x, box_w, box_h = 1.55, 4.75, 0.85
    ys = [7.55, 6.15, 4.75, 3.35, 1.95]
    draw_box(ax, left_x, ys[0], box_w, box_h, "Text Prompt (y)", "#dbeafe")
    draw_box(ax, left_x, ys[1], box_w, box_h, "2D Diffusion Model\n(Prior)", "#bfdbfe")
    draw_box(ax, left_x, ys[2], box_w, box_h, "Score Distillation Sampling\n(2D guidance only)", "#93c5fd")
    draw_box(ax, left_x, ys[3], box_w, box_h, "3D Optimizer\n(Unconstrained)", "#fecaca")
    draw_box(ax, left_x, ys[4], box_w, box_h, "3D Mesh Output\n(Hallucination-prone)", "#fca5a5")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        draw_arrow(ax, (left_x + box_w / 2, y0), (left_x + box_w / 2, y1 + box_h))
    add_label(ax, 3.93, 1.2, "No explicit geometric feedback", color="#b91c1c", fontsize=11, bbox=True)

    # Right column core flow
    right_x = 9.55
    draw_box(ax, right_x, 7.55, box_w, box_h, "Text Prompt (y)", "#dbeafe")
    draw_box(ax, right_x, 6.15, box_w, box_h, "2D Diffusion Model\n(SDS guidance)", "#bfdbfe")
    draw_box(ax, right_x, 4.75, box_w, box_h, "3D Optimizer\n(with grounded constraint)", "#bbf7d0")
    draw_box(ax, right_x, 3.35, box_w, box_h, "Intermediate 3D Mesh", "#fed7aa")
    draw_box(ax, right_x, 1.95, box_w, box_h, "Refined 3D Mesh\n(Geometrically consistent)", "#86efac")
    for y0, y1 in [(7.55, 6.15), (6.15, 4.75), (4.75, 3.35), (3.35, 1.95)]:
        draw_arrow(ax, (right_x + box_w / 2, y0), (right_x + box_w / 2, y1 + box_h))

    # Side feedback loop
    aux_x, aux_w = 13.0, 2.1
    draw_box(ax, aux_x, 6.00, aux_w, 0.72, "Multi-view\nRenderer", "#fef3c7", fontsize=10)
    draw_box(ax, aux_x, 4.95, aux_w, 0.82, "LMM Detector +\nGeometric Critic", "#fdba74", fontsize=10)
    draw_box(ax, aux_x, 3.85, aux_w, 0.82, "Geometric Grounding\n(Vertex weights)", "#fb923c", fontsize=10)

    draw_arrow(ax, (right_x + box_w, 3.77), (aux_x, 6.36), color="#2563eb", connection="arc3,rad=0.18")
    draw_arrow(ax, (aux_x + aux_w / 2, 5.99), (aux_x + aux_w / 2, 5.77), color="#2563eb")
    draw_arrow(ax, (aux_x + aux_w / 2, 4.95), (aux_x + aux_w / 2, 4.67), color="#2563eb")
    draw_arrow(ax, (aux_x, 4.16), (right_x + box_w, 5.18), color="#dc2626", connection="arc3,rad=-0.20")

    add_label(ax, 14.25, 7.1, "Rendered views", color="#2563eb", fontsize=10)
    add_label(ax, 14.25, 4.45, "Spatially grounded feedback", color="#dc2626", fontsize=10, bbox=True)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.05)
    fig.savefig("Figure_3_1_Block_Diagram.png", dpi=300, bbox_inches="tight")
    fig.savefig("Figure_3_1_Block_Diagram.svg", bbox_inches="tight")
    plt.close(fig)


def create_spatial_grounding_figure():
    fig = plt.figure(figsize=(13.8, 6.6))
    grid = fig.add_gridspec(1, 2, wspace=0.05)

    ax1 = fig.add_subplot(grid[0, 0], projection="3d")
    ax2 = fig.add_subplot(grid[0, 1], projection="3d")

    x, y, z, weights = generate_janus_mesh()

    neutral = np.empty(weights.shape + (4,))
    neutral[:] = np.array([0.79, 0.84, 0.89, 1.0])
    heat_colors = cm.inferno(weights)

    ax1.plot_surface(x, y, z, facecolors=neutral, edgecolor="#111827", linewidth=0.28, shade=False)
    ax2.plot_surface(x, y, z, facecolors=heat_colors, edgecolor="#111827", linewidth=0.28, shade=False)

    style_3d_axis(ax1, elev=16, azim=128, lim=1.55)
    style_3d_axis(ax2, elev=16, azim=128, lim=1.55)

    ax1.set_title("Base Mesh", fontweight="bold", pad=10)
    ax2.set_title("Spatial Grounding Heat Map", fontweight="bold", pad=10)

    ax1.text2D(
        0.50,
        0.06,
        "Illustrative Janus-like mesh\nwith duplicated facial region",
        transform=ax1.transAxes,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#9ca3af"),
    )
    ax2.text2D(
        0.50,
        0.06,
        r"Hot colors indicate grounded vertex weights $(w_i \approx w_{max})$",
        transform=ax2.transAxes,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#dc2626"),
    )
    ax2.text2D(
        0.72,
        0.78,
        "Targeted\nhallucination region",
        transform=ax2.transAxes,
        ha="center",
        fontsize=10.5,
        color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#f97316"),
    )

    sm = cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="inferno")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.040, pad=0.025)
    cbar.set_label("Vertex weight $w_i$", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0", "0.5", r"$w_{max}$"])

    fig.text(
        0.5,
        0.02,
        "Conceptual rendering of spatial grounding: vertex weights are formed by combining GCN critic scores\nwith multi-view LMM severity projections to localise the hallucinated region before refinement.",
        ha="center",
        fontsize=11,
        color="#374151",
    )

    fig.subplots_adjust(left=0.03, right=0.95, top=0.90, bottom=0.12)
    fig.savefig("Figure_3_3_Spatial_Grounding.png", dpi=300, bbox_inches="tight")
    fig.savefig("Figure_3_3_Spatial_Grounding.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    create_block_diagram()
    create_geometry_grid()
    create_spatial_grounding_figure()
    print("Generated Figure_3_1_Block_Diagram.(png|svg)")
    print("Generated Figure_3_2_Geometry_Grid.(png|svg)")
    print("Generated Figure_3_3_Spatial_Grounding.(png|svg)")
