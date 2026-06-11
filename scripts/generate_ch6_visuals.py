import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, center, width, height, text, facecolor, edgecolor='black'):
    box = patches.Rectangle((center[0] - width/2, center[1] - height/2), width, height, 
                            linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    ax.text(center[0], center[1], text, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3, wrap=True)

def draw_arrow(ax, start, end, text="", rad=0.0):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black', connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, text, ha='center', va='bottom', fontsize=9, color='darkred',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')
fig.suptitle('Figure 6.1: Implementation Workflow of the AGD Pipeline', fontsize=14, fontweight='bold', y=0.95)

# Boxes
draw_box(ax, (5, 8), 4, 0.8, "Start: Load 3D Mesh\n(trimesh)", '#e6f2ff')
draw_box(ax, (5, 6.5), 4, 0.8, "Baseline Analysis\n(discriminator.py)", '#cce5ff')

# Parallel execution simulation
draw_box(ax, (2.5, 4.5), 3, 0.8, "Geometric Critic\n(critic.py)", '#ffe6cc')
draw_box(ax, (7.5, 5.0), 3, 0.8, "Multi-View Render\n(renderer.py)", '#e6ffe6')
draw_box(ax, (7.5, 3.5), 3, 0.8, "LMM & Consistency\n(lmm_detector.py)", '#ccffcc')

draw_box(ax, (5, 2.0), 4, 0.8, "Geometric Grounding\n(grounding.py)", '#ffcc99')
draw_box(ax, (5, 0.5), 4, 0.8, "Optimizer & Save\n(optimizer.py)", '#ff9999')

# Arrows
draw_arrow(ax, (5, 7.6), (5, 6.9))
draw_arrow(ax, (5, 6.1), (2.5, 4.9))
draw_arrow(ax, (5, 6.1), (7.5, 5.4))
draw_arrow(ax, (7.5, 4.6), (7.5, 3.9))
draw_arrow(ax, (2.5, 4.1), (4.5, 2.4))
draw_arrow(ax, (7.5, 3.1), (5.5, 2.4))
draw_arrow(ax, (5, 1.6), (5, 0.9))

# Loop back
draw_arrow(ax, (3.0, 0.5), (2.0, 0.5), rad=0)
draw_arrow(ax, (2.0, 0.5), (2.0, 6.5), rad=0)
draw_arrow(ax, (2.0, 6.5), (3.0, 6.5), rad=0, text="Iterative Refinement")

ax.set_xlim(1, 10)
ax.set_ylim(0, 8.5)

plt.tight_layout()
plt.savefig('Figure_6_1_Workflow.png', dpi=300, bbox_inches='tight')
print("Generated Figure_6_1_Workflow.png")
