from __future__ import annotations

import html
import xml.etree.ElementTree as ET
from pathlib import Path


OUT_DIR = Path(".")


class IdGen:
    def __init__(self) -> None:
        self.value = 2

    def next(self) -> str:
        current = str(self.value)
        self.value += 1
        return current


class DiagramBuilder:
    def __init__(self, name: str, page_width: int = 2200, page_height: int = 1400) -> None:
        self.name = name
        self.ids = IdGen()
        self.diagram = ET.Element("diagram", {"id": f"{name.lower().replace(' ', '-')}", "name": name})
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            {
                "dx": "1600",
                "dy": "900",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(page_width),
                "pageHeight": str(page_height),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def _add_vertex(self, value: str, style: str, x: float, y: float, w: float, h: float) -> str:
        cell_id = self.ids.next()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": html.escape(value, quote=True),
                "style": style,
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )
        return cell_id

    def add_text(self, value: str, x: float, y: float, w: float, h: float, font_size: int = 14, bold: bool = False) -> str:
        style = (
            "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
            f"fontSize={font_size};"
        )
        if bold:
            style += "fontStyle=1;"
        return self._add_vertex(value, style, x, y, w, h)

    def add_box(
        self,
        value: str,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "#ffffff",
        stroke: str = "#1f2937",
        rounded: bool = True,
        font_size: int = 12,
        bold: bool = False,
    ) -> str:
        style = (
            f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};fontSize={font_size};"
            "align=center;verticalAlign=middle;"
        )
        if bold:
            style += "fontStyle=1;"
        return self._add_vertex(value, style, x, y, w, h)

    def add_container(self, title: str, x: float, y: float, w: float, h: float, fill: str, stroke: str) -> tuple[str, str]:
        outer = self._add_vertex(
            "",
            f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.2;",
            x,
            y,
            w,
            h,
        )
        title_id = self.add_text(title, x + 10, y + 8, w - 20, 28, font_size=14, bold=True)
        return outer, title_id

    def add_note(self, value: str, x: float, y: float, w: float, h: float, fill: str = "#fff2cc", stroke: str = "#d6b656") -> str:
        style = f"shape=note;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=11;"
        return self._add_vertex(value, style, x, y, w, h)

    def add_diamond(self, value: str, x: float, y: float, w: float, h: float, fill: str = "#ffffff", stroke: str = "#1f2937") -> str:
        style = (
            f"rhombus;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            "fontSize=12;align=center;verticalAlign=middle;"
        )
        return self._add_vertex(value, style, x, y, w, h)

    def add_edge(self, source: str, target: str, label: str = "", dashed: bool = False, stroke: str = "#111827") -> str:
        cell_id = self.ids.next()
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
            f"html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor={stroke};"
        )
        if dashed:
            style += "dashed=1;"
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": html.escape(label, quote=True),
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        return cell_id

    def write(self, path: Path) -> None:
        mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "agent": "Codex", "version": "24.7.17"})
        mxfile.append(self.diagram)
        ET.ElementTree(mxfile).write(path, encoding="utf-8", xml_declaration=False)


def build_figure_6_1() -> DiagramBuilder:
    d = DiagramBuilder("Figure 6.1")
    d.add_text("Figure 6.1 - AGD Implementation Workflow", 610, 20, 980, 30, font_size=18, bold=True)

    d.add_container("Inputs and Control", 40, 90, 360, 320, "#dae8fc", "#6c8ebf")
    mesh_in = d.add_box("Input Mesh\n.obj / .ply / .stl", 85, 170, 135, 70, fill="#ffffff", stroke="#1f2937", bold=True)
    prompt_in = d.add_box("Optional Text Prompt Y\nfor LMM-based inspection", 85, 280, 135, 75, fill="#ffffff", stroke="#1f2937")
    cli = d.add_box("agd_pipeline.py\nCLI orchestration", 245, 205, 120, 95, fill="#ffffff", stroke="#1f2937", bold=True)
    args = d.add_box("Hyperparameters\neta, lambda, N,\niterations, --use-vlm", 245, 320, 120, 75, fill="#fff2cc", stroke="#d6b656")
    d.add_edge(mesh_in, cli)
    d.add_edge(prompt_in, cli, "optional")
    d.add_edge(args, cli, "config")

    d.add_container("Preprocessing Stage", 450, 90, 430, 320, "#d5e8d4", "#82b366")
    load_mesh = d.add_box("Load mesh with trimesh\nforce = \"mesh\"", 500, 155, 150, 75)
    validate = d.add_box("Validate and concatenate\nscene geometry if needed", 680, 155, 150, 75)
    normalize = d.add_box("Normalize geometry\ncenter + unit scale", 500, 275, 150, 75)
    graph = d.add_box("Extract adjacency graph\nand snapshot V0", 680, 275, 150, 75)
    prepared = d.add_box("Prepared Mesh\nmesh + adj_graph + V0", 590, 365, 160, 55, fill="#ffffff", stroke="#1f2937", bold=True)
    d.add_edge(load_mesh, validate)
    d.add_edge(load_mesh, normalize)
    d.add_edge(validate, graph)
    d.add_edge(normalize, graph)
    d.add_edge(graph, prepared)

    d.add_container("Parallel ML Engine Analysis", 930, 90, 1200, 520, "#e1d5e7", "#9673a6")
    disc = d.add_box(
        "Discriminator\ncompute topology, Laplacian,\nEuler/genus, Fiedler,\nvariance, geometry quality",
        980,
        170,
        220,
        125,
        fill="#ffffff",
        stroke="#1f2937",
    )
    disc_out = d.add_box("Global structure signals\nS_hall_before\nspectral + topology scores", 1010, 335, 160, 95, fill="#fff2cc", stroke="#d6b656")

    critic = d.add_box(
        "Adversarial Critic\nheuristic z-scores or GNN\nper-vertex anomaly scoring",
        1240,
        170,
        220,
        125,
        fill="#ffffff",
        stroke="#1f2937",
    )
    critic_out = d.add_box("critic_scores [V]", 1290, 340, 120, 70, fill="#fff2cc", stroke="#d6b656")

    render = d.add_box(
        "Renderer\nFibonacci-sphere sampling\nRGB, depth, normal,\nsilhouette, edge buffers",
        1510,
        170,
        220,
        125,
        fill="#ffffff",
        stroke="#1f2937",
    )
    consistency = d.add_box("View Consistency\ncosine similarity\nacross paired views", 1505, 350, 145, 85, fill="#d5e8d4", stroke="#82b366")
    lmm = d.add_box("LMM Detector\nLLaVA severity JSON\noptional branch", 1685, 345, 140, 90, fill="#f8cecc", stroke="#b85450")
    view_out = d.add_box("View-level signals\nview_scores + global_bias", 1870, 335, 165, 95, fill="#fff2cc", stroke="#d6b656")

    d.add_edge(disc, disc_out)
    d.add_edge(critic, critic_out)
    d.add_edge(render, consistency)
    d.add_edge(render, lmm, "--use-vlm", dashed=True, stroke="#b85450")
    d.add_edge(consistency, view_out, "consistency")
    d.add_edge(lmm, view_out, "severity", dashed=True, stroke="#b85450")

    d.add_container("Grounding and Optimization", 930, 660, 760, 610, "#f8cecc", "#b85450")
    grounding = d.add_box(
        "grounding.py\nproject view evidence to vertices\nand fuse with critic scores",
        1000,
        760,
        250,
        100,
        fill="#ffffff",
        stroke="#1f2937",
        bold=True,
    )
    weights = d.add_box("Weight Map W\nvertex weights w_i in [w_min, w_max]", 1310, 775, 240, 70, fill="#fff2cc", stroke="#d6b656")

    loop_outer, _ = d.add_container("Iterative Refinement Loop", 980, 900, 640, 320, "#ffffff", "#1f2937")
    anchor = d.add_box("Initialize anchor\nV0 <- original vertices", 1020, 970, 150, 75)
    lap = d.add_box("Compute Laplacian\nnabla L_geo", 1210, 955, 140, 75)
    distill = d.add_box("Compute distillation term\nnabla D = (V - V0)", 1380, 955, 160, 75)
    update = d.add_box("Weighted update\nV <- V - eta*(lambda*nabla L_geo + (1-lambda)*nabla D) * W", 1130, 1085, 300, 85, fill="#d5e8d4", stroke="#82b366", bold=True)
    iter_check = d.add_diamond("Iterations complete?", 1475, 1075, 110, 90, fill="#ffffff", stroke="#1f2937")
    refined = d.add_box("Refined Mesh M1", 1110, 1230, 180, 55, fill="#ffffff", stroke="#1f2937", bold=True)

    d.add_edge(anchor, update, "anchor")
    d.add_edge(lap, update)
    d.add_edge(distill, update)
    d.add_edge(update, iter_check)
    d.add_edge(iter_check, lap, "No")
    d.add_edge(iter_check, refined, "Yes")

    d.add_container("Evaluation, Export, and Development Testing", 1750, 660, 380, 610, "#dae8fc", "#6c8ebf")
    rerun = d.add_box("Re-run Discriminator\non refined mesh", 1820, 750, 240, 80, fill="#ffffff", stroke="#1f2937")
    metrics = d.add_box("geometry_metrics.py\nCD, HD95, NCS, SSIM,\nDepth RMSE, Edge-IoU,\nCurvature W1, and others", 1820, 865, 240, 120, fill="#ffffff", stroke="#1f2937")
    export = d.add_box("Export outputs\n*_agd mesh + summary table", 1820, 1020, 240, 80, fill="#d5e8d4", stroke="#82b366", bold=True)
    final_out = d.add_box("Final Outputs\nRefined Mesh\nHallucination Scores\nGeometry Report", 1820, 1135, 240, 100, fill="#fff2cc", stroke="#d6b656", bold=True)
    smoke = d.add_note(
        "Development validation:\n3d_samples/ test set\n+ smoke_test.py\nbypassing heavy LLaVA\nfor fast regression checks.",
        1775,
        660,
        160,
        120,
    )
    d.add_edge(rerun, metrics)
    d.add_edge(metrics, export)
    d.add_edge(export, final_out)

    d.add_edge(cli, load_mesh, "start pipeline")
    d.add_edge(prepared, disc, "prepared mesh")
    d.add_edge(prepared, critic, "prepared mesh")
    d.add_edge(prepared, render, "prepared mesh")
    d.add_edge(prepared, grounding, "mesh + V0")
    d.add_edge(disc_out, grounding, "global scores")
    d.add_edge(critic_out, grounding, "local scores")
    d.add_edge(view_out, grounding, "view evidence")
    d.add_edge(grounding, weights, "fused W")
    d.add_edge(weights, update, "weights")
    d.add_edge(prepared, anchor, "snapshot")
    d.add_edge(refined, rerun, "refined mesh")
    d.add_edge(refined, metrics, "evaluate")

    return d


def main() -> None:
    figure = build_figure_6_1()
    figure.write(OUT_DIR / "Figure_6_1_Workflow.drawio")
    figure.write(OUT_DIR / "Chapter_6_Editable_Diagrams.drawio")
    print("Generated Figure_6_1_Workflow.drawio")
    print("Generated Chapter_6_Editable_Diagrams.drawio")


if __name__ == "__main__":
    main()
