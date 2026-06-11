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
    def __init__(self, name: str, page_width: int = 1800, page_height: int = 1100) -> None:
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
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
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
        style = f"rhombus;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=12;align=center;verticalAlign=middle;"
        return self._add_vertex(value, style, x, y, w, h)

    def add_line(self, x: float, y: float, h: float, stroke: str = "#6b7280", dashed: bool = True) -> str:
        style = f"shape=line;html=1;strokeColor={stroke};strokeWidth=1.2;verticalLabelPosition=bottom;verticalAlign=top;"
        if dashed:
            style += "dashed=1;"
        return self._add_vertex("", style, x, y, 1, h)

    def add_edge(
        self,
        source: str,
        target: str,
        label: str = "",
        dashed: bool = False,
        stroke: str = "#111827",
    ) -> str:
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


def build_diagram_5_1() -> DiagramBuilder:
    d = DiagramBuilder("Diagram 5.1", page_width=1900, page_height=980)
    d.add_text("Diagram 5.1 - Top-Level AGD System Architecture", 470, 20, 960, 30, font_size=18, bold=True)

    d.add_container("Preprocessing Module", 30, 80, 450, 720, "#dae8fc", "#6c8ebf")
    pre1 = d.add_box("Load Mesh\n(.obj / .ply / .stl)", 80, 150, 160, 60)
    pre2 = d.add_box("Validate and Normalize\n(center + unit sphere)", 80, 255, 160, 70)
    pre3 = d.add_box("Extract Adjacency Graph\n(NetworkX + edge_index)", 80, 375, 160, 70)
    pre4 = d.add_box("Snapshot Original Vertices\nv_i^(0)", 80, 495, 160, 60)
    d.add_edge(pre1, pre2)
    d.add_edge(pre2, pre3)
    d.add_edge(pre3, pre4)

    d.add_container("ML Engine", 520, 80, 570, 720, "#d5e8d4", "#82b366")
    ml1 = d.add_box("Discriminator\n(topology + spectral +\ngeometry quality)", 570, 150, 190, 80)
    ml2 = d.add_box("Adversarial Critic\n(heuristic z-score or\nGNN GCNConv)", 820, 150, 190, 80)
    ml3 = d.add_box("Multi-View Renderer\n(Fibonacci-sphere,\nN views, 5 buffers)", 570, 310, 190, 80)
    ml4 = d.add_box("LMM Detector\n(LLaVA - per-view\nseverity JSON)", 820, 310, 190, 80)
    ml5 = d.add_box("View Consistency\n(cosine similarity:\nfront-back, left-right,\ntop-bottom)", 690, 470, 210, 95)
    d.add_edge(ml3, ml4)
    d.add_edge(ml3, ml5)

    d.add_container("Extended Module", 1130, 80, 550, 720, "#f8cecc", "#b85450")
    ex1 = d.add_box("Geometric Grounding\n(view scores + critic ->\nvertex weights w_i)", 1230, 180, 260, 80)
    ex2 = d.add_box("Refinement Optimizer\nv_i <- v_i - eta *\n(lambda * grad L_geo +\n(1-lambda) * grad D) * w_i", 1230, 340, 260, 110)
    ex3 = d.add_box("Post-Analysis\n(discriminator +\n10-metric report)", 1230, 530, 260, 80)
    ex4 = d.add_box("Export Refined Mesh\n(*_agd.obj / .ply / .stl)", 1230, 670, 260, 70)
    d.add_edge(ex1, ex2)
    d.add_edge(ex2, ex3)
    d.add_edge(ex3, ex4)

    in_mesh = d.add_box("Input Mesh", 20, 835, 120, 40, fill="#ffffff", stroke="#1f2937")
    out_mesh = d.add_box("Clean 3D Mesh\n+ Geometry Error Report", 1510, 835, 250, 55, fill="#ffffff", stroke="#1f2937")

    d.add_edge(in_mesh, pre1)
    d.add_edge(pre4, ml1)
    d.add_edge(pre4, ml2)
    d.add_edge(pre4, ml3)
    d.add_edge(ml1, ex1, "S_hall")
    d.add_edge(ml2, ex1, "critic_scores [V]")
    d.add_edge(ml4, ex1, "per-view severity + global_bias")
    d.add_edge(ml5, ex1, "consistency severity")
    d.add_edge(ex4, out_mesh)

    d.add_note("All shapes and connectors\nare editable separately\ninside draw.io.", 1540, 120, 200, 80, "#fff2cc", "#d6b656")
    return d


def build_diagram_5_2() -> DiagramBuilder:
    d = DiagramBuilder("Diagram 5.2", page_width=1700, page_height=700)
    d.add_text("Diagram 5.2 - Preprocessing Module Data Flow", 430, 20, 820, 30, font_size=18, bold=True)

    a = d.add_box("Input File\n(.obj / .ply / .stl)", 40, 280, 150, 70, fill="#dae8fc", stroke="#6c8ebf")
    b = d.add_box("trimesh.load()\nforce = \"mesh\"", 240, 280, 170, 70)
    c = d.add_diamond("Scene object?", 470, 275, 120, 80)
    c_yes = d.add_box("Concatenate\nall geometries", 650, 170, 160, 70, fill="#fff2cc", stroke="#d6b656")
    d_valid = d.add_diamond("Valid mesh?\nV > 0 and\nV <= 500k", 870, 260, 130, 95)
    skip = d.add_box("Skip mesh and\nlog warning", 1080, 130, 150, 70, fill="#f8cecc", stroke="#b85450")
    norm = d.add_box("Center and normalize\nto unit bounding sphere", 1080, 280, 180, 80, fill="#d5e8d4", stroke="#82b366")
    adj = d.add_box("Extract nx.Graph\nvertex adjacency", 1320, 215, 160, 70)
    snap = d.add_box("Snapshot original vertices\noriginal_vertices =\nmesh.vertices.copy()", 1320, 330, 190, 90)
    out = d.add_box("Output to ML Engine", 1560, 270, 130, 70, fill="#e1d5e7", stroke="#9673a6")

    d.add_edge(a, b)
    d.add_edge(b, c)
    d.add_edge(c, c_yes, "Yes")
    d.add_edge(c_yes, d_valid)
    d.add_edge(c, d_valid, "No")
    d.add_edge(d_valid, skip, "No")
    d.add_edge(d_valid, norm, "Yes")
    d.add_edge(norm, adj)
    d.add_edge(norm, snap)
    d.add_edge(adj, out)
    d.add_edge(snap, out)
    return d


def build_diagram_5_3() -> DiagramBuilder:
    d = DiagramBuilder("Diagram 5.3", page_width=1900, page_height=1080)
    d.add_text("Diagram 5.3 - ML Engine Internal Architecture", 500, 20, 900, 30, font_size=18, bold=True)
    mesh = d.add_box("Normalized Mesh\n(trimesh.Trimesh)", 840, 80, 180, 60, fill="#dae8fc", stroke="#6c8ebf", bold=True)
    ground = d.add_box("Grounding Module Input", 820, 960, 220, 50, fill="#ffffff", stroke="#1f2937")

    d.add_container("Discriminator - discriminator.py", 40, 170, 520, 740, "#d5e8d4", "#82b366")
    d1 = d.add_box("compute_topology()\nV, E, F -> chi,\ngenus, beta_0", 90, 250, 190, 80)
    d2 = d.add_box("compute_laplacian_metrics()\nL = D - A -> Var(lambda),\nFiedler lambda_2,\ncotangent variance", 320, 235, 190, 110)
    d3 = d.add_box("compute_geometry_quality()\naspect ratio,\ndegenerate faces,\nnon-manifold edges", 90, 420, 190, 100)
    d4 = d.add_box("hallucination_score()\n0.20*P_cc + 0.30*P_genus +\n0.15*P_var + 0.10*P_fiedler +\n0.10*P_cot + 0.15*P_quality", 220, 620, 220, 120, fill="#ffffff", stroke="#1f2937")
    d.add_edge(d1, d4)
    d.add_edge(d2, d4)
    d.add_edge(d3, d4)

    d.add_container("Adversarial Critic - critic.py", 650, 170, 560, 320, "#fff2cc", "#d6b656")
    c1 = d.add_box("Heuristic Critic\ndegree z-score + edge-length z-score\n-> sigmoid blend per vertex", 720, 260, 190, 95)
    c2 = d.add_box("GNN Critic (optional)\n2-layer GCNConv\ninput: [deg, mean_edge_len]\noutput: anomaly score [V]", 950, 245, 200, 110)
    d.add_edge(c1, c2, "--use-gnn", dashed=True, stroke="#6b7280")

    d.add_container("Renderer + Detectors", 650, 530, 560, 380, "#f8cecc", "#b85450")
    r1 = d.add_box("renderer.py\nFibonacci-sphere N views\n5 buffers per view:\nRGB, depth, normal,\nsilhouette, edge_map", 700, 620, 190, 120)
    r2 = d.add_box("view_consistency.py\n32x32 grayscale cosine similarity\nfront-back, left-right,\ntop-bottom", 930, 620, 220, 120)
    r3 = d.add_box("lmm_detector.py\nLLaVA v1.5 / v1.6 local\nJSON: {severity, notes}", 815, 790, 210, 90)
    d.add_edge(r1, r2)
    d.add_edge(r1, r3, "--use-vlm")

    d.add_edge(mesh, d1)
    d.add_edge(mesh, d2)
    d.add_edge(mesh, d3)
    d.add_edge(mesh, c1)
    d.add_edge(mesh, c2)
    d.add_edge(mesh, r1)

    d.add_edge(d4, ground, "S_hall")
    d.add_edge(c1, ground, "critic_scores [V]")
    d.add_edge(c2, ground, "critic_scores [V]")
    d.add_edge(r2, ground, "view_scores dict")
    d.add_edge(r3, ground, "view_scores + global_bias")
    return d


def build_diagram_5_4() -> DiagramBuilder:
    d = DiagramBuilder("Diagram 5.4", page_width=1900, page_height=820)
    d.add_text("Diagram 5.4 - Extended Module: Grounding and Refinement", 460, 20, 980, 30, font_size=18, bold=True)

    signals = d.add_box(
        "Extension Inputs\ncritic_scores [V]\nview_scores dict\nglobal_bias\noriginal_vertices",
        40,
        310,
        180,
        140,
        fill="#ffffff",
        stroke="#1f2937",
        bold=True,
    )

    d.add_container("Geometric Grounding", 280, 150, 340, 460, "#dae8fc", "#6c8ebf")
    g1 = d.add_box("Project multi-view evidence\ninto 3D vertex space", 345, 245, 210, 90)
    g2 = d.add_box("Build vertex weight map\nw_i in [w_min, w_max]", 345, 405, 210, 90)
    d.add_edge(g1, g2)

    d.add_container("Refinement Optimizer", 700, 150, 430, 460, "#d5e8d4", "#82b366")
    o1 = d.add_box("Geometric regularization\nL_geo", 760, 225, 140, 75)
    o2 = d.add_box("Anchor constraint\nD = sum ||v_i - v_i^(0)||^2", 930, 225, 150, 75)
    o3 = d.add_box("Weighted refinement update", 810, 405, 210, 95, fill="#ffffff", stroke="#1f2937", bold=True)
    d.add_edge(o1, o3)
    d.add_edge(o2, o3)

    d.add_container("Post-Refinement Evaluation", 1210, 150, 340, 460, "#fff2cc", "#d6b656")
    e1 = d.add_box("Re-check hallucination score\nand geometry metrics", 1275, 255, 210, 95)
    e2 = d.add_box("Export refined mesh\nand summary report", 1275, 420, 210, 90)
    d.add_edge(e1, e2)

    outputs = d.add_box(
        "Extension Outputs\nrefined mesh\nS_hall_after\ngeometry report",
        1640,
        315,
        180,
        130,
        fill="#ffffff",
        stroke="#1f2937",
        bold=True,
    )

    d.add_edge(signals, g1)
    d.add_edge(signals, o2, "anchor reference")
    d.add_edge(g2, o3, "w_i [V]")
    d.add_edge(o3, e1, "refined mesh")
    d.add_edge(e2, outputs)
    return d


def build_diagram_5_5() -> DiagramBuilder:
    d = DiagramBuilder("Diagram 5.5", page_width=1750, page_height=820)
    d.add_text("Diagram 5.5 - Simplified AGD Data Flow", 400, 20, 950, 30, font_size=18, bold=True)

    inp = d.add_box("Input Mesh", 60, 320, 140, 70, fill="#ffffff", stroke="#1f2937", bold=True)
    pre = d.add_box("Preprocessing\nload, validate,\nnormalize", 280, 300, 180, 110, fill="#dae8fc", stroke="#6c8ebf")

    geo = d.add_box(
        "Geometry Analysis\nDiscriminator + Critic",
        590,
        170,
        210,
        90,
        fill="#e1d5e7",
        stroke="#9673a6",
    )
    view = d.add_box(
        "Multi-View Analysis\nRenderer + View Consistency\n+ optional LMM",
        590,
        430,
        210,
        100,
        fill="#d5e8d4",
        stroke="#82b366",
    )

    gnd = d.add_box(
        "Grounding Module\ncombine analysis signals\n-> weights w_i",
        930,
        285,
        220,
        120,
        fill="#f8cecc",
        stroke="#b85450",
    )
    opt = d.add_box(
        "Refinement Optimizer\nupdate mesh geometry",
        1270,
        300,
        190,
        90,
        fill="#fff2cc",
        stroke="#d6b656",
    )
    eval_box = d.add_box(
        "Evaluation and Export\nhallucination score\n+ geometry report",
        1540,
        290,
        190,
        110,
        fill="#ffffff",
        stroke="#1f2937",
    )
    out = d.add_box("Outputs\nRefined Mesh\n+ Quality Report", 1540, 520, 190, 110, fill="#dae8fc", stroke="#6c8ebf", bold=True)

    note = d.add_note("Optional LMM branch can be\nkept or removed depending\non the experiment setup.", 850, 470, 180, 90)

    d.add_edge(inp, pre)
    d.add_edge(pre, geo)
    d.add_edge(pre, view)
    d.add_edge(geo, gnd, "geometry signals")
    d.add_edge(view, gnd, "view signals")
    d.add_edge(gnd, opt, "weights w_i")
    d.add_edge(opt, eval_box, "refined mesh")
    d.add_edge(eval_box, out)
    return d


def build_combined(diagrams: list[DiagramBuilder], path: Path) -> None:
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "agent": "Codex", "version": "24.7.17"})
    for diagram in diagrams:
        mxfile.append(diagram.diagram)
    ET.ElementTree(mxfile).write(path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    diagrams = [
        ("diagram_5_1_top_level.drawio", build_diagram_5_1()),
        ("diagram_5_2_preprocessing.drawio", build_diagram_5_2()),
        ("diagram_5_3_ml_engine.drawio", build_diagram_5_3()),
        ("diagram_5_4_extended_module.drawio", build_diagram_5_4()),
        ("diagram_5_5_full_dataflow.drawio", build_diagram_5_5()),
    ]

    built = []
    for filename, diagram in diagrams:
        diagram.write(OUT_DIR / filename)
        built.append(diagram)
        print(f"Generated {filename}")

    build_combined(built, OUT_DIR / "Chapter_5_Editable_Diagrams.drawio")
    print("Generated Chapter_5_Editable_Diagrams.drawio")


if __name__ == "__main__":
    main()
