"""Build the one-page reading PDF and a results presentation."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
OUTPUTS = ROOT / "outputs"
NAVY = RGBColor(17, 34, 64)
TEAL = RGBColor(43, 196, 177)
WHITE = RGBColor(248, 250, 252)
MUTED = RGBColor(184, 199, 218)


def reading_pdf() -> None:
    source = DELIVERABLES / "reactive_optimizers_summary.md"
    lines = source.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ").strip()
    paragraphs = [line.strip() for line in lines[1:] if line.strip()]
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SummaryTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, alignment=TA_LEFT, textColor=colors.HexColor("#112240"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "SummaryBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=10.4, leading=14.1, alignment=TA_LEFT,
        textColor=colors.HexColor("#263445"), spaceAfter=8,
    )
    story = [Paragraph(html.escape(title), title_style), Spacer(1, 2 * mm)]
    for paragraph in paragraphs:
        text = html.escape(paragraph)
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            r'<link href="\2" color="#147D78">\1</link>',
            text,
        )
        story.append(Paragraph(text, body_style))
    path = DELIVERABLES / "reactive_optimizers_summary.pdf"
    document = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title="Reactive Optimizers: a short summary",
        author="Federated Learning Screening Task",
    )
    document.build(story)


def read_result(run_name: str) -> dict | None:
    path = OUTPUTS / run_name / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_rows(run_name: str) -> list[dict[str, str]]:
    path = OUTPUTS / run_name / "metrics.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add_background(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = NAVY


def add_text(slide, text: str, x: float, y: float, w: float, h: float,
             size: int, color=WHITE, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color
    return box


def add_slide(prs: Presentation, title: str, section: str = "SCREENING TASK"):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_text(slide, section.upper(), 0.7, 0.28, 5.0, 0.3, 10, TEAL, True)
    add_text(slide, title, 0.7, 0.72, 12.0, 0.6, 27, WHITE, True)
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.42), Inches(0.75), Inches(0.055)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()
    add_text(slide, "PathMNIST  |  5 clients  |  Dirichlet alpha = 0.3  |  seed 0",
             0.7, 7.12, 12.0, 0.22, 9, MUTED)
    return slide


def add_bullets(slide, items: list[str], x=0.95, y=1.8, w=11.4,
                font_size=20, gap=0.82) -> None:
    for index, item in enumerate(items):
        add_text(slide, "•", x, y + index * gap, 0.4, 0.5, font_size + 2, TEAL, True)
        add_text(slide, item, x + 0.42, y + index * gap, w - 0.42, 0.62,
                 font_size, WHITE)


def make_comparison_plot(fedavg: list[dict], fedprox: list[dict]) -> Path | None:
    if not fedavg or not fedprox:
        return None
    path = DELIVERABLES / "global_accuracy_comparison.png"
    fig, (axis, client_axis) = plt.subplots(
        1, 2, figsize=(11.0, 4.5), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )
    for label, rows in (("FedAvg", fedavg), ("FedProx", fedprox)):
        rounds = [int(row["round"]) for row in rows]
        accuracy = [float(row["global/test_accuracy"]) for row in rows]
        axis.plot(rounds, accuracy, marker="o", markersize=3, linewidth=2, label=label)
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Accuracy")
    axis.set_title("Global test accuracy")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    client_ids = range(5)
    positions = list(client_ids)
    width = 0.36
    for offset, label, rows, color in (
        (-width / 2, "FedAvg", fedavg, "#087E8B"),
        (width / 2, "FedProx", fedprox, "#FF5A5F"),
    ):
        final = rows[-1]
        values = [float(final[f"client/client_{client_id}/test_accuracy"]) for client_id in client_ids]
        client_axis.bar([position + offset for position in positions], values,
                        width=width, label=label, color=color)
    client_axis.set_xticks(positions, [f"C{index}" for index in client_ids])
    client_axis.set_xlabel("Client")
    client_axis.set_ylabel("Accuracy")
    client_axis.set_title("Final per-client accuracy")
    client_axis.set_ylim(0, 1)
    client_axis.grid(axis="y", alpha=0.25)
    client_axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, transparent=False)
    plt.close(fig)
    return path


def write_final_metrics_csv(avg: dict | None, prox: dict | None) -> None:
    rows = [("global", "global/test_accuracy"), ("global", "global/test_loss")]
    rows.extend(
        (f"client_{client_id}", f"client/client_{client_id}/{metric}")
        for client_id in range(5)
        for metric in ("test_accuracy", "test_loss")
    )
    path = DELIVERABLES / "final_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("evaluation_group", "metric", "fedavg_seed0", "fedprox_mu0.01_seed0"))
        for group, key in rows:
            avg_value = avg["final_round"].get(key, "") if avg else ""
            prox_value = prox["final_round"].get(key, "") if prox else ""
            writer.writerow((group, key.rsplit("/", 1)[-1], avg_value, prox_value))


def format_accuracy(value) -> str:
    return "Pending" if value is None else f"{100 * float(value):.1f}%"


def presentation() -> None:
    avg = read_result("fedavg_seed0")
    prox = read_result("fedprox_mu0.01_seed0")
    avg_rows = read_rows("fedavg_seed0")
    prox_rows = read_rows("fedprox_mu0.01_seed0")
    comparison = make_comparison_plot(avg_rows, prox_rows)
    write_final_metrics_csv(avg, prox)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_text(slide, "FEDERATED LEARNING  /  PATHMNIST", 0.85, 0.85, 10.8, 0.4,
             13, TEAL, True)
    add_text(slide, "FedAvg vs FedProx", 0.85, 1.7, 11.3, 0.9, 38, WHITE, True)
    add_text(slide, "Non-IID medical image classification with client label noise",
             0.9, 2.8, 10.8, 0.6, 22, MUTED)
    add_text(slide, "5 clients  •  Dirichlet α = 0.3  •  40% noisy labels at client 0",
             0.9, 4.1, 10.8, 0.55, 18, WHITE)
    add_text(slide, "Screening task project overview and experiment results",
             0.9, 6.8, 11.5, 0.3, 12, MUTED)

    slide = add_slide(prs, "Question and benchmark")
    add_bullets(slide, [
        "Compare two standard federated optimization baselines under the same setup.",
        "PathMNIST is a 9-class, 28×28 RGB histology image benchmark.",
        "The experiment introduces client heterogeneity and a single noisy-label client.",
        "This is a controlled research exercise, not a clinical model evaluation.",
    ])

    slide = add_slide(prs, "Federated data setup")
    add_bullets(slide, [
        "5 clients receive non-IID training examples via class-wise Dirichlet allocation.",
        "Alpha 0.3 creates different label mixtures across clients.",
        "Client 0 has 40% of its training labels changed to a uniformly sampled wrong class.",
        "Global and client test sets keep their original labels; client test subsets are disjoint.",
    ])

    slide = add_slide(prs, "The two baselines")
    add_text(slide, "FedAvg", 1.0, 1.9, 4.5, 0.55, 25, TEAL, True)
    add_text(slide, "Each client trains locally; the server averages client models weighted by local sample count.",
             1.0, 2.55, 4.8, 1.8, 20, WHITE)
    add_text(slide, "FedProx", 7.0, 1.9, 4.5, 0.55, 25, TEAL, True)
    add_text(slide, "Adds μ/2 · ||w − w_global||² to the local objective to keep client models near the current global model.",
             7.0, 2.55, 4.8, 1.8, 20, WHITE)
    add_text(slide, "FedProx μ = 0.01. Both methods use the same model, partition, seed, and training settings.",
             1.0, 5.4, 11.2, 0.75, 17, MUTED)

    slide = add_slide(prs, "What we track each round")
    add_bullets(slide, [
        "Global test accuracy and loss on the full official test set.",
        "Per-client test accuracy and loss on clean, client-specific test subsets.",
        "Per-client training loss and local update L2 norm.",
        "Cosine similarity between client updates, plus each client's mean similarity to peers.",
        "W&B stores the run history; CSV and JSON summaries are also saved locally.",
    ], font_size=18, gap=0.8)

    slide = add_slide(prs, "Global accuracy by round", "RESULTS")
    if comparison:
        slide.shapes.add_picture(str(comparison), Inches(1.65), Inches(1.7), width=Inches(10.0))
    else:
        add_text(slide, "Results are pending. Run both methods to build the comparison chart.",
                 1.0, 2.2, 11.0, 1.0, 22, MUTED)

    slide = add_slide(prs, "Final accuracy summary", "RESULTS")
    table = slide.shapes.add_table(7, 3, Inches(1.0), Inches(1.75), Inches(11.1), Inches(4.6)).table
    for col, width in enumerate((3.5, 3.8, 3.8)):
        table.columns[col].width = Inches(width)
    cells = [
        ["Evaluation group", "FedAvg", "FedProx (μ=0.01)"],
        ["Global", format_accuracy(avg["final_round"]["global/test_accuracy"] if avg else None),
         format_accuracy(prox["final_round"]["global/test_accuracy"] if prox else None)],
    ]
    for client_id in range(5):
        avg_acc = avg["final_round"].get(f"client/client_{client_id}/test_accuracy") if avg else None
        prox_acc = prox["final_round"].get(f"client/client_{client_id}/test_accuracy") if prox else None
        cells.append([f"Client {client_id}", format_accuracy(avg_acc), format_accuracy(prox_acc)])
    for row_id, row in enumerate(cells):
        for col_id, value in enumerate(row):
            cell = table.cell(row_id, col_id)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = TEAL if row_id == 0 else NAVY
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Aptos"
                paragraph.font.size = Pt(14)
                paragraph.font.bold = row_id == 0
                paragraph.font.color.rgb = NAVY if row_id == 0 else WHITE
    add_text(slide, "Full per-client values are available in each run's results.json and metrics.csv.",
             1.0, 6.55, 11.0, 0.3, 11, MUTED)

    slide = add_slide(prs, "Reactive optimizers: reading connection", "READING")
    add_bullets(slide, [
        "OPAL uses an AI agent to inspect training signals and rewrite optimizer logic during training.",
        "The post shows adaptive schedules and responses to loss spikes and overfitting.",
        "Limitation: agent-driven decisions may reduce reproducibility; most reported runs were single runs.",
        "Federated idea: use round-level loss, update norms, and update disagreement to tune server steps.",
        "Per-client signals can expose information, so any deployment must respect its privacy design.",
    ], font_size=17, gap=0.83)

    slide = add_slide(prs, "Interpretation and limits", "TAKEAWAYS")
    add_bullets(slide, [
        "Read global and client-level scores together: non-IID data can hide weak client outcomes.",
        "The noisy client makes robustness to mislabeled local data part of the comparison.",
        "A single seed is an initial result; repeated seeds are needed to assess variability.",
        "This single-process simulation does not model communication failures, secure aggregation, or deployment privacy.",
    ], font_size=18, gap=0.9)

    prs.save(DELIVERABLES / "federated_pathmnist_project.pptx")


if __name__ == "__main__":
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    reading_pdf()
    presentation()
    print("Created the Reactive Optimizers PDF and project presentation.")
