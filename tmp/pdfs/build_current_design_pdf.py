"""Build the current multi-maze MPC and imitation-learning design brief."""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "multi_maze_v2_current_design.pdf"
ASSETS = ROOT / "tmp" / "pdfs" / "current_design_assets"
GIF_DIR = ROOT / "artifacts" / "local_segments" / "mpc_gifs"

NAVY = colors.HexColor("#14213D")
BLUE = colors.HexColor("#2563EB")
CYAN = colors.HexColor("#0EA5E9")
GREEN = colors.HexColor("#16A34A")
AMBER = colors.HexColor("#D97706")
RED = colors.HexColor("#DC2626")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#5B6472")
LIGHT = colors.HexColor("#F3F6FA")
LINE = colors.HexColor("#D7DEE8")
WHITE = colors.white


def paragraph_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=26, leading=30, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=10),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=11.5, leading=16, textColor=MUTED, spaceAfter=16),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=18, leading=22, textColor=NAVY, spaceBefore=2,
            spaceAfter=10),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=8,
            spaceAfter=6),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8, leading=11, textColor=MUTED, spaceAfter=4),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=14, textColor=NAVY, alignment=TA_CENTER),
        "table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8, leading=10, textColor=INK),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=WHITE),
    }


S = paragraph_styles()


def footer(canvas, doc):
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(LINE)
    canvas.line(42, 31, width - 42, 31)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(42, 19, "multi_maze_v2 - current design brief - 7 Aug 2026")
    canvas.drawRightString(width - 42, 19, f"Page {doc.page}")
    canvas.restoreState()


def box_flow(labels, fills=None, width=520, height=90):
    fills = fills or [LIGHT] * len(labels)
    drawing = Drawing(width, height)
    gap = 16
    box_w = (width - gap * (len(labels) - 1)) / len(labels)
    box_h = 46
    y = 25
    for i, label in enumerate(labels):
        x = i * (box_w + gap)
        drawing.add(Rect(x, y, box_w, box_h, rx=7, ry=7,
                         fillColor=fills[i], strokeColor=LINE,
                         strokeWidth=0.8))
        lines = label.split("\n")
        for row, text in enumerate(lines):
            drawing.add(String(
                x + box_w / 2, y + box_h / 2 + 5 - row * 11, text,
                fontName="Helvetica-Bold" if row == 0 else "Helvetica",
                fontSize=8.2, fillColor=NAVY, textAnchor="middle"))
        if i < len(labels) - 1:
            x1 = x + box_w + 2
            x2 = x + box_w + gap - 2
            mid = y + box_h / 2
            drawing.add(Line(x1, mid, x2, mid, strokeColor=BLUE,
                             strokeWidth=1.5))
            drawing.add(Polygon(
                [x2, mid, x2 - 5, mid + 3, x2 - 5, mid - 3],
                fillColor=BLUE, strokeColor=BLUE))
    return drawing


def progress_drawing(completed=4825, total=5000):
    width, height = 520, 68
    d = Drawing(width, height)
    ratio = completed / total
    d.add(String(0, 52, "Server demonstration generation",
                 fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    d.add(String(width, 52, f"{completed:,} / {total:,} safely saved",
                 fontName="Helvetica-Bold", fontSize=10, fillColor=GREEN,
                 textAnchor="end"))
    d.add(Rect(0, 23, width, 16, rx=8, ry=8,
               fillColor=LINE, strokeColor=None))
    d.add(Rect(0, 23, width * ratio, 16, rx=8, ry=8,
               fillColor=GREEN, strokeColor=None))
    d.add(String(0, 5, f"{ratio:.1%} complete",
                 fontName="Helvetica", fontSize=8, fillColor=MUTED))
    d.add(String(width, 5, "4 active CPU workers - 0 errors",
                 fontName="Helvetica", fontSize=8, fillColor=MUTED,
                 textAnchor="end"))
    return d


def equation_drawing():
    """Two readable equation cards using PDF-safe ASCII notation."""
    width, height = 520, 116
    d = Drawing(width, height)
    cards = [
        (0, "x-axis motion", "x'' = (5/7) g sin(beta) - c x'"),
        (265, "y-axis motion", "y'' = -(5/7) g sin(alpha) - c y'"),
    ]
    for x, title, equation in cards:
        d.add(Rect(x, 29, 255, 72, rx=8, ry=8,
                   fillColor=colors.HexColor("#E8F1FF"),
                   strokeColor=LINE, strokeWidth=0.8))
        d.add(String(x + 127.5, 79, title,
                     fontName="Helvetica-Bold", fontSize=9,
                     fillColor=BLUE, textAnchor="middle"))
        d.add(String(x + 127.5, 54, equation,
                     fontName="Courier-Bold", fontSize=10.5,
                     fillColor=NAVY, textAnchor="middle"))
    d.add(String(width / 2, 7,
                 "Rolling gain: (5/7) g = 7.007 m/s^2 per radian for small tilts",
                 fontName="Helvetica", fontSize=8.5, fillColor=MUTED,
                 textAnchor="middle"))
    return d


def make_snapshot_strip(gif_path: Path, output: Path):
    image = PILImage.open(gif_path)
    count = getattr(image, "n_frames", 1)
    indices = [0, max(0, count // 2), max(0, count - 1)]
    frames = []
    for index in indices:
        image.seek(index)
        frame = image.convert("RGB").resize((300, 225), PILImage.Resampling.LANCZOS)
        frames.append(frame)
    strip = PILImage.new("RGB", (900, 225), "white")
    for index, frame in enumerate(frames):
        strip.paste(frame, (index * 300, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    strip.save(output, quality=90)


def table(data, widths, header=True):
    converted = []
    for row_index, row in enumerate(data):
        style = S["table_head"] if header and row_index == 0 else S["table"]
        converted.append([Paragraph(str(cell), style) for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0,
                   hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ]
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT]))
    result.setStyle(TableStyle(commands))
    return result


def bullet(text):
    return Paragraph(f"<font color='#2563EB'>-</font> {text}", S["body"])


def build_story():
    story = []

    # Page 1
    story += [
        Spacer(1, 10),
        Paragraph("multi_maze_v2", S["title"]),
        Paragraph("Current control, simulation, and learning design", S["subtitle"]),
        Paragraph(
            "Goal: learn a reliable ball-maze policy from measured physics and "
            "model-predictive demonstrations, then adapt safely to real hardware.",
            S["callout"]),
        Spacer(1, 14),
        progress_drawing(),
        Spacer(1, 10),
        Paragraph("System at a glance", S["h1"]),
        box_flow([
            "Measured physics\nball + actuator model",
            "MPC teacher\nlocal successful skills",
            "Behavior cloning\none conditioned\npolicy",
            "Closed-loop tests\nsegments + full maze",
            "Real hardware\nresidual adaptation",
        ], [colors.HexColor("#E8F1FF"), colors.HexColor("#E7F7FC"),
            colors.HexColor("#EEF2FF"), colors.HexColor("#ECFDF3"),
            colors.HexColor("#FFF6E8")]),
        Spacer(1, 8),
        table([
            ["Evidence already established", "Result"],
            ["Representative MPC canary", "75/75 goals; 0 falls; 0 timeouts"],
            ["Geometry coverage", "Authentic, mirrored-authentic, and procedural"],
            ["Robustness coverage", "Sensor noise and physics randomization up to 25%"],
            ["Local episode policy", "Completion first; 60 s maximum budget"],
            ["Server execution", "20 disjoint workers; resumable 25-episode shards"],
        ], [210, 310]),
        Spacer(1, 12),
        Paragraph(
            "Design principle: known mechanics supply structure; learning is used "
            "for flexible route following, recovery, and the remaining sim-to-real gap.",
            S["callout"]),
        PageBreak(),
    ]

    # Page 2
    story += [
        Paragraph("1. Current MPC teacher design", S["h1"]),
        Paragraph(
            "The teacher replans at 20 Hz. It predicts candidate action sequences "
            "with an analytical rolling-ball and measured actuator surrogate, then "
            "executes only the first action in MuJoCo before planning again.", S["body"]),
        box_flow([
            "Estimated state\nposition + velocity",
            "Analytical prior\npure-pursuit controller",
            "CEM search\n192 candidates\n30 steps",
            "MuJoCo step\ncontacts are authoritative",
            "Replan\nclosed-loop correction",
        ], [LIGHT, colors.HexColor("#E8F1FF"), colors.HexColor("#E7F7FC"),
            colors.HexColor("#FFF6E8"), colors.HexColor("#ECFDF3")]),
        Paragraph("Teacher parameters", S["h2"]),
        table([
            ["Element", "Current setting", "Reason"],
            ["Planning horizon", "30 control steps / 1.5 s", "Anticipates momentum and braking"],
            ["CEM population", "192 candidates, 4 iterations, 24 elites", "Broad search with bounded runtime"],
            ["Analytical prior", "Pure-pursuit velocity field", "Known-safe feedback anchor"],
            ["MPC residual", "+/- 0.03 normalized action", "Prevents model-error wall stalls"],
            ["Execution plant", "MuJoCo", "Retains real contacts, walls, and holes"],
            ["Local timeout", "60 s", "Allows robust completion under randomization"],
        ], [112, 142, 266]),
        Spacer(1, 10),
        Paragraph("Why the teacher is hybrid", S["h2"]),
        table([
            ["Known physics contributes", "MuJoCo contributes", "Learning later contributes"],
            ["Gravity-driven ball acceleration; actuator dead time, lag, slew, and backlash; route geometry and stopping logic.",
             "Wall and floor contacts; hole falls; exact maze geometry; authoritative next state.",
             "One compact policy; recovery from unfamiliar states; residual compensation on the real rig."],
        ], [173, 173, 174]),
        Spacer(1, 12),
        Paragraph(
            "Safety correction discovered by the canary: a 12% MPC residual degraded "
            "one mirrored sharp-right geometry. Reducing it to 3% and restoring the "
            "60 s completion budget produced 75/75 representative successes.",
            S["callout"]),
        PageBreak(),
    ]

    # Mathematical model page
    story += [
        Paragraph("2. Ball-on-plate mathematical model", S["h1"]),
        Paragraph(
            "This model lives inside the MPC prediction loop. For every candidate "
            "plate-command sequence, it predicts the ball trajectory cheaply; "
            "MuJoCo then executes the selected first action and supplies the "
            "authoritative contact-rich next state.", S["body"]),
        box_flow([
            "Action sequence\nnormalized commands",
            "Actuator model\ndelay + lag + slew",
            "Plate angles\nalpha + beta",
            "Ball equations\nacceleration + damping",
            "Predicted path\nposition + velocity",
        ], [colors.HexColor("#EEF2FF"), colors.HexColor("#FFF6E8"),
            LIGHT, colors.HexColor("#E8F1FF"), colors.HexColor("#ECFDF3")]),
        Paragraph("Rolling-sphere dynamics", S["h2"]),
        equation_drawing(),
        Paragraph(
            "The 5/7 factor follows from a solid sphere rolling without slipping: "
            "part of gravity accelerates translation and part accelerates rotation. "
            "The fitted linear term c opposes velocity and approximates rolling "
            "resistance plus residual simulator damping.", S["body"]),
        Paragraph("Variables and sign convention", S["h2"]),
        table([
            ["Symbol", "Meaning", "Project convention"],
            ["alpha", "Plate roll angle", "Positive alpha accelerates the ball toward negative y"],
            ["beta", "Plate pitch angle", "Positive beta accelerates the ball toward positive x"],
            ["g", "Gravity", "9.81 m/s^2"],
            ["c", "Linear damping coefficient", "ball.linear_damping from parameters.json"],
            ["x', y'", "Board-frame ball velocity", "Estimated and latency-predicted state"],
        ], [75, 175, 270]),
        Spacer(1, 8),
        Paragraph("Closed-form control-step integration", S["h2"]),
        table([
            ["Update", "Equation"],
            ["Position", "p_next = p + v dt + 0.5 a dt^2"],
            ["Velocity", "v_next = v + a dt"],
        ], [130, 390]),
        Spacer(1, 8),
        Paragraph("Implementation and limits", S["h2"]),
        bullet("sim/analytic_model.py contains the scalar equations, rollout, stopping-time, and switching-curve helpers."),
        bullet("control/mpc_teacher.py applies the same equations to 192 candidate sequences in parallel."),
        bullet("The predictor includes measured actuator dead time, first-order lag, slew limits, and backlash before the ball equations."),
        PageBreak(),
    ]

    # Page 3
    story += [
        Paragraph("3. Balanced local-skill curriculum", S["h1"]),
        Paragraph(
            "The teacher demonstrates short maneuvers separately. Balanced sampling "
            "prevents abundant straight motion from dominating rare turns.", S["body"]),
        table([
            ["Maneuver", "Geometries", "Initial conditions", "Episode target"],
            ["Straight", "50", "20 per geometry", "1,000"],
            ["Gentle left", "50", "20 per geometry", "1,000"],
            ["Gentle right", "50", "20 per geometry", "1,000"],
            ["Sharp left", "50", "20 per geometry", "1,000"],
            ["Sharp right", "50", "20 per geometry", "1,000"],
            ["Total", "250", "5,000 specifications", "5,000"],
        ], [150, 110, 150, 110]),
        Spacer(1, 8),
        Paragraph("Geometry sources", S["h2"]),
        table([
            ["Source", "Count", "Purpose"],
            ["Authentic", "42", "Local slices of the actual maze"],
            ["Mirrored authentic", "42", "Valid reflected layouts with walls and holes reflected too"],
            ["Procedural", "166", "Fills class imbalance and expands shape diversity"],
        ], [150, 70, 300]),
        Spacer(1, 10),
        Paragraph("Initial-condition coverage", S["h2"]),
        bullet("Lateral offsets span both sides of the route centerline within local clearance."),
        bullet("Forward and lateral velocities vary so the teacher demonstrates braking and recovery, not only rest starts."),
        bullet("Physics randomization levels include nominal, 10%, and 25%, with sensor noise enabled."),
        PageBreak(),
        Paragraph("Representative teacher demonstrations", S["h1"]),
        Paragraph(
            "Each row shows the beginning, middle, and successful end of one "
            "authentic-maze local maneuver.", S["body"]),
    ]
    for label, filename, result in [
        ("Straight", "straight-mpc-teacher.gif", "Goal in 3.85 s; mean cross-track 1.83 mm"),
        ("Gentle left", "gentle_left-mpc-teacher.gif", "Goal in 5.80 s; mean cross-track 2.53 mm"),
        ("Sharp right", "sharp_right-mpc-teacher.gif", "Goal in 13.70 s; mean cross-track 4.04 mm"),
    ]:
        strip_path = ASSETS / (Path(filename).stem + "-strip.jpg")
        make_snapshot_strip(GIF_DIR / filename, strip_path)
        story += [
            Paragraph(f"<b>{label}</b> - {result}", S["small"]),
            Image(str(strip_path), width=7.22 * inch, height=1.805 * inch),
            Spacer(1, 5),
        ]
    story.append(PageBreak())

    # Page 4
    story += [
        Paragraph("4. One route-conditioned imitation policy", S["h1"]),
        Paragraph(
            "The short skills are not deployed as five separate controllers. Their "
            "successful transitions are pooled to train one policy that selects the "
            "appropriate behavior from the local route shape.", S["body"]),
        box_flow([
            "5 skill datasets\nbalanced examples",
            "22-float observation\nstate + route lookahead",
            "Neural policy\nshared representation",
            "2-float action\nroll + pitch command",
            "Complete route\nskills blend continuously",
        ], [colors.HexColor("#E7F7FC"), LIGHT, colors.HexColor("#EEF2FF"),
            colors.HexColor("#FFF6E8"), colors.HexColor("#ECFDF3")]),
        Paragraph("Policy contract", S["h2"]),
        table([
            ["Observation block", "Floats", "Meaning"],
            ["Ball position", "2", "Board-normalized x and y"],
            ["Predicted ball velocity", "2", "Latency-compensated vx and vy"],
            ["Board angles", "2", "Normalized roll and pitch"],
            ["Action history", "6", "Three previous two-axis commands"],
            ["Route lookahead", "10", "Five ball-relative points at 12 mm spacing"],
            ["Total", "22", "State-based, route-conditioned observation"],
        ], [170, 60, 290]),
        Spacer(1, 10),
        Paragraph("Behavior-cloning objective", S["h2"]),
        Paragraph(
            "For every saved teacher transition, minimize the squared difference "
            "between the policy action and the MPC action: "
            "<b>L = mean || policy(observation) - teacher_action ||^2</b>.",
            S["body"]),
        Paragraph("Training split and balancing", S["h2"]),
        bullet("Split by complete geometry, never by individual frame, so validation routes are genuinely unseen."),
        bullet("Balance maneuver classes and emphasize recovery states instead of letting long straight trajectories dominate."),
        bullet("Train on the server GPU, but judge models through closed-loop simulation rather than action loss alone."),
        bullet("Save the best checkpoint by evaluation success, not merely the final epoch."),
        Spacer(1, 12),
        Paragraph(
            "The route-conditioned policy does not explicitly choose 'turn left' or "
            "'go straight.' Five relative lookahead points describe the local shape, "
            "and the shared network produces the matching action continuously.",
            S["callout"]),
        PageBreak(),
    ]

    # Page 5
    story += [
        Paragraph("5. Closed-loop evaluation and recovery", S["h1"]),
        Paragraph(
            "Low imitation loss is necessary but not sufficient. Small action errors "
            "can move the ball into states absent from the teacher data, so every "
            "candidate policy must be executed in simulation.", S["body"]),
        box_flow([
            "Train policy\nbehavior cloning",
            "Run closed loop\nheld-out segments",
            "Collect failures\ndrift + recovery states",
            "MPC relabels\ncorrect action",
            "Retrain\nexpanded dataset",
        ], [colors.HexColor("#EEF2FF"), colors.HexColor("#ECFDF3"),
            colors.HexColor("#FFF6E8"), colors.HexColor("#E7F7FC"),
            colors.HexColor("#EEF2FF")]),
        Paragraph("Evaluation ladder", S["h2"]),
        table([
            ["Gate", "Primary measures", "Decision"],
            ["Held-out local segments", "Success by class; falls; completion; cross-track", "Target >=95% success with near-zero falls"],
            ["Unseen procedural routes", "Generalization across radius, angle, and placement", "Reject memorized geometry"],
            ["Full nominal maze", "Start-to-goal success and accumulated drift", "Prove that local skills compose"],
            ["Randomized simulation", "Success under friction, latency, bias, and noise", "Measure robustness before hardware"],
            ["Hardware canary", "Conservative speed, tilt, and emergency-stop behavior", "Expand only after safe repeatability"],
        ], [132, 235, 153]),
        Spacer(1, 10),
        Paragraph("Anti-collapse protections for later SAC fine-tuning", S["h2"]),
        bullet("Retain teacher demonstrations in replay rather than replacing them with weak on-policy behavior."),
        bullet("Use an imitation regularizer so the actor cannot drift rapidly away from the successful teacher manifold."),
        bullet("Use a small learning rate, periodic deterministic evaluation, and best-checkpoint protection."),
        bullet("Stop or roll back when success falls, even if TensorBoard training reward rises."),
        Spacer(1, 12),
        Paragraph(
            "Success is always an executed outcome. Route completion alone does not "
            "prove that the ball reached the goal, and training reward alone does not "
            "prove that the policy is stable.", S["callout"]),
        PageBreak(),
    ]

    # Page 6
    story += [
        Paragraph("6. Execution roadmap", S["h1"]),
        Paragraph("Current server layout", S["h2"]),
        table([
            ["Item", "Current design"],
            ["Remote project", "/home/tn22833/multi_maze_v2"],
            ["Teacher generation", "20 disjoint CPU workers; 25 episodes per resumable shard"],
            ["Hardware available", "256 CPU cores; five RTX 6000 Ada GPUs with 49 GB each"],
            ["Current status snapshot", "4,825/5,000 saved; 4 workers active; 0 error logs"],
            ["Next runtime", "GPU-enabled PyTorch for behavior cloning and later SAC"],
        ], [175, 345]),
        Spacer(1, 12),
        Paragraph("Milestones from here", S["h2"]),
        table([
            ["Stage", "Work", "Exit condition"],
            ["A. Corpus closeout", "Verify 5,000 episodes, unique IDs, readable shards, class/source balance", "Integrity report passes"],
            ["B. Behavior cloning", "Train one route-conditioned policy on GPU", "Validation loss stable; no train/validation leakage"],
            ["C. Segment evaluation", "Run held-out authentic, mirrored, and procedural routes", ">=95% local success; near-zero falls"],
            ["D. DAgger recovery", "MPC relabels states reached by policy drift", "Recovery failures materially reduced"],
            ["E. Full-maze evaluation", "Nominal and randomized start-to-goal runs", "Reliable goal success, not completion only"],
            ["F. Protected RL", "Fine-tune with SAC plus demonstration retention", "Beats imitation without checkpoint collapse"],
            ["G. Hardware transfer", "Conservative canary, system identification, residual adaptation", "Safe repeatable real-maze runs"],
        ], [92, 250, 178]),
        Spacer(1, 14),
        Paragraph("Final architecture", S["h2"]),
        box_flow([
            "Physics model\nstructured prior",
            "MPC demos\nsuccessful local\nbehavior",
            "Imitation policy\nfast 20 Hz inference",
            "Protected RL\nperformance refinement",
            "Real rig\nmeasured residual gap",
        ], [colors.HexColor("#E8F1FF"), colors.HexColor("#E7F7FC"),
            colors.HexColor("#EEF2FF"), colors.HexColor("#ECFDF3"),
            colors.HexColor("#FFF6E8")]),
        Spacer(1, 12),
        Paragraph(
            "The intended result is not an MPC controller running permanently on "
            "the robot. MPC supplies high-quality behavior offline; the learned "
            "route-conditioned policy supplies fast online control; hardware data "
            "then corrects the remaining model mismatch.", S["callout"]),
    ]
    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUT), pagesize=letter, leftMargin=42, rightMargin=42,
        topMargin=42, bottomMargin=42, title="multi_maze_v2 Current Design",
        author="Codex and Thanh")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="main")
    doc.addPageTemplates(PageTemplate(id="design", frames=[frame], onPage=footer))
    doc.build(build_story())
    print(OUT.resolve())


if __name__ == "__main__":
    main()
