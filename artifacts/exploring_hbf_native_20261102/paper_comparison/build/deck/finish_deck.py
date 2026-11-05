"""Add the XY charts, send all chart text to the theme fonts, set the theme's CJK font and split the notes into paragraphs."""
import json
import re
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_LINE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
DATA = json.loads((HERE / "deck_data.json").read_text(encoding="utf-8"))
SOURCE, TARGET = sys.argv[1], sys.argv[2]
INK, INK2, MUTED, GRID, RED, PAPER, NATIVE = "14181F", "52514E", "6F6D67", "E1E0D9", "FF0000", "83878E", "2A78D6"
CJK = "Microsoft YaHei"
SHORT = {"llama3_405b": "405B", "llama4_maverick": "Mav"}
# Children of a run's properties that must come before its fonts.
BEFORE_FONTS = {qn(f"a:{name}") for name in ("ln", "noFill", "solidFill", "gradFill", "blipFill", "pattFill", "grpFill", "effectLst", "effectDag", "highlight", "uLnTx", "uLn", "uFillTx", "uFill")}
KEY = {(r["model"], r["workload"], r["hardware"]): r for r in DATA["rows"]}


def rgb(value):
    return RGBColor.from_string(value)


def style_axis(axis, title, low, high, unit, number_format="0.0", tick_size=10, title_size=12):
    axis.minimum_scale, axis.maximum_scale, axis.major_unit = low, high, unit
    axis.has_major_gridlines = True
    axis.major_gridlines.format.line.color.rgb = rgb(GRID)
    axis.major_gridlines.format.line.width = Pt(0.75)
    axis.format.line.color.rgb = rgb("C3C2B7")
    axis.tick_labels.font.size = Pt(tick_size)
    axis.tick_labels.font.color.rgb = rgb(MUTED)
    axis.tick_labels.number_format = number_format
    axis.tick_labels.number_format_is_linked = False
    axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    axis.has_title = bool(title)
    if not title:
        return
    axis.axis_title.text_frame.text = title
    run = axis.axis_title.text_frame.paragraphs[0].runs[0]
    run.font.size, run.font.bold = Pt(title_size), False
    run.font.color.rgb = rgb(INK2)


def line_series(series, color, width):
    series.format.line.color.rgb = rgb(color)
    series.format.line.width = Pt(width)
    series.marker.style = XL_MARKER_STYLE.NONE
    series.smooth = False


def dot_series(series, fill, size=9, style=XL_MARKER_STYLE.CIRCLE):
    series.format.line.fill.background()
    series.marker.style = style
    series.marker.size = size
    series.smooth = False
    series.marker.format.fill.solid()
    series.marker.format.fill.fore_color.rgb = rgb(fill)
    series.marker.format.line.color.rgb = rgb("FFFFFF")
    series.marker.format.line.width = Pt(1)


def point_label(series, index, text, position, color=INK, bold=False, size=10):
    label = series.points[index].data_label
    label.text_frame.text = text
    run = label.text_frame.paragraphs[0].runs[0]
    run.font.size, run.font.bold = Pt(size), bold
    run.font.color.rgb = rgb(color)
    label.position = position
    return label


def nudge(label, x, y):
    layout = etree.Element(qn("c:layout"))
    manual = etree.SubElement(layout, qn("c:manualLayout"))
    etree.SubElement(manual, qn("c:x")).set("val", str(x))
    etree.SubElement(manual, qn("c:y")).set("val", str(y))
    label._dLbl.find(qn("c:idx")).addnext(layout)


def add_xy(slide, x, y, w, h, series, name, description):
    data = XyChartData()
    for label, points in series:
        target = data.add_series(label)
        for px, py in points:
            target.add_data_point(px, py)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), data)
    frame.name = name
    frame._element.xpath("./p:nvGraphicFramePr/p:cNvPr")[0].set("descr", description)
    chart = frame.chart
    chart.has_title = False
    chart.has_legend = False
    return chart


class Plot:
    """Pin a chart's inner plot area so annotations can be placed by data value."""

    def __init__(self, chart, frame, layout, xs, ys):
        self.frame, self.layout, self.xs, self.ys = frame, layout, xs, ys
        area = chart._chartSpace.find(qn("c:chart")).find(qn("c:plotArea"))
        node = area.find(qn("c:layout"))
        if node is None:
            node = etree.Element(qn("c:layout"))
            area.insert(0, node)
        manual = etree.SubElement(node, qn("c:manualLayout"))
        for tag, value in (("layoutTarget", "inner"), ("xMode", "edge"), ("yMode", "edge"), ("x", layout[0]), ("y", layout[1]), ("w", layout[2]), ("h", layout[3])):
            etree.SubElement(manual, qn(f"c:{tag}")).set("val", str(value))

    def x(self, value):
        fx, _, fw, _ = self.frame
        return fx + fw * (self.layout[0] + self.layout[2] * (value - self.xs[0]) / (self.xs[1] - self.xs[0]))

    def y(self, value):
        _, fy, _, fh = self.frame
        return fy + fh * (self.layout[1] + self.layout[3] * (self.ys[1] - value) / (self.ys[1] - self.ys[0]))


def note_box(slide, x, y, w, h, text, name, size=10, color=INK, bold=True, align=PP_ALIGN.CENTER):
    """A short annotation laid over a chart; x, y is the box's top-left corner."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    frame = box.text_frame
    frame.word_wrap = False
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.text = text
    for paragraph in frame.paragraphs:
        paragraph.alignment = align
        for run in paragraph.runs:
            run.font.size, run.font.bold = Pt(size), bold
            run.font.color.rgb = rgb(color)


def use_theme_fonts(chart_space):
    """Latin text takes the theme's Latin font and Chinese text its East Asian font, so one theme edit changes every chart."""
    for properties in chart_space.iter(qn("a:defRPr"), qn("a:rPr")):
        for tag in ("a:latin", "a:ea", "a:cs"):
            for old in properties.findall(qn(tag)):
                properties.remove(old)
        position = sum(child.tag in BEFORE_FONTS for child in properties)
        for offset, (tag, face) in enumerate((("a:latin", "+mn-lt"), ("a:ea", "+mn-ea"), ("a:cs", "+mn-cs"))):
            font = etree.Element(qn(tag))
            font.set("typeface", face)
            properties.insert(position + offset, font)


def spread(rows, field):
    values = [r[field] for r in rows]
    return f"{min(values):.2f}–{max(values):.2f}"


def parity_chart(slide):
    others = [r for r in DATA["rows"] if r["hardware"] != "HBM4"]
    hbf = [r for r in others if r["hardware"] in ("HBF", "HBF+")]
    conv = [r for r in others if r["hardware"] in ("CONV", "CONV+")]
    point = lambda r: (round(r["paper_norm_tps"], 3), round(r["native_norm_tps"], 3))
    frame = (0.55, 1.64, 7.1, 5.08)
    chart = add_xy(slide, *frame,
                   (("y = x", [(0, 0), (1.6, 1.6)]), ("+0.15", [(0, 0.15), (1.45, 1.6)]), ("−0.15", [(0.15, 0), (1.6, 1.45)]),
                    ("HBF、HBF+", [point(r) for r in hbf]), ("CONV、CONV+", [point(r) for r in conv])),
                   "归一化吞吐散点图", "24 个非 HBM4 点的论文与 native 归一化吞吐，6 个 HBF 系的点超出正负 0.15，其余在范围内")
    chart.font.size = Pt(11)
    style_axis(chart.category_axis, "论文的归一化吞吐（Fig. 4）", 0, 1.6, 0.4)
    style_axis(chart.value_axis, "native 的归一化吞吐", 0, 1.6, 0.4)
    series = list(chart.plots[0].series)
    line_series(series[0], INK2, 1.25)
    line_series(series[1], "C3C2B7", 0.75)
    line_series(series[2], "C3C2B7", 0.75)
    dot_series(series[3], INK)
    dot_series(series[4], "A9ADB3")
    positions = {("llama4_maverick", "SHORT", "HBF+"): XL_LABEL_POSITION.RIGHT, ("llama4_maverick", "SHORT", "HBF"): XL_LABEL_POSITION.LEFT, ("llama4_maverick", "MID", "HBF+"): XL_LABEL_POSITION.RIGHT,
                 ("llama4_maverick", "MID", "HBF"): XL_LABEL_POSITION.RIGHT, ("llama3_405b", "LONG", "HBF"): XL_LABEL_POSITION.LEFT, ("llama3_405b", "SHORT", "HBF+"): XL_LABEL_POSITION.LEFT}
    for index, r in enumerate(hbf):
        if abs(r["native_norm_tps"] - r["paper_norm_tps"]) > 0.15:
            # Pure red marks the points where native falls short of the paper.
            below = r["native_norm_tps"] < r["paper_norm_tps"]
            case = (r["model"], r["workload"], r["hardware"])
            label = point_label(series[3], index, f"{SHORT[r['model']]} {r['workload']} {r['hardware']}", positions[case], RED if below else INK, below)
            if case == ("llama4_maverick", "SHORT", "HBF"):
                # Nudged off the neighbouring 405B SHORT HBF point and the diagonal.
                nudge(label, 0.035, 0.035)
    plot = Plot(chart, frame, (0.09, 0.03, 0.88, 0.85), (0, 1.6), (0, 1.6))
    note_box(slide, plot.x(0.05), plot.y(1.5) - 0.13, 3.2, 0.26, "黑点 HBF 系：与 HBM4 同档", "HBF 系注解", 12, INK, True, PP_ALIGN.LEFT)
    note_box(slide, plot.x(0.05), plot.y(1.5) + 0.13, 3.2, 0.22, f"论文 {spread(hbf, 'paper_norm_tps')}，native {spread(hbf, 'native_norm_tps')}", "HBF 系范围", 11, INK2, False, PP_ALIGN.LEFT)
    note_box(slide, plot.x(0.62), plot.y(0.22) - 0.13, 3.2, 0.26, "灰点 CONV 系：明显落后", "CONV 系注解", 12, INK, True, PP_ALIGN.LEFT)
    note_box(slide, plot.x(0.62), plot.y(0.22) + 0.13, 3.2, 0.22, f"论文 {spread(conv, 'paper_norm_tps')}，native {spread(conv, 'native_norm_tps')}", "CONV 系范围", 11, INK2, False, PP_ALIGN.LEFT)


def ratio_chart(slide):
    """Batch ratio against TPOT ratio: every point sits under the line where throughput equals the paper's."""
    rows = DATA["rows"]
    hbm4 = [r for r in rows if r["hardware"] == "HBM4"]
    others = [r for r in rows if r["hardware"] != "HBM4"]
    point = lambda r: (round(r["tpot_ratio"], 3), round(r["batch_ratio"], 3))
    frame = (0.55, 2.62, 4.45, 4.1)
    chart = add_xy(slide, *frame, (("吞吐等于论文", [(0.4, 0.4), (1.05, 1.05)]), ("HBM4", [point(r) for r in hbm4]), ("其余配置", [point(r) for r in others])),
                   "batch 比与 TPOT 比散点图", "30 个点的 batch 比与 TPOT 比，全部低于吞吐等于论文的对角线；HBM4 的点在右上，其余配置在左下")
    chart.font.size = Pt(10.5)
    style_axis(chart.category_axis, "TPOT 比（越右，单步比论文越慢）", 0.4, 2.0, 0.4, tick_size=9.5, title_size=10.5)
    style_axis(chart.value_axis, "batch 比（越低，请求越少）", 0, 1.2, 0.4, tick_size=9.5, title_size=10.5)
    series = list(chart.plots[0].series)
    line_series(series[0], INK2, 1.25)
    dot_series(series[1], INK, 8)
    dot_series(series[2], "6F9FD8", 7)
    plot = Plot(chart, frame, (0.145, 0.04, 0.815, 0.78), (0.4, 2.0), (0, 1.2))
    note_box(slide, plot.x(0.97) - 1.2, plot.y(1.1) - 0.1, 1.2, 0.2, "吞吐等于论文", "对角线注解", 9.5, INK2, False, PP_ALIGN.RIGHT)
    note_box(slide, plot.x(1.56) - 1.1, plot.y(1.08) - 0.18, 2.2, 0.36, "黑点 HBM4：货差不多，路上慢\n（405B SHORT 例外）", "HBM4 注解", 10.5)
    note_box(slide, plot.x(0.97) - 1.3, plot.y(0.12) - 0.12, 2.6, 0.24, "蓝点 其余配置：路上不慢，货装得少", "其余配置注解", 10.5)
    note_box(slide, plot.x(1.34) + 0.08, plot.y(0.5) - 0.1, 1.2, 0.2, "Mav SHORT HBF+", "Maverick SHORT HBF+ 注解", 9.5, RED, True, PP_ALIGN.LEFT)


def curve_chart(slide, frame, curve, selected, case, name):
    """TPOT against batch: where native stopped on its curve and where the paper's point sits."""
    r = KEY[case]
    measured = [p for p in curve if p["status"] != "forecast_rejected"]
    estimated = [p for p in curve if p["status"] == "forecast_rejected"]
    last = measured[-1]
    pick = next(p for p in curve if p["batch"] == selected)
    paper = (round(r["paper_batch_per_gpu"] * 8), r["paper_tpot_ms"])
    series = [("100 ms 上限", [(0, 100), (6500, 100)]), ("native", [(p["batch"], round(p["tpot"], 1)) for p in measured])]
    if estimated:
        series.append(("native 估计", [(last["batch"], round(last["tpot"], 1))] + [(p["batch"], round(p["tpot"], 1)) for p in estimated]))
    series += [("选中点", [(pick["batch"], round(pick["tpot"], 1))]), ("论文", [paper])]
    chart = add_xy(slide, *frame, series, name, f"{name}：TPOT 随 batch 的变化，native 选中 B{selected}，TPOT {pick['tpot']:.1f} ms；论文 B{paper[0]}，TPOT {paper[1]:.1f} ms")
    chart.font.size = Pt(9.5)
    # A vertical axis title would sit on the tick labels at this size, so the one axis title names both axes.
    style_axis(chart.category_axis, "横轴 batch 总数，纵轴 TPOT（ms）", 0, 6500, 2000, "#,##0", 9, 9)
    style_axis(chart.value_axis, None, 0, 125, 50, "0", 9)
    plot = Plot(chart, frame, (0.085, 0.05, 0.885, 0.68), (0, 6500), (0, 125))
    plotted = list(chart.plots[0].series)
    line_series(plotted[0], RED, 1)
    plotted[1].format.line.color.rgb = rgb(NATIVE)
    plotted[1].format.line.width = Pt(1.75)
    plotted[1].smooth = False
    plotted[1].marker.style = XL_MARKER_STYLE.CIRCLE
    plotted[1].marker.size = 5
    plotted[1].marker.format.fill.solid()
    plotted[1].marker.format.fill.fore_color.rgb = rgb(NATIVE)
    plotted[1].marker.format.line.color.rgb = rgb(NATIVE)
    offset = 2
    if estimated:
        plotted[2].format.line.color.rgb = rgb(NATIVE)
        plotted[2].format.line.width = Pt(1.25)
        plotted[2].format.line.dash_style = MSO_LINE.DASH
        plotted[2].smooth = False
        plotted[2].marker.style = XL_MARKER_STYLE.CIRCLE
        plotted[2].marker.size = 5
        plotted[2].marker.format.fill.solid()
        plotted[2].marker.format.fill.fore_color.rgb = rgb("FFFFFF")
        plotted[2].marker.format.line.color.rgb = rgb(NATIVE)
        offset = 3
    dot_series(plotted[offset], NATIVE, 10)
    dot_series(plotted[offset + 1], PAPER, 10, XL_MARKER_STYLE.DIAMOND)
    note_box(slide, plot.x(pick["batch"]) + 0.09, plot.y(pick["tpot"]) + 0.02, 1.3, 0.18, f"native B{pick['batch']:,}，{pick['tpot']:.1f} ms", name + " 选中点", 9.5, INK, True, PP_ALIGN.LEFT)
    right = min(plot.x(paper[0]) + 0.55, plot.x(6500))
    note_box(slide, right - 1.45, plot.y(paper[1]) + 0.08, 1.45, 0.18, f"论文 B{paper[0]:,}，{paper[1]:.1f} ms", name + " 论文点", 9.5, INK2, False, PP_ALIGN.RIGHT)
    note_box(slide, plot.x(100), plot.y(100) - 0.2, 0.9, 0.17, "100 ms 上限", name + " 上限", 9, RED, False, PP_ALIGN.LEFT)


deck = Presentation(SOURCE)
slides = list(deck.slides)
parity_chart(slides[0])
ratio_chart(slides[1])
curves = DATA["curves"]
curve_chart(slides[2], (0.55, 5.12, 3.85, 1.62), curves["mid_hbf_plus"], 1504, ("llama4_maverick", "MID", "HBF+"), "MID HBF+ 的 TPOT 曲线")
curve_chart(slides[2], (6.8, 5.12, 3.85, 1.62), curves["mid_hbf"], 1648, ("llama4_maverick", "MID", "HBF"), "MID HBF 的 TPOT 曲线")

for page in deck.slides:
    for shape in page.shapes:
        if shape.has_chart:
            use_theme_fonts(shape.chart._chartSpace)
    # pptxgenjs writes the notes as one run; assigning the text back makes each line a paragraph.
    notes = page.notes_slide.notes_text_frame
    notes.text = notes.text

# pptxgenjs leaves the East Asian theme fonts empty, so Chinese text would fall back to a serif face.
for part in deck.part.package.iter_parts():
    if str(part.partname).startswith("/ppt/theme/"):
        xml = part.blob.decode("utf-8")
        xml = re.sub(r'<a:ea typeface="[^"]*"\s*/>', f'<a:ea typeface="{CJK}"/>', xml)
        xml = re.sub(r'<a:font script="Hans" typeface="[^"]*"\s*/>', f'<a:font script="Hans" typeface="{CJK}"/>', xml)
        part._blob = xml.encode("utf-8")
deck.save(TARGET)
print("saved", TARGET)
