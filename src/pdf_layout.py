"""Stacked PDF page layout that reserves space from font sizes and legend size.

Axes are placed in absolute inches (not GridSpec fractions). Each slot owns:
  [title pad] + [data axes] + [tick/xlabel] + [legend or table] + [gap]
so changing fonts, toggling panels, or wrapping long legend labels grows the
page instead of overlapping the next graph.

Adding a panel: append a Slot to the page spec (and draw into axes[slot.key]).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PT_IN = 1.0 / 72.0
SlotKind = Literal["header", "mea", "plot"]
ExtraBelow = Literal["none", "psth_table"]


@dataclass(frozen=True)
class LayoutFonts:
    legend: float
    axis_title: float
    axis_label: float
    tick: float
    section_header: float
    mea_title: float
    table: float
    unavailable: float


@dataclass
class Slot:
    """One vertical block on a stacked page."""

    key: str
    kind: SlotKind = "plot"
    plot_height_in: float = 2.0
    width_in: float | None = None
    header_text: str | None = None
    has_legend: bool = False
    extra_below: ExtraBelow = "none"
    table_rows: int = 0


@dataclass
class StackedPage:
    fig: Any
    axes: dict[str, Any]
    fonts: LayoutFonts
    width_in: float
    height_in: float
    left_in: float
    axes_width_in: float
    n_legend_rows: int


def text_height_in(fontsize: float, lines: float = 1.0, leading: float = 1.45) -> float:
    return max(0.0, float(fontsize) * PT_IN * leading * max(lines, 0.0))


def wrap_label(text: str, fontsize: float, width_in: float) -> str:
    """Wrap a legend / table label to the available axes width."""
    raw = " ".join(str(text).split())
    if not raw:
        return raw
    char_in = max(float(fontsize) * 0.50 * PT_IN, 0.045)
    max_chars = max(18, int(max(width_in, 0.5) / char_in))
    lines = textwrap.wrap(raw, width=max_chars, break_long_words=True, break_on_hyphens=False)
    return "\n".join(lines) if lines else raw


def count_wrapped_lines(text: str, fontsize: float, width_in: float) -> int:
    wrapped = wrap_label(text, fontsize, width_in)
    return max(1, wrapped.count("\n") + 1)


def estimate_legend_rows(labels: Sequence[str], fonts: LayoutFonts, width_in: float) -> int:
    if not labels:
        return 1
    return max(1, sum(count_wrapped_lines(lab, fonts.legend, width_in) for lab in labels if lab))


def xlabel_block_in(fonts: LayoutFonts) -> float:
    return text_height_in(fonts.tick, 1.15) + text_height_in(fonts.axis_label, 1.15) + 0.06


def legend_block_in(fonts: LayoutFonts, n_rows: int) -> float:
    rows = max(1, int(n_rows))
    return text_height_in(fonts.legend, rows, leading=1.50) + 0.18


def table_block_in(fonts: LayoutFonts, n_data_rows: int) -> float:
    rows = max(1, int(n_data_rows)) + 1  # header
    return text_height_in(fonts.table, rows, leading=1.80) + 0.12


def title_block_in(fonts: LayoutFonts, *, mea: bool = False) -> float:
    size = fonts.mea_title if mea else fonts.axis_title
    return text_height_in(size, 1.2) + 0.10


def header_block_in(fonts: LayoutFonts) -> float:
    return text_height_in(fonts.section_header, 1.25) + 0.10


def slot_below_in(slot: Slot, fonts: LayoutFonts, n_legend_rows: int) -> float:
    if slot.kind == "header":
        return 0.04
    if slot.kind == "mea":
        return 0.12
    below = xlabel_block_in(fonts)
    extras: list[float] = []
    if slot.has_legend:
        extras.append(legend_block_in(fonts, n_legend_rows))
    if slot.extra_below == "psth_table":
        extras.append(table_block_in(fonts, slot.table_rows or n_legend_rows))
    if extras:
        below += max(extras)
    return below + 0.16


def slot_above_in(slot: Slot, fonts: LayoutFonts) -> float:
    if slot.kind == "header":
        return 0.04
    if slot.kind == "mea":
        return title_block_in(fonts, mea=True)
    return title_block_in(fonts, mea=False)


def slot_body_in(slot: Slot) -> float:
    if slot.kind == "header":
        return max(0.28, slot.plot_height_in)
    return max(0.8, float(slot.plot_height_in))


def pack_slot_pages(
    slots: Sequence[Slot],
    *,
    fonts: LayoutFonts,
    n_legend_rows: int,
    max_height_in: float = 36.0,
    top_in: float = 0.22,
    bottom_in: float = 0.22,
) -> list[list[Slot]]:
    """Group slots into pages, keeping a section header with the following plot."""
    if not slots:
        return [[]]
    units: list[list[Slot]] = []
    idx = 0
    while idx < len(slots):
        slot = slots[idx]
        if slot.kind == "header" and idx + 1 < len(slots):
            units.append([slot, slots[idx + 1]])
            idx += 2
        else:
            units.append([slot])
            idx += 1

    def _unit_height(unit: Sequence[Slot]) -> float:
        return sum(
            slot_above_in(item, fonts) + slot_body_in(item) + slot_below_in(item, fonts, n_legend_rows)
            for item in unit
        )

    pages: list[list[Slot]] = []
    current: list[Slot] = []
    current_h = float(top_in) + float(bottom_in)
    limit = max(8.0, float(max_height_in))
    for unit in units:
        unit_h = _unit_height(unit)
        if current and current_h + unit_h > limit:
            pages.append(current)
            current = list(unit)
            current_h = float(top_in) + float(bottom_in) + unit_h
        else:
            current.extend(unit)
            current_h += unit_h
    if current:
        pages.append(current)
    return pages


def build_stacked_pages(
    slots: Sequence[Slot],
    *,
    fonts: LayoutFonts,
    n_legend_rows: int,
    width_in: float = 12.0,
    left_in: float = 1.28,
    right_in: float = 0.28,
    top_in: float = 0.22,
    bottom_in: float = 0.22,
    sharex_groups: dict[str, str] | None = None,
    max_height_in: float = 36.0,
) -> list[StackedPage]:
    packed = pack_slot_pages(
        slots,
        fonts=fonts,
        n_legend_rows=n_legend_rows,
        max_height_in=max_height_in,
        top_in=top_in,
        bottom_in=bottom_in,
    )
    return [
        build_stacked_page(
            page_slots,
            fonts=fonts,
            n_legend_rows=n_legend_rows,
            width_in=width_in,
            left_in=left_in,
            right_in=right_in,
            top_in=top_in,
            bottom_in=bottom_in,
            sharex_groups=sharex_groups,
        )
        for page_slots in packed
        if page_slots
    ]


def build_stacked_page(
    slots: Sequence[Slot],
    *,
    fonts: LayoutFonts,
    n_legend_rows: int,
    width_in: float = 12.0,
    left_in: float = 1.28,
    right_in: float = 0.28,
    top_in: float = 0.22,
    bottom_in: float = 0.22,
    sharex_groups: dict[str, str] | None = None,
) -> StackedPage:
    """Create a figure and axes stacked from top to bottom in inches."""
    if not slots:
        slots = [Slot(key="ax_empty", kind="header", header_text="")]
    n_rows = max(1, int(n_legend_rows))
    axes_width_in = max(2.0, float(width_in) - float(left_in) - float(right_in))

    budgets: list[tuple[Slot, float, float, float]] = []
    total = float(top_in) + float(bottom_in)
    for slot in slots:
        above = slot_above_in(slot, fonts)
        body = slot_body_in(slot)
        below = slot_below_in(slot, fonts, n_rows)
        budgets.append((slot, above, body, below))
        total += above + body + below

    height_in = max(4.0, total)
    fig = plt.figure(figsize=(float(width_in), height_in), layout=None)
    axes: dict[str, Any] = {}
    sharex_by_group: dict[str, Any] = {}
    groups = sharex_groups or {}

    y_top = height_in - float(top_in)

    for slot, above, body, below in budgets:
        y_top -= above
        y0 = y_top - body
        slot_w = float(slot.width_in) if slot.width_in is not None else axes_width_in
        slot_w = min(max(0.8, slot_w), axes_width_in)
        left_in_slot = float(left_in) + 0.5 * (axes_width_in - slot_w)
        rect = [
            left_in_slot / float(width_in),
            y0 / height_in,
            slot_w / float(width_in),
            body / height_in,
        ]
        group = groups.get(slot.key)
        sharex = sharex_by_group.get(group) if group else None
        ax = fig.add_axes(rect, sharex=sharex)
        axes[slot.key] = ax
        if group and group not in sharex_by_group and slot.kind == "plot":
            sharex_by_group[group] = ax
        if slot.kind == "header":
            ax.axis("off")
            if slot.header_text:
                ax.text(
                    0.0,
                    0.35,
                    slot.header_text,
                    ha="left",
                    va="center",
                    fontsize=fonts.section_header,
                    fontweight="bold",
                    transform=ax.transAxes,
                    clip_on=False,
                )
        y_top = y0 - below

    return StackedPage(
        fig=fig,
        axes=axes,
        fonts=fonts,
        width_in=float(width_in),
        height_in=height_in,
        left_in=float(left_in),
        axes_width_in=axes_width_in,
        n_legend_rows=n_rows,
    )


def place_legend_below(
    ax: Any,
    fonts: LayoutFonts,
    *,
    ncol: int = 1,
    handles: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
    framealpha: float = 0.92,
) -> Any:
    """Place a legend under the axes (below ticks + xlabel), wrapping long labels."""
    if ax is None:
        return None
    fig = ax.figure
    pos = ax.get_position()
    fig_h = float(fig.get_figheight())
    fig_w = float(fig.get_figwidth())
    width_in = pos.width * fig_w
    offset_in = xlabel_block_in(fonts) + 0.02
    anchor_x = pos.x0 + pos.width * 0.5
    anchor_y = pos.y0 - offset_in / fig_h

    if handles is None or labels is None:
        raw_handles, raw_labels = ax.get_legend_handles_labels()
        use_handles: list[Any] = []
        use_labels: list[str] = []
        seen: set[str] = set()
        for handle, lab in zip(raw_handles, raw_labels):
            if not lab or lab == "_nolegend_" or lab in seen:
                continue
            seen.add(lab)
            use_handles.append(handle)
            use_labels.append(lab)
    else:
        use_handles = list(handles)
        use_labels = list(labels)

    if not use_labels:
        return None
    wrapped = [wrap_label(lab, fonts.legend, width_in * 0.94) for lab in use_labels]
    legend = ax.legend(
        use_handles,
        wrapped,
        loc="upper center",
        bbox_to_anchor=(anchor_x, anchor_y),
        bbox_transform=fig.transFigure,
        ncol=max(1, int(ncol)),
        fontsize=fonts.legend,
        framealpha=framealpha,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.15,
        labelspacing=0.45,
        fancybox=False,
        edgecolor="0.75",
    )
    if legend is not None:
        legend.set_in_layout(False)
        legend.set_clip_on(False)
    return legend


def psth_table_bbox(ax: Any, fonts: LayoutFonts, n_data_rows: int) -> list[float]:
    """Axes-fraction bbox for a PSTH table sitting under xlabel, in reserved space."""
    fig_h = float(ax.figure.get_figheight())
    ax_h_in = max(ax.get_position().height * fig_h, 1e-6)
    table_in = table_block_in(fonts, n_data_rows)
    gap_in = xlabel_block_in(fonts) + 0.04
    height_frac = table_in / ax_h_in
    y0 = -(gap_in / ax_h_in) - height_frac
    return [0.0, y0, 1.0, height_frac]


def rasterize_data_artists(fig: Any) -> None:
    """Rasterize traces/collections; keep text, ticks, and legends as vectors."""
    skip: set[Any] = set()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", None) or []
        for handle in handles:
            skip.add(handle)
        for line in legend.get_lines():
            skip.add(line)
        for patch in legend.get_patches():
            skip.add(patch)
    for ax in fig.axes:
        if str(ax.get_title() or "").startswith("MEA"):
            continue
        for line in ax.get_lines():
            if line not in skip:
                line.set_rasterized(True)
        for coll in ax.collections:
            coll.set_rasterized(True)
        for patch in ax.patches:
            if patch not in skip:
                patch.set_rasterized(True)


def save_figure_to_pdf(
    pdf: PdfPages,
    fig: Any,
    *,
    dpi: int,
    soften_linewidths=None,
    apply_fonts=None,
) -> None:
    """Rasterize data, optionally restyle, then write one PDF page."""
    if soften_linewidths is not None:
        soften_linewidths(fig)
    if apply_fonts is not None:
        apply_fonts(fig)
    rasterize_data_artists(fig)
    pdf.savefig(
        fig,
        dpi=int(dpi),
        facecolor="white",
        edgecolor="none",
        pad_inches=0.12,
    )
    plt.close(fig)
