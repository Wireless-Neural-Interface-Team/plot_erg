"""Fenêtre principale : planche de blocs, panneaux de paramètres, menus."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QDockWidget, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

from . import config as C
from .block_widget import BlockWidget
from .config import BlockConfig, FilterSettings, LayoutSettings, TriggerSettings
from .engine import ComputeEngine
from .recording import DIGITAL_STREAMS, STREAM_LABELS, Recording, RhsRecording, SyntheticRecording
from .widgets import BoundForm, ChannelPicker, Field, scroll

TRIGGER_DEFAULT_THRESHOLDS = {"adc": 1.5, "dac": 1.5, "din": 0.5, "dout": 0.5, "amp": 200.0, "stim": 1.0, "dc": 0.5}

PRESETS = [
    ("trigger", "Signal de trigger (continu)"),
    ("wide", "WIDE moyen aligné"),
    ("low", "LOW moyen aligné"),
    ("high", "HIGH superposé"),
    ("raster", "SPIKES — raster"),
    ("psth", "SPIKES — PSTH"),
    ("trace", "SPIKES — trace continue"),
    ("scope", "SPIKE SCOPE"),
    ("rms", "RMS glissant aligné"),
    ("continuous", "WIDE continu"),
]


def make_block(kind: str, rec: Recording | None, trig: TriggerSettings) -> BlockConfig:
    b = BlockConfig()
    if kind == "trigger":
        stream = trig.source_stream if rec is None or trig.source_stream in rec.streams else "amp"
        b.title = "Signal de trigger"
        b.stream, b.channels, b.signal = stream, [trig.source_index], "wide"
        b.mode, b.time_unit, b.t_start_s, b.t_end_s = "continuous", "s", 0.0, 5.0
        b.baseline, b.legend_show = "none", False
    elif kind == "wide":
        b.title, b.signal, b.trial_display = "WIDE — moyenne ± SD", "wide", "mean+sd"
    elif kind == "low":
        b.title, b.signal, b.trial_display = "LOW — moyenne + essais", "low", "mean+overlay"
    elif kind == "high":
        b.title, b.signal, b.trial_display = "HIGH — essais superposés", "high", "overlay"
        b.legend_show = False
    elif kind in ("raster", "psth"):
        b.title = "SPIKES — raster" if kind == "raster" else "SPIKES — PSTH"
        b.signal, b.spikes_view = "spikes", kind
    elif kind == "trace":
        b.title, b.signal, b.spikes_view = "SPIKES — trace HIGH", "spikes", "trace"
        b.mode, b.time_unit, b.t_start_s, b.t_end_s = "continuous", "s", 0.0, 3.0
    elif kind == "scope":
        b.title, b.signal = "SPIKE SCOPE", "scope"
    elif kind == "rms":
        b.title, b.signal, b.trial_display = "RMS glissant", "rms", "mean+sem"
    elif kind == "continuous":
        b.title, b.signal = "WIDE continu", "wide"
        b.mode, b.time_unit, b.t_start_s, b.t_end_s = "continuous", "s", 0.0, 5.0
        b.baseline = "mean"
    return b


def default_blocks(rec: Recording | None, trig: TriggerSettings) -> list[BlockConfig]:
    blocks = [make_block(k, rec, trig) for k in ("trigger", "wide", "low", "high", "raster", "psth", "scope", "rms", "trace")]
    blocks[1].zoom_show = True
    blocks[1].zoom_x0, blocks[1].zoom_x1 = 0.0, 100.0
    blocks[3].sb_show = True
    blocks[3].sb_x_len, blocks[3].sb_y_len = 100.0, 50.0
    blocks[3].show_x_axis = blocks[3].show_y_axis = False
    blocks[6].sb_show = True
    blocks[6].sb_x_len, blocks[6].sb_y_len = 1.0, 50.0
    blocks[6].legend_show = False
    return blocks


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RHS Studio — traitement de données Intan RHS")
        self.resize(1680, 980)
        self.setAcceptDrops(True)
        self.settings = QSettings("plot_erg", "rhs_studio")

        self.rec: Recording | None = None
        self.blocks: list[BlockConfig] = []
        self.widgets: dict[str, BlockWidget] = {}
        self.selected_id: str | None = None
        self.filters = FilterSettings()
        self.trigger = TriggerSettings()
        self.layout_cfg = LayoutSettings()
        self.stats_dialog = None

        self.engine = ComputeEngine(self)
        self.engine.filters = self.filters
        self.engine.trigger = self.trigger
        self.engine.resultReady.connect(self._on_result)
        self.engine.failed.connect(self._on_failed)
        self.engine.triggersReady.connect(self._on_triggers)
        self.engine.triggerStarted.connect(lambda: self.statusBar().showMessage("Détection des triggers…"))

        self._pending: set[str] = set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._flush)

        self._build_canvas()
        self._build_docks()
        self._build_menus()
        self.file_label = QLabel()
        self.statusBar().addPermanentWidget(self.file_label)
        self._apply_layout_settings()

    # ================================================================ UI
    def _build_canvas(self) -> None:
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.canvas = QWidget()
        self.canvas.setObjectName("canvas")
        self.grid = QGridLayout(self.canvas)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.canvas)
        self.setCentralWidget(self.scroll_area)

    def _dock(self, title: str, widget: QWidget, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        self.addDockWidget(area, d)
        return d

    def _build_docks(self) -> None:
        # --- Liste des blocs
        w = QWidget()
        lay = QVBoxLayout(w)
        self.block_list = QListWidget()
        self.block_list.currentItemChanged.connect(self._list_selection)
        lay.addWidget(self.block_list)
        row = QHBoxLayout()
        add_btn = QToolButton()
        add_btn.setText("Ajouter ▾")
        add_btn.setPopupMode(QToolButton.InstantPopup)
        add_btn.setMenu(self._preset_menu())
        row.addWidget(add_btn)
        for text, slot in (
            ("Dupliquer", self.duplicate_selected),
            ("Supprimer", self.delete_selected),
            ("▲", lambda: self.move_selected(-1)),
            ("▼", lambda: self.move_selected(1)),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)

        self.layout_form = BoundForm([
            ("Planche", [
                Field("columns", "Colonnes", "int", minimum=1, maximum=12),
                Field("spacing", "Espacement", "int", minimum=0, maximum=80, suffix=" px"),
                Field("canvas_bg", "Fond de la planche", "color"),
                Field("antialias", "Anticrénelage", "bool"),
                Field("export_scale", "Échelle d'export PNG", "float", minimum=1, maximum=8, step=0.5, decimals=1, suffix=" ×"),
            ]),
        ])
        self.layout_form.bind(self.layout_cfg)
        self.layout_form.changed.connect(lambda _n: self._apply_layout_settings())
        for text, slot in (
            ("Appliquer les dimensions du bloc sélectionné à tous", self._apply_size_all),
            ("Appliquer le style du bloc sélectionné à tous", self._apply_style_all),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            self.layout_form.extra_layout().insertWidget(self.layout_form.extra_layout().count() - 1, b)
        lay.addWidget(self.layout_form)
        dock_blocks = self._dock("Blocs", w, Qt.LeftDockWidgetArea)

        # --- Propriétés du bloc
        self.block_tabs = QTabWidget()
        self.block_tabs.setUsesScrollButtons(True)
        self.block_forms: list[BoundForm] = []
        for title, groups in self._block_form_specs():
            form = BoundForm(groups)
            form.changed.connect(self._block_field_changed)
            self.block_forms.append(form)
            if title == "Données":
                self.channel_picker = ChannelPicker()
                self.channel_picker.changed.connect(self._channels_changed)
                box_lay = form.extra_layout()
                box_lay.insertWidget(2, QLabel("<b>Canaux</b> (cochez, puis cliquez la couleur pour la changer)"))
                box_lay.insertWidget(3, self.channel_picker)
            self.block_tabs.addTab(scroll(form), title)
        dock_props = self._dock("Propriétés du bloc", self.block_tabs, Qt.RightDockWidgetArea)

        # --- Trigger
        tw = QWidget()
        tl = QVBoxLayout(tw)
        self.trigger_form = BoundForm(self._trigger_form_specs())
        self.trigger_form.bind(self.trigger)
        self.trigger_form.changed.connect(self._trigger_changed)
        tl.addWidget(self.trigger_form)
        row = QHBoxLayout()
        detect = QPushButton("Détecter maintenant")
        detect.clicked.connect(self.engine.detect_now)
        auto = QPushButton("Seuil auto (mi-hauteur)")
        auto.clicked.connect(self._auto_threshold)
        row.addWidget(detect)
        row.addWidget(auto)
        tl.addLayout(row)
        self.trigger_status = QLabel("—")
        self.trigger_status.setWordWrap(True)
        tl.addWidget(self.trigger_status)
        dock_trig = self._dock("Trigger", scroll(tw), Qt.RightDockWidgetArea)

        # --- Filtres
        self.filter_form = BoundForm(self._filter_form_specs())
        self.filter_form.bind(self.filters)
        self.filter_form.changed.connect(lambda _n: self.schedule_all())
        dock_filt = self._dock("Filtres", scroll(self.filter_form), Qt.RightDockWidgetArea)

        self.tabifyDockWidget(dock_props, dock_trig)
        self.tabifyDockWidget(dock_trig, dock_filt)
        dock_props.raise_()
        self.resizeDocks([dock_blocks, dock_props], [300, 430], Qt.Horizontal)
        self.docks = [dock_blocks, dock_props, dock_trig, dock_filt]

    def _preset_menu(self) -> QMenu:
        menu = QMenu(self)
        for key, label in PRESETS:
            act = menu.addAction(label)
            act.triggered.connect(lambda _=False, k=key: self.add_block(make_block(k, self.rec, self.trigger)))
        return menu

    def _stream_choices(self, _obj=None):
        streams = self.rec.streams if self.rec else {"amp": []}
        return [(k, STREAM_LABELS[k]) for k in streams]

    def _trigger_channel_choices(self, obj):
        names = self.rec.streams.get(obj.source_stream, []) if self.rec else []
        return [(i, f"{i}  {n}") for i, n in enumerate(names)] or [(0, "—")]

    def _block_form_specs(self):
        trig = lambda o: o.mode == "triggered"  # noqa: E731
        cont = lambda o: o.mode == "continuous"  # noqa: E731
        return [
            ("Données", [
                ("Bloc", [Field("title", "Titre", "str")]),
                ("Source", [
                    Field("stream", "Flux", "choice", choices_fn=self._stream_choices),
                    Field("signal", "Signal", "choice", C.SIGNALS,
                          enabled_if=lambda o: o.stream not in DIGITAL_STREAMS),
                ]),
                ("Fenêtre temporelle", [
                    Field("mode", "Mode", "choice", C.MODES),
                    Field("time_unit", "Unité de temps", "choice", C.TIME_UNITS),
                    Field("pre_ms", "Pré-trigger", "float", minimum=0, maximum=600000, step=10, decimals=2, suffix=" ms", enabled_if=trig),
                    Field("post_ms", "Post-trigger", "float", minimum=0, maximum=600000, step=10, decimals=2, suffix=" ms", enabled_if=trig),
                    Field("t_start_s", "Début", "float", minimum=0, maximum=1e7, step=0.5, decimals=4, suffix=" s", enabled_if=cont),
                    Field("t_end_s", "Fin (≤ début = fin du fichier)", "float", minimum=0, maximum=1e7, step=0.5, decimals=4, suffix=" s", enabled_if=cont),
                ]),
                ("Essais", [
                    Field("trial_display", "Affichage", "choice", C.TRIAL_DISPLAYS,
                          enabled_if=lambda o: o.mode == "triggered" and o.signal not in ("scope",)),
                    Field("baseline", "Ligne de base", "choice", C.BASELINES),
                    Field("spikes_view", "Vue SPIKES", "choice", C.SPIKE_VIEWS, enabled_if=lambda o: o.signal == "spikes"),
                    Field("max_overlay", "Essais superposés max", "int", minimum=1, maximum=100000, step=10),
                ]),
            ]),
            ("Bloc", [
                ("Dimensions", [
                    Field("width", "Largeur", "int", minimum=200, maximum=4000, step=10, suffix=" px"),
                    Field("height", "Hauteur", "int", minimum=150, maximum=4000, step=10, suffix=" px"),
                ]),
                ("Titre", [
                    Field("title_show", "Afficher", "bool"),
                    Field("title_size", "Taille", "int", minimum=6, maximum=48, suffix=" pt"),
                    Field("title_color", "Couleur", "color"),
                    Field("title_bold", "Gras", "bool"),
                    Field("title_align", "Alignement", "choice", C.ALIGNMENTS),
                ]),
                ("Cadre", [
                    Field("bg_color", "Fond", "color"),
                    Field("frame_color", "Bordure", "color"),
                    Field("frame_width", "Épaisseur bordure", "int", minimum=0, maximum=10, suffix=" px"),
                    Field("info_show", "Ligne d'informations", "bool"),
                ]),
            ]),
            ("Courbes", [
                ("Couleurs", [
                    Field("palette", "Palette seaborn", "choice", [(p, p) for p in C.SEABORN_PALETTES]),
                ]),
                ("Épaisseurs", [
                    Field("line_width", "Trace", "float", minimum=0.1, maximum=20, step=0.25, decimals=2, suffix=" px"),
                    Field("mean_width", "Moyenne", "float", minimum=0.1, maximum=20, step=0.25, decimals=2, suffix=" px"),
                    Field("trial_width", "Essais", "float", minimum=0.1, maximum=20, step=0.25, decimals=2, suffix=" px"),
                ]),
                ("Essais et bande", [
                    Field("trial_alpha", "Opacité essais", "float", minimum=0.01, maximum=1, step=0.05, decimals=2),
                    Field("trial_use_channel_color", "Couleur du canal", "bool"),
                    Field("trial_color", "Couleur des essais", "color", enabled_if=lambda o: not o.trial_use_channel_color),
                    Field("band_alpha", "Opacité bande ± ", "float", minimum=0.0, maximum=1, step=0.05, decimals=2),
                ]),
                ("Spikes et seuil", [
                    Field("spike_color", "Couleur des marqueurs", "color"),
                    Field("spike_size", "Taille marqueurs / raster", "int", minimum=1, maximum=40),
                    Field("threshold_show", "Afficher le seuil", "bool"),
                    Field("threshold_color", "Couleur du seuil", "color"),
                    Field("threshold_width", "Épaisseur du seuil", "float", minimum=0.1, maximum=10, step=0.25, decimals=2),
                    Field("threshold_style", "Style du seuil", "choice", C.LINE_STYLES),
                ]),
            ]),
            ("Axes", [
                ("Titres des axes (vide = automatique)", [
                    Field("x_label", "Axe X", "str"),
                    Field("y_label", "Axe Y", "str"),
                    Field("axis_label_size", "Taille des titres", "int", minimum=5, maximum=40, suffix=" pt"),
                    Field("tick_size", "Taille des graduations", "int", minimum=5, maximum=40, suffix=" pt"),
                    Field("axis_color", "Couleur", "color"),
                    Field("axis_width", "Épaisseur", "float", minimum=0.1, maximum=10, step=0.25, decimals=2),
                    Field("show_x_axis", "Afficher l'axe X", "bool"),
                    Field("show_y_axis", "Afficher l'axe Y", "bool"),
                    Field("grid", "Grille", "bool"),
                    Field("grid_alpha", "Opacité grille", "float", minimum=0, maximum=1, step=0.05, decimals=2, enabled_if=lambda o: o.grid),
                ]),
                ("Échelles", [
                    Field("x_auto", "X automatique", "bool"),
                    Field("x_min", "X min", "float", decimals=4, step=10, enabled_if=lambda o: not o.x_auto),
                    Field("x_max", "X max", "float", decimals=4, step=10, enabled_if=lambda o: not o.x_auto),
                    Field("y_auto", "Y automatique", "bool"),
                    Field("y_min", "Y min", "float", decimals=4, step=10, enabled_if=lambda o: not o.y_auto),
                    Field("y_max", "Y max", "float", decimals=4, step=10, enabled_if=lambda o: not o.y_auto),
                ]),
            ]),
            ("Échelle", [
                ("Échelle flottante (glisser pour déplacer)", [
                    Field("sb_show", "Afficher", "bool"),
                    Field("sb_show_x", "Barre horizontale", "bool"),
                    Field("sb_x_len", "Longueur X", "float", minimum=0, decimals=4, step=1),
                    Field("sb_x_unit", "Unité X (vide = auto)", "str"),
                    Field("sb_show_y", "Barre verticale", "bool"),
                    Field("sb_y_len", "Longueur Y", "float", minimum=0, decimals=4, step=1),
                    Field("sb_y_unit", "Unité Y (vide = auto)", "str"),
                    Field("sb_color", "Couleur", "color"),
                    Field("sb_width", "Épaisseur", "float", minimum=0.5, maximum=20, step=0.5, decimals=1, suffix=" px"),
                    Field("sb_font_size", "Taille du texte", "int", minimum=5, maximum=40, suffix=" pt"),
                    Field("sb_fx", "Position X (0–1)", "float", minimum=0, maximum=1, step=0.02, decimals=3),
                    Field("sb_fy", "Position Y (0–1)", "float", minimum=0, maximum=1, step=0.02, decimals=3),
                ]),
            ]),
            ("Légende", [
                ("Légende", [
                    Field("legend_show", "Afficher", "bool"),
                    Field("legend_pos", "Position", "choice", C.LEGEND_POSITIONS),
                    Field("legend_size", "Taille du texte", "int", minimum=5, maximum=30, suffix=" pt"),
                    Field("legend_text_color", "Couleur du texte", "color"),
                    Field("legend_bg", "Fond", "color_alpha"),
                    Field("legend_frame", "Bordure", "color_alpha"),
                ]),
            ]),
            ("Zoom", [
                ("Zone de zoom (glisser la zone ou zoomer dans l'encart)", [
                    Field("zoom_show", "Afficher", "bool"),
                    Field("zoom_x0", "Début", "float", decimals=4, step=5),
                    Field("zoom_x1", "Fin", "float", decimals=4, step=5),
                    Field("zoom_side", "Position de l'encart", "choice", C.ZOOM_SIDES),
                    Field("zoom_ratio", "Taille de l'encart", "float", minimum=0.1, maximum=0.8, step=0.05, decimals=2),
                    Field("zoom_color", "Couleur", "color"),
                    Field("zoom_alpha", "Opacité de la zone", "float", minimum=0, maximum=1, step=0.05, decimals=2),
                    Field("zoom_y_auto", "Y automatique dans l'encart", "bool"),
                ]),
            ]),
            ("Trigger", [
                ("Ligne trigger ON", [
                    Field("trig_on_show", "Afficher", "bool"),
                    Field("trig_on_color", "Couleur", "color"),
                    Field("trig_on_width", "Épaisseur", "float", minimum=0.1, maximum=20, step=0.25, decimals=2),
                    Field("trig_on_style", "Style", "choice", C.LINE_STYLES),
                ]),
                ("Ligne trigger OFF", [
                    Field("trig_off_show", "Afficher", "bool"),
                    Field("trig_off_color", "Couleur", "color"),
                    Field("trig_off_width", "Épaisseur", "float", minimum=0.1, maximum=20, step=0.25, decimals=2),
                    Field("trig_off_style", "Style", "choice", C.LINE_STYLES),
                ]),
                ("Options", [
                    Field("trig_label_show", "Étiquettes ON / OFF", "bool"),
                    Field("trig_max_lines", "Lignes max (mode continu)", "int", minimum=1, maximum=10000),
                ]),
            ]),
        ]

    def _trigger_form_specs(self):
        return [
            ("Source", [
                Field("enabled", "Trigger actif", "bool"),
                Field("source_stream", "Flux", "choice", choices_fn=self._stream_choices),
                Field("source_index", "Canal", "choice", choices_fn=self._trigger_channel_choices),
                Field("threshold", "Seuil (unité du flux)", "float", decimals=4, step=0.1),
                Field("edge", "Front ON", "choice", [
                    ("rising", "Montant (actif au-dessus du seuil)"),
                    ("falling", "Descendant (actif sous le seuil)"),
                ]),
                Field("use_abs", "Valeur absolue", "bool", tooltip="Utile pour les canaux de stimulation biphasiques"),
            ]),
            ("Validation", [
                Field("min_duration_ms", "Durée minimale ON", "float", minimum=0, maximum=1e6, decimals=3, suffix=" ms"),
                Field("min_interval_ms", "Intervalle minimal (temps mort)", "float", minimum=0, maximum=1e7, decimals=3, step=10, suffix=" ms"),
                Field("search_start_s", "Recherche à partir de", "float", minimum=0, maximum=1e7, decimals=3, suffix=" s"),
                Field("search_end_s", "Recherche jusqu'à (0 = fin)", "float", minimum=0, maximum=1e7, decimals=3, suffix=" s"),
            ]),
            ("Sélection des essais", [
                Field("first_trial", "Premier essai", "int", minimum=1, maximum=10**7),
                Field("max_trials", "Nombre max (0 = tous)", "int", minimum=0, maximum=10**7),
                Field("step", "Un essai sur", "int", minimum=1, maximum=1000),
                Field("exclude", "Exclure (ex. 3, 7-9)", "str"),
            ]),
        ]

    def _filter_form_specs(self):
        return [
            ("WIDE", [
                Field("notch_hz", "Coupe-bande secteur", "choice", [(0.0, "Aucun"), (50.0, "50 Hz"), (60.0, "60 Hz")]),
                Field("phase", "Phase des filtres", "choice", [
                    ("zero", "Zéro-phase (aller-retour)"), ("causal", "Causal (comme Intan RHX)"),
                ]),
            ]),
            ("LOW (passe-bas)", [
                Field("low_type", "Type", "choice", C.FILTER_TYPES),
                Field("low_order", "Ordre", "int", minimum=1, maximum=10),
                Field("low_cutoff_hz", "Coupure", "float", minimum=0.1, maximum=1e5, decimals=2, step=10, suffix=" Hz"),
            ]),
            ("HIGH (passe-haut)", [
                Field("high_type", "Type", "choice", C.FILTER_TYPES),
                Field("high_order", "Ordre", "int", minimum=1, maximum=10),
                Field("high_cutoff_hz", "Coupure", "float", minimum=0.1, maximum=1e5, decimals=2, step=10, suffix=" Hz"),
            ]),
            ("SPIKES", [
                Field("spike_threshold_mode", "Mode de seuil", "choice", [
                    ("rms", "× RMS du HIGH"), ("mad", "× MAD (robuste)"), ("uv", "Absolu (µV)"),
                ]),
                Field("spike_threshold", "Valeur du seuil", "float", minimum=0, maximum=1e5, decimals=2, step=0.5),
                Field("spike_polarity", "Polarité", "choice", [
                    ("negative", "Négative"), ("positive", "Positive"), ("both", "Les deux"),
                ]),
                Field("spike_refractory_ms", "Période réfractaire", "float", minimum=0.05, maximum=1000, decimals=2, step=0.1, suffix=" ms"),
                Field("spike_artifact_uv", "Rejet artefacts (0 = non)", "float", minimum=0, maximum=1e6, decimals=1, step=100, suffix=" µV"),
                Field("psth_bin_ms", "Largeur de classe PSTH", "float", minimum=0.1, maximum=1e5, decimals=2, step=1, suffix=" ms"),
            ]),
            ("SPIKE SCOPE", [
                Field("scope_pre_ms", "Avant le spike", "float", minimum=0.05, maximum=100, decimals=2, step=0.1, suffix=" ms"),
                Field("scope_post_ms", "Après le spike", "float", minimum=0.05, maximum=100, decimals=2, step=0.1, suffix=" ms"),
                Field("scope_max_waveforms", "Formes d'onde max", "int", minimum=1, maximum=100000, step=50),
                Field("scope_align", "Alignement", "choice", [("threshold", "Franchissement du seuil"), ("peak", "Pic")]),
            ]),
            ("RMS", [
                Field("rms_window_ms", "Fenêtre glissante", "float", minimum=0.1, maximum=1e5, decimals=2, step=5, suffix=" ms"),
                Field("rms_source", "Signal source", "choice", [("high", "HIGH"), ("low", "LOW"), ("wide", "WIDE")]),
            ]),
        ]

    def _build_menus(self) -> None:
        mb = self.menuBar()

        def act(menu, text, slot, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        m = mb.addMenu("&Fichier")
        act(m, "Ouvrir un fichier RHS…", self.open_dialog, "Ctrl+O")
        act(m, "Données de démonstration", self.load_demo)
        m.addSeparator()
        act(m, "Charger une configuration…", self.load_config_dialog, "Ctrl+Shift+O")
        act(m, "Enregistrer la configuration…", self.save_config_dialog, "Ctrl+S")
        m.addSeparator()
        act(m, "Exporter le bloc sélectionné (PNG)…", lambda: self._export_block("png"), "Ctrl+E")
        act(m, "Exporter le bloc sélectionné (SVG)…", lambda: self._export_block("svg"))
        act(m, "Exporter toute la planche (PNG)…", self.export_board, "Ctrl+Shift+E")
        m.addSeparator()
        act(m, "Quitter", self.close, "Ctrl+Q")

        m = mb.addMenu("&Blocs")
        sub = m.addMenu("Ajouter un bloc")
        for key, label in PRESETS:
            a = sub.addAction(label)
            a.triggered.connect(lambda _=False, k=key: self.add_block(make_block(k, self.rec, self.trigger)))
        act(m, "Dupliquer le bloc", self.duplicate_selected, "Ctrl+D")
        act(m, "Supprimer le bloc", self.delete_selected, "Ctrl+Del")
        act(m, "Recalculer tous les blocs", self.schedule_all, "F5")
        act(m, "Réinitialiser la planche par défaut", self.reset_blocks)

        m = mb.addMenu("&Analyse")
        act(m, "Détecter les triggers", self.engine.detect_now, "F6")
        act(m, "Statistiques (seaborn)…", self.open_stats, "Ctrl+T")

        m = mb.addMenu("&Affichage")
        for d in self.docks:
            m.addAction(d.toggleViewAction())

    # ========================================================= enregistrement
    def open_dialog(self) -> None:
        start = self.settings.value("last_dir", str(Path.cwd()))
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un fichier Intan", start, "Intan RHS (*.rhs);;Tous (*)")
        if path:
            self.open_recording(path)

    def open_recording(self, path: str) -> None:
        try:
            rec = RhsRecording(path)
        except Exception as exc:
            QMessageBox.critical(self, "Lecture impossible", f"{path}\n\n{exc}")
            return
        if rec.n_samples == 0:
            QMessageBox.warning(self, "Fichier vide", "Ce fichier RHS ne contient aucune donnée.")
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        self.set_recording(rec)

    def load_demo(self) -> None:
        self.set_recording(SyntheticRecording())

    def set_recording(self, rec: Recording, adapt: bool = True) -> None:
        self.rec = rec
        self.engine.set_recording(rec)
        if adapt:
            self._adapt_trigger()
        for b in self.blocks:
            self._adapt_block(b)
        self.trigger_form.reload()
        self.trigger_status.setText("—")
        self.file_label.setText(rec.describe())
        self.setWindowTitle(f"RHS Studio — {rec.name}")
        if not self.blocks:
            for b in default_blocks(rec, self.trigger):
                self.add_block(b, select=False, schedule=False)
            self._relayout()
        if self._block(self.selected_id) is None and self.blocks:
            self.select_block(self.blocks[0].id)
        else:
            self._bind_selected()
        self.schedule_all()

    def _adapt_trigger(self) -> None:
        t, rec = self.trigger, self.rec
        if t.source_stream in rec.streams and t.source_index < len(rec.streams[t.source_stream]):
            return
        for stream in ("adc", "din", "amp"):
            if rec.streams.get(stream):
                t.source_stream, t.source_index = stream, 0
                t.threshold = TRIGGER_DEFAULT_THRESHOLDS[stream]
                return

    def _adapt_block(self, b: BlockConfig) -> None:
        if b.stream not in self.rec.streams:
            b.stream = "amp" if "amp" in self.rec.streams else next(iter(self.rec.streams))
        n = len(self.rec.streams[b.stream])
        b.channels = [c for c in b.channels if 0 <= c < n] or [0]
        b.colors = b.colors[: len(b.channels)]

    def _auto_threshold(self) -> None:
        if self.rec is None:
            return
        t = self.trigger
        names = self.rec.streams.get(t.source_stream, [])
        if not names:
            return
        n = min(self.rec.n_samples, int(120 * self.rec.fs))
        x = self.rec.read(t.source_stream, t.source_index, 0, n)
        if t.use_abs:
            x = np.abs(x)
        lo, hi = np.percentile(x, [0.5, 99.5])
        t.threshold = float((lo + hi) / 2)
        self.trigger_form.reload()
        self._trigger_changed("threshold")

    # ================================================================ blocs
    def add_block(self, cfg: BlockConfig, select: bool = True, schedule: bool = True) -> None:
        if self.rec is not None:
            self._adapt_block(cfg)
        self.blocks.append(cfg)
        w = BlockWidget(cfg)
        w.clicked.connect(self.select_block)
        w.zoomRegionChanged.connect(self._zoom_region_changed)
        w.scaleBarMoved.connect(self._scale_bar_moved)
        w.contextAction.connect(self._block_context)
        self.widgets[cfg.id] = w
        item = QListWidgetItem(cfg.title)
        item.setData(Qt.UserRole, cfg.id)
        self.block_list.addItem(item)
        if select:
            self._relayout()
            self.select_block(cfg.id)
            self.scroll_area.ensureWidgetVisible(w)
        if schedule:
            self.schedule(cfg.id)

    def _block(self, block_id: str | None) -> BlockConfig | None:
        return next((b for b in self.blocks if b.id == block_id), None)

    def select_block(self, block_id: str) -> None:
        if self.selected_id == block_id:
            return
        if self.selected_id in self.widgets:
            self.widgets[self.selected_id].set_selected(False)
        self.selected_id = block_id
        if block_id in self.widgets:
            self.widgets[block_id].set_selected(True)
        for i in range(self.block_list.count()):
            if self.block_list.item(i).data(Qt.UserRole) == block_id:
                self.block_list.blockSignals(True)
                self.block_list.setCurrentRow(i)
                self.block_list.blockSignals(False)
        self._bind_selected()
        if self.stats_dialog is not None and self.stats_dialog.isVisible():
            self.stats_dialog.select_block(block_id)

    def _list_selection(self, cur, _prev) -> None:
        if cur is not None:
            bid = cur.data(Qt.UserRole)
            self.select_block(bid)
            if bid in self.widgets:
                self.scroll_area.ensureWidgetVisible(self.widgets[bid])

    def _bind_selected(self) -> None:
        cfg = self._block(self.selected_id)
        self.block_tabs.setEnabled(cfg is not None)
        if cfg is None:
            return
        for form in self.block_forms:
            form.bind(cfg)
        self._refresh_picker(cfg)

    def _refresh_picker(self, cfg: BlockConfig) -> None:
        names = self.rec.streams.get(cfg.stream, []) if self.rec else []
        self.channel_picker.set_channels(names, cfg.channels, cfg.colors, cfg.palette)

    def _block_field_changed(self, name: str) -> None:
        cfg = self._block(self.selected_id)
        if cfg is None:
            return
        if name == "stream":
            cfg.channels, cfg.colors = [0], []
            if cfg.stream in DIGITAL_STREAMS:
                cfg.signal = "wide"
            self._refresh_picker(cfg)
            for form in self.block_forms:
                form.reload()
        elif name == "palette":
            cfg.colors = []
            self.channel_picker.set_palette(cfg.palette)
            return
        elif name == "title":
            item = self.block_list.currentItem()
            if item is not None:
                item.setText(cfg.title)
        elif name in ("mode", "signal"):
            for form in self.block_forms:
                form._refresh_enabled()
        if name in C.DATA_FIELDS:
            self.schedule(cfg.id)
        else:
            self.widgets[cfg.id].redraw()

    def _channels_changed(self) -> None:
        cfg = self._block(self.selected_id)
        if cfg is None:
            return
        chans, colors = self.channel_picker.selection()
        changed = chans != list(cfg.channels)
        cfg.channels, cfg.colors = chans, colors
        if changed:
            self.schedule(cfg.id)
        else:
            self.widgets[cfg.id].redraw()

    def _zoom_region_changed(self, block_id: str, *_):
        if block_id == self.selected_id:
            for form in self.block_forms:
                if "zoom_x0" in form._widgets:
                    form.reload()

    def _scale_bar_moved(self, block_id: str, *_):
        if block_id == self.selected_id:
            for form in self.block_forms:
                if "sb_fx" in form._widgets:
                    form.reload()

    def _block_context(self, block_id: str, action: str) -> None:
        self.select_block(block_id)
        cfg, w = self._block(block_id), self.widgets.get(block_id)
        if cfg is None or w is None:
            return
        if action == "freeze":
            (cfg.x_min, cfg.x_max), (cfg.y_min, cfg.y_max) = w.current_ranges()
            cfg.x_auto = cfg.y_auto = False
            w.redraw()
            self._bind_selected()
        elif action == "auto":
            cfg.x_auto = cfg.y_auto = True
            w._new_data = True
            w.redraw()
            self._bind_selected()
        elif action == "export_png":
            self._export_block("png")
        elif action == "export_svg":
            self._export_block("svg")
        elif action == "duplicate":
            self.duplicate_selected()
        elif action == "delete":
            self.delete_selected()

    def duplicate_selected(self) -> None:
        cfg = self._block(self.selected_id)
        if cfg is None:
            return
        dup = copy.deepcopy(cfg)
        dup.id = C.new_id()
        dup.title = f"{cfg.title} (copie)"
        idx = self.blocks.index(cfg) + 1
        self.add_block(dup, select=False)
        self.blocks.insert(idx, self.blocks.pop())
        item = self.block_list.takeItem(self.block_list.count() - 1)
        self.block_list.insertItem(idx, item)
        self._relayout()
        self.select_block(dup.id)

    def delete_selected(self) -> None:
        cfg = self._block(self.selected_id)
        if cfg is None:
            return
        idx = self.blocks.index(cfg)
        self.blocks.pop(idx)
        self.block_list.takeItem(idx)
        w = self.widgets.pop(cfg.id)
        self.engine.forget(cfg.id)
        w.setParent(None)
        w.deleteLater()
        self.selected_id = None
        self._relayout()
        if self.blocks:
            self.select_block(self.blocks[min(idx, len(self.blocks) - 1)].id)
        else:
            self._bind_selected()

    def move_selected(self, delta: int) -> None:
        cfg = self._block(self.selected_id)
        if cfg is None:
            return
        i = self.blocks.index(cfg)
        j = i + delta
        if not 0 <= j < len(self.blocks):
            return
        self.blocks[i], self.blocks[j] = self.blocks[j], self.blocks[i]
        item = self.block_list.takeItem(i)
        self.block_list.insertItem(j, item)
        self.block_list.blockSignals(True)
        self.block_list.setCurrentRow(j)
        self.block_list.blockSignals(False)
        self._relayout()

    def clear_blocks(self) -> None:
        for w in self.widgets.values():
            w.setParent(None)
            w.deleteLater()
        for b in self.blocks:
            self.engine.forget(b.id)
        self.blocks.clear()
        self.widgets.clear()
        self.block_list.clear()
        self.selected_id = None

    def reset_blocks(self) -> None:
        self.clear_blocks()
        for b in default_blocks(self.rec, self.trigger):
            self.add_block(b, select=False)
        self._relayout()
        if self.blocks:
            self.select_block(self.blocks[0].id)

    def _apply_size_all(self) -> None:
        src = self._block(self.selected_id)
        if src is None:
            return
        for b in self.blocks:
            b.width, b.height = src.width, src.height
            self.widgets[b.id].redraw()
        self._relayout()

    def _apply_style_all(self) -> None:
        src = self._block(self.selected_id)
        if src is None:
            return
        for b in self.blocks:
            if b is not src:
                C.copy_style(src, b)
                self.widgets[b.id].redraw()
        self._relayout()

    def _relayout(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        cols = max(1, int(self.layout_cfg.columns))
        for i, b in enumerate(self.blocks):
            self.grid.addWidget(self.widgets[b.id], i // cols, i % cols, Qt.AlignTop | Qt.AlignLeft)
        sp = int(self.layout_cfg.spacing)
        self.grid.setSpacing(sp)
        self.grid.setContentsMargins(sp, sp, sp, sp)

    def _apply_layout_settings(self) -> None:
        pg.setConfigOptions(antialias=bool(self.layout_cfg.antialias))
        self.canvas.setStyleSheet(f"QWidget#canvas {{ background: {self.layout_cfg.canvas_bg}; }}")
        self._relayout()

    # ================================================================ calcul
    def schedule(self, block_id: str) -> None:
        self._pending.add(block_id)
        if block_id in self.widgets:
            self.widgets[block_id].set_busy(True)
        self._timer.start()

    def schedule_all(self) -> None:
        for b in self.blocks:
            self.schedule(b.id)

    def _flush(self) -> None:
        pending, self._pending = self._pending, set()
        for bid in pending:
            cfg = self._block(bid)
            if cfg is not None:
                self.engine.submit(cfg)

    def _trigger_changed(self, name: str) -> None:
        if name == "source_stream":
            self.trigger.source_index = 0
            self.trigger.threshold = TRIGGER_DEFAULT_THRESHOLDS.get(self.trigger.source_stream, 1.0)
            self.trigger_form.reload()
        cached = self.engine.cached_triggers()
        if cached is not None:
            self._on_triggers(cached)
        else:
            self.trigger_status.setText("Paramètres modifiés : nouvelle détection…")
        self.schedule_all()

    def _on_result(self, block_id: str, gen: int, data) -> None:
        if not self.engine.is_current(block_id, gen) or block_id not in self.widgets:
            return
        self.widgets[block_id].set_data(data)
        self.statusBar().showMessage("Calcul terminé", 2000)
        if self.stats_dialog is not None and self.stats_dialog.isVisible():
            if self.stats_dialog.block_combo.currentData() == block_id:
                self.stats_dialog.redraw()

    def _on_failed(self, block_id: str, gen: int, msg: str) -> None:
        print(msg, file=sys.stderr)
        if not block_id:
            self.statusBar().showMessage(msg.splitlines()[0], 8000)
            return
        if self.engine.is_current(block_id, gen) and block_id in self.widgets:
            self.widgets[block_id].set_error(msg.splitlines()[0])

    def _on_triggers(self, res) -> None:
        from .dsp import select_trials

        if res.count == 0:
            self.trigger_status.setText(
                "<span style='color:#cf222e'>Aucun trigger détecté.</span> Ajustez le seuil ou la source."
            )
            self.statusBar().showMessage("Aucun trigger détecté", 4000)
            return
        on, _off, _n = select_trials(res, self.trigger)
        dur = res.durations_ms()
        isi = np.diff(res.on) / res.fs if res.count > 1 else np.empty(0)
        lines = [f"<b>{res.count}</b> triggers détectés ({on.size} sélectionnés)"]
        if dur.size:
            lines.append(f"Durée ON : médiane {np.median(dur):.2f} ms (min {dur.min():.2f}, max {dur.max():.2f})")
        if isi.size:
            lines.append(f"Intervalle : médiane {np.median(isi):.3f} s")
        lines.append(f"Premier à {res.on[0] / res.fs:.3f} s, dernier à {res.on[-1] / res.fs:.3f} s")
        self.trigger_status.setText("<br>".join(lines))
        self.statusBar().showMessage(f"{res.count} triggers détectés", 4000)

    # ========================================================= statistiques
    def open_stats(self) -> None:
        from .stats_window import StatsDialog

        if self.stats_dialog is None:
            self.stats_dialog = StatsDialog(lambda: [(b, self.widgets[b.id].data) for b in self.blocks], self)
        self.stats_dialog.refresh_blocks()
        if self.selected_id:
            self.stats_dialog.select_block(self.selected_id)
        self.stats_dialog.show()
        self.stats_dialog.raise_()

    # ============================================================= exports
    def _export_block(self, fmt: str) -> None:
        cfg = self._block(self.selected_id)
        if cfg is None:
            return
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in cfg.title).strip() or "bloc"
        filt = "PNG (*.png)" if fmt == "png" else "SVG (*.svg)"
        path, _ = QFileDialog.getSaveFileName(self, "Exporter le bloc", f"{safe}.{fmt}", filt)
        if not path:
            return
        w = self.widgets[cfg.id]
        if fmt == "svg":
            w.export_svg(path)
        else:
            w.render_image(self.layout_cfg.export_scale).save(path)
        self.statusBar().showMessage(f"Exporté : {path}", 5000)

    def export_board(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exporter la planche", "planche.png", "PNG (*.png)")
        if not path:
            return
        sel = self.selected_id
        if sel in self.widgets:
            self.widgets[sel].set_selected(False)
        scale = self.layout_cfg.export_scale
        size = self.canvas.size()
        img = QImage(size * scale, QImage.Format_ARGB32)
        img.setDevicePixelRatio(scale)
        img.fill(Qt.white)
        painter = QPainter(img)
        self.canvas.render(painter, QPoint())
        painter.end()
        img.save(path)
        if sel in self.widgets:
            self.widgets[sel].set_selected(True)
        self.statusBar().showMessage(f"Planche exportée : {path}", 5000)

    # ======================================================= configuration
    def save_config_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la configuration", "rhs_studio.json", "JSON (*.json)")
        if path:
            self.save_config(path)

    def save_config(self, path: str) -> None:
        payload = {
            "app": "rhs_studio",
            "version": 1,
            "recording": getattr(self.rec, "path", "") if self.rec else "",
            "filters": C.to_dict(self.filters),
            "trigger": C.to_dict(self.trigger),
            "layout": C.to_dict(self.layout_cfg),
            "blocks": [C.to_dict(b) for b in self.blocks],
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.statusBar().showMessage(f"Configuration enregistrée : {path}", 5000)

    def load_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Charger une configuration", "", "JSON (*.json)")
        if path:
            self.load_config(path)

    def load_config(self, path: str) -> None:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Configuration illisible", str(exc))
            return
        self._copy_into(self.filters, C.from_dict(FilterSettings, raw.get("filters")))
        self._copy_into(self.trigger, C.from_dict(TriggerSettings, raw.get("trigger")))
        self._copy_into(self.layout_cfg, C.from_dict(LayoutSettings, raw.get("layout")))
        self.filter_form.reload()
        self.layout_form.reload()
        self.clear_blocks()
        for rb in raw.get("blocks", []):
            self.add_block(C.from_dict(BlockConfig, rb), select=False, schedule=False)
        self._apply_layout_settings()

        rec_path = raw.get("recording") or ""
        current = getattr(self.rec, "path", "") if self.rec else ""
        if rec_path and Path(rec_path).exists() and rec_path != current:
            answer = QMessageBox.question(
                self, "Ouvrir l'enregistrement ?",
                f"Cette configuration a été créée avec :\n{rec_path}\n\nL'ouvrir maintenant ?",
            )
            if answer == QMessageBox.Yes:
                try:
                    self.engine.set_recording(None)
                    self.set_recording(RhsRecording(rec_path), adapt=False)
                except Exception as exc:
                    QMessageBox.critical(self, "Lecture impossible", str(exc))
        elif self.rec is not None:
            self.set_recording(self.rec, adapt=False)
        self.trigger_form.reload()
        if self.blocks:
            self.select_block(self.blocks[0].id)

    @staticmethod
    def _copy_into(dst, src) -> None:
        for k, v in vars(src).items():
            setattr(dst, k, v)

    # ============================================================ divers
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".rhs"):
                self.open_recording(path)
            elif path.lower().endswith(".json"):
                self.load_config(path)

    def closeEvent(self, ev):
        self.engine.shutdown()
        super().closeEvent(ev)
