from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import winsound
except ImportError:  # pragma: no cover - Windows is the target platform.
    winsound = None


# ================= PATHS =================

TOOLS_DIR = Path(__file__).resolve().parents[2]

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from acquisition.gui.csi_parser import CSIFrameParser
from csi.csi_binary_io import write_packets


# ================= CONFIG =================

DEFAULT_BAUD = 921600

HISTORY_SIZE = 150
PLOT_UPDATE_MS = 100
MAX_EVENTS_PER_UPDATE = 250
EVENT_QUEUE_MAX_SIZE = 5000
SERIAL_READ_CHUNK_SIZE = 4096
COLLECTION_FINALIZE_GRACE_SECONDS = 0.75
COLLECTION_FINALIZE_MAX_SECONDS = 3.0
PROGRAMMED_NEXT_DELAY_MS = 1500

SUBCARRIERS_TO_PLOT = [0, 5, 10, 15, 20, 25]

PROGRAMMED_CLASS_ORDERS = (
    ("empty", "static_presence", "movement"),
    ("movement", "empty", "static_presence"),
    ("static_presence", "movement", "empty"),
)

CLASS_SPOKEN_NAMES = {
    "empty": "Empty",
    "static_presence": "Static",
    "movement": "Movement",
}

# Used only as a fallback if Windows TTS is unavailable.
CLASS_BEEP_COUNTS = {
    "empty": 1,
    "static_presence": 2,
    "movement": 3,
}

DATASET_DIR = TOOLS_DIR / "datasets"
RAW_BIN_DIR = DATASET_DIR / "raw_bin"
DEBUG_CSV_DIR = DATASET_DIR / "debug_csv"


# ================= SERIAL READER =================

class SerialReader:
    def __init__(self, port: str, baud: int, event_queue: queue.Queue):
        self.port = port
        self.baud = baud
        self.event_queue = event_queue

        self.running = False
        self.thread: threading.Thread | None = None
        self.serial_conn: serial.Serial | None = None

        self.parser = CSIFrameParser()
        self.pc_queue_drops = 0
        self.serial_errors = 0
        self.last_error = ""

        self._esp_to_pc_offset: float | None = None
        self._last_esp_timestamp_us: int | None = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.running = False

        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def diagnostics(self) -> dict[str, int | str]:
        diagnostics: dict[str, int | str] = self.parser.diagnostics()
        diagnostics["pc_queue_drops"] = self.pc_queue_drops
        diagnostics["serial_errors"] = self.serial_errors
        diagnostics["last_error"] = self.last_error
        return diagnostics

    def _read_loop(self) -> None:
        try:
            self.serial_conn = serial.Serial(
                self.port,
                self.baud,
                timeout=0.05,
                write_timeout=0.5,
            )

            # Do not toggle DTR/RTS after opening. Some USB bridges use these
            # lines to reset the ESP32-S3.
            self.serial_conn.dtr = False
            self.serial_conn.rts = False

            while self.running:
                try:
                    waiting = self.serial_conn.in_waiting
                    read_size = min(
                        max(waiting, 1),
                        SERIAL_READ_CHUNK_SIZE,
                    )

                    data = self.serial_conn.read(read_size)

                    if not data:
                        continue

                    for event in self.parser.feed(data):
                        self._attach_timestamps(event)
                        self._enqueue_event(event)

                except (serial.SerialException, OSError) as exc:
                    self.serial_errors += 1
                    self.last_error = str(exc)
                    break

                except Exception as exc:
                    self.serial_errors += 1
                    self.last_error = str(exc)

        except Exception as exc:
            self.serial_errors += 1
            self.last_error = str(exc)

        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()

            self.running = False

    def _attach_timestamps(self, event: dict) -> None:
        pc_timestamp = time.time()
        metadata = event.get("metadata", {})
        esp_timestamp_us = int(metadata.get("timestamp_us", 0) or 0)

        event["pc_timestamp"] = pc_timestamp
        event["capture_timestamp"] = pc_timestamp

        if esp_timestamp_us <= 0:
            return

        # Recalculate the offset if the ESP32 restarted and its timer returned
        # to a lower value.
        if (
            self._last_esp_timestamp_us is None
            or esp_timestamp_us < self._last_esp_timestamp_us
            or self._esp_to_pc_offset is None
        ):
            self._esp_to_pc_offset = pc_timestamp - (esp_timestamp_us / 1e6)

        self._last_esp_timestamp_us = esp_timestamp_us
        event["capture_timestamp"] = (
            self._esp_to_pc_offset + (esp_timestamp_us / 1e6)
        )

    def _enqueue_event(self, event: dict) -> None:
        try:
            self.event_queue.put(event, timeout=0.05)
        except queue.Full:
            self.pc_queue_drops += 1


# ================= MAIN VIEWER =================

class CSIViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ESP32-S3 CSI Viewer and Binary Collector")
        self.resize(1400, 850)

        self.event_queue: queue.Queue = queue.Queue(
            maxsize=EVENT_QUEUE_MAX_SIZE
        )

        self.reader: SerialReader | None = None
        self.running = False

        self.amplitude_history = deque(maxlen=HISTORY_SIZE)
        self.rssi_history = deque(maxlen=HISTORY_SIZE)
        self.packet_rate_timestamps = deque()

        self.total_packets = 0
        self.total_stats_frames = 0
        self.latest_esp_stats: dict[str, int] = {}

        self.collecting = False
        self.waiting_offset = False
        self.collection_session_active = False
        self.collection_finalizing = False

        self.collection_start_time: float | None = None
        self.collection_end_time: float | None = None
        self.collection_finalize_after: float | None = None
        self.collection_force_finalize_after: float | None = None

        self.collection_packets: list[dict] = []
        self.collection_packet_index = 0

        self.active_collection_label = "empty"
        self.active_collection_session = "session_01"
        self.active_collection_quadrant = "quad1"
        self.active_collection_programmed = False
        self.active_programmed_item_number: int | None = None
        self.active_output_dir = DATASET_DIR

        self.programmed_active = False
        self.programmed_pause_requested = False
        self.programmed_cancel_requested = False
        self.programmed_plan: list[dict[str, int | str]] = []
        self.programmed_index = 0

        self._build_ui()
        self._configure_window_shortcuts()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_events_and_update_plots)
        self.timer.start(PLOT_UPDATE_MS)

    # ================= UI =================

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Keep the serial controls and the long diagnostics text on separate
        # rows so the interface adapts to smaller screens and Windows scaling.
        controls_widget = QtWidgets.QWidget()
        controls = QtWidgets.QGridLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(3)

        self.port_combo = QtWidgets.QComboBox()
        self.refresh_ports()

        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_ports)

        self.baud_input = QtWidgets.QLineEdit(str(DEFAULT_BAUD))
        self.baud_input.setFixedWidth(100)

        self.start_serial_button = QtWidgets.QPushButton("Start Serial")
        self.start_serial_button.clicked.connect(self.toggle_serial)

        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setMinimumWidth(0)

        self.stats_label = QtWidgets.QLabel(
            "Packets: 0 | Rate: 0.0 Hz | Queue: 0"
        )
        self.stats_label.setWordWrap(True)
        self.stats_label.setMinimumWidth(0)
        self.stats_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )

        controls.addWidget(QtWidgets.QLabel("Port"), 0, 0)
        controls.addWidget(self.port_combo, 0, 1)
        controls.addWidget(refresh_button, 0, 2)
        controls.addWidget(QtWidgets.QLabel("Baud"), 0, 3)
        controls.addWidget(self.baud_input, 0, 4)
        controls.addWidget(self.start_serial_button, 0, 5)
        controls.addWidget(self.status_label, 0, 6)
        controls.setColumnStretch(7, 1)
        controls.addWidget(self.stats_label, 1, 0, 1, 8)

        main_layout.addWidget(controls_widget)

        # The left configuration area is scrollable. This prevents controls
        # from extending below the screen on notebooks or with DPI scaling.
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.collection_panel = self._create_collection_panel()
        self.programmed_panel = self._create_programmed_panel()

        left_layout.addWidget(self.collection_panel)
        left_layout.addWidget(self.programmed_panel)
        left_layout.addStretch()

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(440)

        plots_panel = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots_panel)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(6)

        self.waveform_widget = pg.PlotWidget(
            title="CSI Waveform - Multiple Subcarriers"
        )
        self.waveform_widget.addLegend()
        self.waveform_widget.setLabel("left", "Amplitude")
        self.waveform_widget.setLabel("bottom", "Packets")

        self.waveform_curves = {}
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 150, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for index, subcarrier in enumerate(SUBCARRIERS_TO_PLOT):
            curve = self.waveform_widget.plot(
                pen=pg.mkPen(
                    color=colors[index % len(colors)],
                    width=1,
                ),
                name=f"SC {subcarrier}",
            )
            self.waveform_curves[subcarrier] = curve

        self.rssi_widget = pg.PlotWidget(title="RSSI Over Time")
        self.rssi_curve = self.rssi_widget.plot(
            pen=pg.mkPen(color=(255, 255, 255), width=1)
        )
        self.rssi_widget.setLabel("left", "RSSI dBm")
        self.rssi_widget.setLabel("bottom", "Packets")

        plots_layout.addWidget(self.waveform_widget, 2)
        plots_layout.addWidget(self.rssi_widget, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_scroll)
        splitter.addWidget(plots_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1040])

        main_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _create_collection_panel(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Dataset Collection")
        layout = QtWidgets.QVBoxLayout(group)

        self.label_combo = QtWidgets.QComboBox()
        self.label_combo.addItems(
            ["empty", "static_presence", "movement"]
        )

        self.session_input = QtWidgets.QLineEdit("session_01")

        self.quadrant_combo = QtWidgets.QComboBox()
        self.quadrant_combo.addItems(
            ["quad1", "quad2", "quad3", "quad4", "quad5"]
        )

        self.offset_input = QtWidgets.QDoubleSpinBox()
        self.offset_input.setRange(0, 300)
        self.offset_input.setValue(0)
        self.offset_input.setSuffix(" s")

        self.duration_input = QtWidgets.QDoubleSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(5)
        self.duration_input.setSuffix(" s")

        self.output_dir_input = QtWidgets.QLineEdit(str(DATASET_DIR))

        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.select_output_dir)

        output_layout = QtWidgets.QHBoxLayout()
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(self.browse_button)

        self.start_collection_button = QtWidgets.QPushButton(
            "Start Collection"
        )
        self.start_collection_button.clicked.connect(self.start_collection)

        self.stop_collection_button = QtWidgets.QPushButton("Stop and Save")
        self.stop_collection_button.clicked.connect(self.stop_collection)
        self.stop_collection_button.setEnabled(False)

        self.collection_status_label = QtWidgets.QLabel("Collection idle")

        self.collection_progress = QtWidgets.QProgressBar()
        self.collection_progress.setValue(0)

        form = QtWidgets.QFormLayout()
        form.addRow("Session", self.session_input)
        form.addRow("Quadrant", self.quadrant_combo)
        form.addRow("Label", self.label_combo)
        form.addRow("Start offset", self.offset_input)
        form.addRow("Duration", self.duration_input)
        form.addRow("Output folder", output_layout)

        layout.addLayout(form)
        layout.addWidget(self.start_collection_button)
        layout.addWidget(self.stop_collection_button)
        layout.addWidget(self.collection_progress)
        layout.addWidget(self.collection_status_label)
        layout.addStretch()

        info = QtWidgets.QLabel(
            "Saved files:\n"
            "raw_bin/session/quadrant/label/*.bin\n\n"
            "Binary serial protocol:\n"
            "CSI2 at 921600 baud\n\n"
            "Labels:\n"
            "empty | static_presence | movement"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        return group

    def _create_programmed_panel(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Programmed Collection")
        layout = QtWidgets.QVBoxLayout(group)

        self.program_cycles_input = QtWidgets.QSpinBox()
        self.program_cycles_input.setRange(1, 100)
        self.program_cycles_input.setValue(10)

        self.program_prepare_input = QtWidgets.QDoubleSpinBox()
        self.program_prepare_input.setRange(0, 300)
        self.program_prepare_input.setValue(8)
        self.program_prepare_input.setSuffix(" s")

        self.program_duration_input = QtWidgets.QDoubleSpinBox()
        self.program_duration_input.setRange(1, 3600)
        self.program_duration_input.setValue(10)
        self.program_duration_input.setSuffix(" s")

        form = QtWidgets.QFormLayout()
        form.addRow("Cycles", self.program_cycles_input)
        form.addRow("Preparation", self.program_prepare_input)
        form.addRow("Duration", self.program_duration_input)

        layout.addLayout(form)

        order_label = QtWidgets.QLabel(
            "One programmed run uses the selected quadrant.\n"
            "Change the quadrant only after the protocol finishes.\n\n"
            "Order: E-S-M → M-E-S → S-M-E\n"
            "Voice = next class\n"
            "Long beep = recording started\n"
            "Two high beeps = recording ended\n"
            "F11 = fullscreen | Esc = leave fullscreen"
        )
        order_label.setWordWrap(True)
        layout.addWidget(order_label)

        self.start_program_button = QtWidgets.QPushButton(
            "Start Programmed Collection"
        )
        self.start_program_button.clicked.connect(
            self.start_programmed_collection
        )

        self.pause_program_button = QtWidgets.QPushButton(
            "Pause after current"
        )
        self.pause_program_button.clicked.connect(
            self.toggle_programmed_pause
        )
        self.pause_program_button.setEnabled(False)

        self.cancel_program_button = QtWidgets.QPushButton(
            "Cancel after current"
        )
        self.cancel_program_button.clicked.connect(
            self.cancel_programmed_collection
        )
        self.cancel_program_button.setEnabled(False)

        self.program_progress = QtWidgets.QProgressBar()
        self.program_progress.setValue(0)

        self.program_status_label = QtWidgets.QLabel(
            "Programmed collection idle"
        )
        self.program_status_label.setWordWrap(True)

        layout.addWidget(self.start_program_button)
        layout.addWidget(self.pause_program_button)
        layout.addWidget(self.cancel_program_button)
        layout.addWidget(self.program_progress)
        layout.addWidget(self.program_status_label)

        return group

    # ================= WINDOW CONTROL =================

    def _configure_window_shortcuts(self) -> None:
        self.fullscreen_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("F11"),
            self,
        )
        self.fullscreen_shortcut.activated.connect(
            self.toggle_fullscreen
        )

        self.leave_fullscreen_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Escape"),
            self,
        )
        self.leave_fullscreen_shortcut.activated.connect(
            self.leave_fullscreen
        )

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def leave_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()

    # ================= SERIAL CONTROL =================

    def refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()

        for port in serial.tools.list_ports.comports():
            self.port_combo.addItem(port.device)

        if current:
            index = self.port_combo.findText(current)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)

    def toggle_serial(self) -> None:
        if self.running:
            self.stop_serial()
        else:
            self.start_serial()

    def start_serial(self) -> None:
        port = self.port_combo.currentText()

        if not port:
            self.status_label.setText("No COM selected")
            return

        try:
            baud = int(self.baud_input.text())
        except ValueError:
            self.status_label.setText("Invalid baud rate")
            return

        self.clear_runtime_data()

        self.reader = SerialReader(
            port=port,
            baud=baud,
            event_queue=self.event_queue,
        )
        self.reader.start()

        self.running = True
        self.start_serial_button.setText("Stop Serial")
        self.status_label.setText(f"Connected: {port}")

    def stop_serial(self) -> None:
        if self.programmed_active:
            self.programmed_cancel_requested = True

        if self.collection_session_active:
            self.stop_collection()

        if self.reader:
            self.reader.stop()
            self.reader = None

        self.running = False
        self.start_serial_button.setText("Start Serial")
        self.status_label.setText("Disconnected")

    # ================= COLLECTION =================

    def select_output_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_dir_input.text(),
        )

        if folder:
            self.output_dir_input.setText(folder)

    def start_collection(self) -> None:
        if self.programmed_active:
            self.collection_status_label.setText(
                "Finish or cancel the programmed protocol first."
            )
            return

        self._begin_collection(
            label=self.label_combo.currentText(),
            offset=float(self.offset_input.value()),
            duration=float(self.duration_input.value()),
            programmed=False,
        )

    def _begin_collection(
        self,
        label: str,
        offset: float,
        duration: float,
        programmed: bool,
        programmed_item_number: int | None = None,
    ) -> None:
        if not self.running:
            self.collection_status_label.setText(
                "Start serial before collecting."
            )
            return

        if self.collection_session_active:
            return

        self.collection_packets = []
        self.collection_packet_index = 0

        self.active_collection_label = label
        self.active_collection_session = self._sanitize_path_component(
            self.session_input.text(),
            fallback="session_01",
        )
        self.active_collection_quadrant = self._sanitize_path_component(
            self.quadrant_combo.currentText(),
            fallback="quad1",
        )
        self.active_collection_programmed = programmed
        self.active_programmed_item_number = programmed_item_number
        self.active_output_dir = Path(self.output_dir_input.text())

        now = time.time()
        self.collection_start_time = now + offset
        self.collection_end_time = self.collection_start_time + duration
        self.collection_finalize_after = None
        self.collection_force_finalize_after = None

        self.collection_session_active = True
        self.collection_finalizing = False
        self.waiting_offset = offset > 0
        self.collecting = offset == 0

        self.label_combo.setCurrentText(label)
        self.collection_progress.setValue(0)
        self.start_collection_button.setEnabled(False)
        self.stop_collection_button.setEnabled(not programmed)

        if programmed:
            self._speak_class_label(label)

        if self.waiting_offset:
            self.collection_status_label.setText(
                f"Preparing {label}: {offset:.1f} s"
            )
        else:
            self.collection_status_label.setText(
                f"Collecting: {label}"
            )
            self._play_recording_start_beep()

    def stop_collection(self) -> None:
        if not self.collection_session_active:
            return

        now = time.time()

        if self.collection_start_time is None:
            self.collection_start_time = now

        self.collection_end_time = now
        self.collecting = False
        self.waiting_offset = False
        self._start_collection_finalization(now)

    def _start_collection_finalization(self, now: float) -> None:
        if self.collection_finalizing:
            return

        self.collection_finalizing = True
        self.collection_finalize_after = (
            now + COLLECTION_FINALIZE_GRACE_SECONDS
        )
        self.collection_force_finalize_after = (
            now + COLLECTION_FINALIZE_MAX_SECONDS
        )

        if (
            self.collection_start_time is not None
            and now >= self.collection_start_time
        ):
            self._play_recording_end_beep()

        self.collection_status_label.setText(
            "Finalizing queued packets..."
        )

    def _finish_collection(self) -> None:
        was_programmed = self.active_collection_programmed

        self.collection_session_active = False
        self.collection_finalizing = False
        self.collecting = False
        self.waiting_offset = False

        saved_path = self.save_collection()

        self.start_collection_button.setEnabled(not self.programmed_active)
        self.stop_collection_button.setEnabled(False)
        self.collection_progress.setValue(100)

        if was_programmed and self.programmed_active:
            self._complete_programmed_item(saved_path)

    def save_collection(self) -> Path | None:
        if not self.collection_packets:
            self.collection_status_label.setText("No data collected.")
            return None

        raw_bin_dir = (
            self.active_output_dir
            / "raw_bin"
            / self.active_collection_session
            / self.active_collection_quadrant
            / self.active_collection_label
        )
        raw_bin_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        item_suffix = ""

        if self.active_programmed_item_number is not None:
            item_suffix = f"_p{self.active_programmed_item_number:03d}"

        file_name = (
            f"{self.active_collection_label}_"
            f"{self.active_collection_session}_"
            f"{self.active_collection_quadrant}"
            f"{item_suffix}_{timestamp}.bin"
        )
        bin_file_path = raw_bin_dir / file_name

        write_packets(bin_file_path, self.collection_packets)

        print(f"BIN saved at: {bin_file_path}")
        self.collection_status_label.setText(
            f"Saved {len(self.collection_packets)} packets:\n"
            f"{bin_file_path.name}"
        )

        return bin_file_path

    @staticmethod
    def _sanitize_path_component(value: str, fallback: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
        normalized = normalized.strip("_")
        return normalized or fallback

    # ================= PROGRAMMED COLLECTION =================

    def start_programmed_collection(self) -> None:
        if not self.running:
            self.program_status_label.setText(
                "Start serial before starting the protocol."
            )
            return

        if self.collection_session_active or self.programmed_active:
            return

        cycles = int(self.program_cycles_input.value())
        plan: list[dict[str, int | str]] = []

        for cycle_index in range(cycles):
            order = PROGRAMMED_CLASS_ORDERS[
                cycle_index % len(PROGRAMMED_CLASS_ORDERS)
            ]

            for label in order:
                plan.append(
                    {
                        "cycle": cycle_index + 1,
                        "label": label,
                    }
                )

        self.programmed_plan = plan
        self.programmed_index = 0
        self.programmed_active = True
        self.programmed_pause_requested = False
        self.programmed_cancel_requested = False

        self.program_progress.setRange(0, len(plan))
        self.program_progress.setValue(0)

        self.start_program_button.setEnabled(False)
        self.pause_program_button.setEnabled(True)
        self.cancel_program_button.setEnabled(True)
        self.pause_program_button.setText("Pause after current")

        self.start_collection_button.setEnabled(False)
        self.stop_collection_button.setEnabled(False)

        self._set_program_configuration_enabled(False)
        self._start_next_programmed_item()

    def _start_next_programmed_item(self) -> None:
        if not self.programmed_active:
            return

        if self.programmed_cancel_requested:
            self._finish_programmed_collection(cancelled=True)
            return

        if self.programmed_pause_requested:
            self.program_status_label.setText(
                "Protocol paused. Press Resume to continue."
            )
            return

        if self.programmed_index >= len(self.programmed_plan):
            self._finish_programmed_collection(cancelled=False)
            return

        item = self.programmed_plan[self.programmed_index]
        label = str(item["label"])
        cycle = int(item["cycle"])
        item_number = self.programmed_index + 1
        total = len(self.programmed_plan)

        session = self._sanitize_path_component(
            self.session_input.text(),
            fallback="session_01",
        )
        quadrant = self._sanitize_path_component(
            self.quadrant_combo.currentText(),
            fallback="quad1",
        )

        self.program_status_label.setText(
            f"{session} | {quadrant} | "
            f"Item {item_number}/{total} | Cycle {cycle} | "
            f"Prepare for: {label}"
        )

        self._begin_collection(
            label=label,
            offset=float(self.program_prepare_input.value()),
            duration=float(self.program_duration_input.value()),
            programmed=True,
            programmed_item_number=item_number,
        )

    def _complete_programmed_item(
        self,
        saved_path: Path | None,
    ) -> None:
        completed_item = self.programmed_index + 1
        self.programmed_index += 1
        self.program_progress.setValue(self.programmed_index)

        if saved_path is None:
            self.program_status_label.setText(
                f"Item {completed_item} produced no packets. "
                "Protocol stopped for verification."
            )
            self.programmed_pause_requested = True
            self.pause_program_button.setText("Resume")
            return

        if self.programmed_cancel_requested:
            self._finish_programmed_collection(cancelled=True)
            return

        if self.programmed_pause_requested:
            self.program_status_label.setText(
                f"Paused after item {completed_item}. "
                "Press Resume to continue."
            )
            return

        QtCore.QTimer.singleShot(
            PROGRAMMED_NEXT_DELAY_MS,
            self._start_next_programmed_item,
        )

    def toggle_programmed_pause(self) -> None:
        if not self.programmed_active:
            return

        self.programmed_pause_requested = (
            not self.programmed_pause_requested
        )

        if self.programmed_pause_requested:
            self.pause_program_button.setText("Resume")

            if self.collection_session_active:
                self.program_status_label.setText(
                    "Pause requested. The current collection will finish."
                )
            else:
                self.program_status_label.setText(
                    "Protocol paused. Press Resume to continue."
                )
        else:
            self.pause_program_button.setText("Pause after current")
            self.program_status_label.setText("Resuming protocol...")

            if not self.collection_session_active:
                QtCore.QTimer.singleShot(
                    200,
                    self._start_next_programmed_item,
                )

    def cancel_programmed_collection(self) -> None:
        if not self.programmed_active:
            return

        self.programmed_cancel_requested = True
        self.cancel_program_button.setEnabled(False)

        if self.collection_session_active:
            self.program_status_label.setText(
                "Cancellation requested. "
                "The current collection will finish and be saved."
            )
        else:
            self._finish_programmed_collection(cancelled=True)

    def _finish_programmed_collection(self, cancelled: bool) -> None:
        completed = self.programmed_index
        total = len(self.programmed_plan)

        self.programmed_active = False
        self.programmed_pause_requested = False
        self.programmed_cancel_requested = False
        self.programmed_plan = []

        self.start_program_button.setEnabled(True)
        self.pause_program_button.setEnabled(False)
        self.cancel_program_button.setEnabled(False)
        self.pause_program_button.setText("Pause after current")

        self.start_collection_button.setEnabled(self.running)
        self.stop_collection_button.setEnabled(False)
        self._set_program_configuration_enabled(True)

        if cancelled:
            self.program_status_label.setText(
                f"Protocol cancelled after {completed}/{total} collections."
            )
        else:
            self.program_status_label.setText(
                f"Protocol completed: {completed}/{total} collections."
            )
            self._play_program_complete_beep()

    def _set_program_configuration_enabled(self, enabled: bool) -> None:
        self.session_input.setEnabled(enabled)
        self.quadrant_combo.setEnabled(enabled)
        self.output_dir_input.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.program_cycles_input.setEnabled(enabled)
        self.program_prepare_input.setEnabled(enabled)
        self.program_duration_input.setEnabled(enabled)

    # ================= AUDIO CUES =================

    def _speak_class_label(self, label: str) -> None:
        spoken_text = CLASS_SPOKEN_NAMES.get(label, label)

        def worker() -> None:
            if sys.platform != "win32":
                self._play_class_beep_fallback(label)
                return

            escaped_text = spoken_text.replace("'", "''")
            powershell_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object "
                "System.Speech.Synthesis.SpeechSynthesizer; "
                "$speaker.Rate = -1; "
                "$speaker.Volume = 100; "
                f"$speaker.Speak('{escaped_text}'); "
                "$speaker.Dispose();"
            )

            try:
                completed = subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        powershell_script,
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(
                        subprocess,
                        "CREATE_NO_WINDOW",
                        0,
                    ),
                )

                if completed.returncode != 0:
                    self._play_class_beep_fallback(label)

            except (OSError, subprocess.SubprocessError):
                self._play_class_beep_fallback(label)

        threading.Thread(target=worker, daemon=True).start()

    def _play_class_beep_fallback(self, label: str) -> None:
        count = CLASS_BEEP_COUNTS.get(label, 1)
        pattern = [(750, 180, 160)] * count
        self._play_beep_pattern(pattern)

    def _play_recording_start_beep(self) -> None:
        self._play_beep_pattern([(1200, 650, 0)])

    def _play_recording_end_beep(self) -> None:
        self._play_beep_pattern(
            [
                (1500, 170, 140),
                (1500, 170, 0),
            ]
        )

    def _play_program_complete_beep(self) -> None:
        self._play_beep_pattern(
            [
                (900, 180, 100),
                (1100, 180, 100),
                (1300, 300, 0),
            ]
        )

    @staticmethod
    def _play_beep_pattern(
        pattern: list[tuple[int, int, int]],
    ) -> None:
        if winsound is None:
            QtWidgets.QApplication.beep()
            return

        def worker() -> None:
            for frequency, duration_ms, pause_ms in pattern:
                try:
                    winsound.Beep(frequency, duration_ms)
                except RuntimeError:
                    winsound.MessageBeep()

                if pause_ms > 0:
                    time.sleep(pause_ms / 1000.0)

        threading.Thread(target=worker, daemon=True).start()

    # ================= PROCESSING =================

    def clear_runtime_data(self) -> None:
        self.amplitude_history.clear()
        self.rssi_history.clear()
        self.packet_rate_timestamps.clear()

        self.total_packets = 0
        self.total_stats_frames = 0
        self.latest_esp_stats = {}

        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

        for curve in self.waveform_curves.values():
            curve.setData([])

        self.rssi_curve.setData([])
        self.update_stats_label()

    def process_events_and_update_plots(self) -> None:
        now = time.time()
        self.update_collection_timing(now)

        events_processed = 0
        new_packets = 0

        while (
            not self.event_queue.empty()
            and events_processed < MAX_EVENTS_PER_UPDATE
        ):
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            events_processed += 1
            event_type = event.get("type")

            if event_type == "sample":
                if self._process_sample_event(event):
                    new_packets += 1

            elif event_type == "stats":
                self.total_stats_frames += 1
                self.latest_esp_stats = dict(event.get("stats", {}))

        if new_packets > 0:
            self.update_plots()

        self._check_collection_finalization(time.time())
        self._check_reader_status()
        self.update_stats_label()

    def _process_sample_event(self, event: dict) -> bool:
        imag = event.get("imag")
        real = event.get("real")

        if imag is None or real is None:
            return False

        if len(imag) == 0 or len(real) == 0:
            return False

        amplitude = [
            (float(i) ** 2 + float(r) ** 2) ** 0.5
            for i, r in zip(imag, real)
        ]

        metadata = event.get("metadata", {})
        rssi = int(metadata.get("rssi", 0) or 0)

        self.amplitude_history.append(amplitude)
        self.rssi_history.append(rssi)

        self.total_packets += 1

        pc_timestamp = float(event.get("pc_timestamp", time.time()))
        self.packet_rate_timestamps.append(pc_timestamp)
        self._trim_packet_rate_history(pc_timestamp)

        if self._event_belongs_to_collection(event):
            self.append_collection_packet(event)

        return True

    def _event_belongs_to_collection(self, event: dict) -> bool:
        if not self.collection_session_active:
            return False

        if self.collection_start_time is None or self.collection_end_time is None:
            return False

        capture_timestamp = float(
            event.get("capture_timestamp", event.get("pc_timestamp", 0.0))
        )

        return (
            self.collection_start_time
            <= capture_timestamp
            <= self.collection_end_time
        )

    def update_collection_timing(self, now: float) -> None:
        if not self.collection_session_active:
            return

        if self.waiting_offset and self.collection_start_time is not None:
            remaining = self.collection_start_time - now

            if remaining <= 0:
                self.waiting_offset = False
                self.collecting = True
                self.collection_status_label.setText(
                    f"Collecting: {self.active_collection_label}"
                )
                self._play_recording_start_beep()
            else:
                self.collection_status_label.setText(
                    f"Waiting offset: {remaining:.1f} s"
                )

        if (
            self.collecting
            and self.collection_start_time is not None
            and self.collection_end_time is not None
        ):
            duration = self.collection_end_time - self.collection_start_time
            elapsed = now - self.collection_start_time

            progress = int(
                max(0, min(100, (elapsed / duration) * 100))
            )
            self.collection_progress.setValue(progress)

            if now >= self.collection_end_time:
                self.collecting = False
                self._start_collection_finalization(now)

    def _check_collection_finalization(self, now: float) -> None:
        if not self.collection_finalizing:
            return

        grace_elapsed = (
            self.collection_finalize_after is not None
            and now >= self.collection_finalize_after
        )
        force_elapsed = (
            self.collection_force_finalize_after is not None
            and now >= self.collection_force_finalize_after
        )

        if (grace_elapsed and self.event_queue.empty()) or force_elapsed:
            self._finish_collection()

    def append_collection_packet(self, event: dict) -> None:
        metadata = event.get("metadata", {})
        imag = event.get("imag")
        real = event.get("real")

        if imag is None or real is None:
            return

        self.collection_packet_index += 1

        packet = {
            "label": self.active_collection_label,
            "pc_timestamp": float(event.get("pc_timestamp", time.time())),
            "capture_timestamp": float(
                event.get("capture_timestamp", event.get("pc_timestamp", 0.0))
            ),
            "esp_timestamp_us": int(metadata.get("timestamp_us", 0) or 0),
            "sequence": int(metadata.get("sequence", 0) or 0),
            "packet_index": self.collection_packet_index,
            "rssi": int(metadata.get("rssi", 0) or 0),
            "rate": int(metadata.get("rate", 0) or 0),
            "channel": int(metadata.get("channel", 0) or 0),
            "csi_len": int(metadata.get("csi_len", 0) or 0),
            "flags": int(metadata.get("flags", 0) or 0),
            "imag": [int(value) for value in imag],
            "real": [int(value) for value in real],
        }

        self.collection_packets.append(packet)

    # ================= PLOTS / STATUS =================

    def update_plots(self) -> None:
        if not self.amplitude_history:
            return

        for subcarrier, curve in self.waveform_curves.items():
            values = []

            for amplitude in self.amplitude_history:
                if subcarrier < len(amplitude):
                    values.append(float(amplitude[subcarrier]))

            curve.setData(values)

        self.rssi_curve.setData(list(self.rssi_history))

    def _trim_packet_rate_history(self, now: float) -> None:
        minimum_timestamp = now - 1.0

        while (
            self.packet_rate_timestamps
            and self.packet_rate_timestamps[0] < minimum_timestamp
        ):
            self.packet_rate_timestamps.popleft()

    def _check_reader_status(self) -> None:
        if not self.running or self.reader is None:
            return

        if not self.reader.running and self.reader.last_error:
            self.running = False
            self.start_serial_button.setText("Start Serial")
            self.status_label.setText(
                f"Serial error: {self.reader.last_error}"
            )

    def update_stats_label(self) -> None:
        rate_hz = float(len(self.packet_rate_timestamps))

        diagnostics = self.reader.diagnostics() if self.reader else {}
        sequence_gaps = int(diagnostics.get("sequence_gaps", 0) or 0)
        crc_errors = int(diagnostics.get("crc_errors", 0) or 0)
        pc_drops = int(diagnostics.get("pc_queue_drops", 0) or 0)

        esp_drops = int(self.latest_esp_stats.get("queue_drops", 0) or 0)
        esp_pending = int(self.latest_esp_stats.get("queue_pending", 0) or 0)

        self.stats_label.setText(
            f"Packets: {self.total_packets} | "
            f"Rate: {rate_hz:.1f} Hz | "
            f"PC queue: {self.event_queue.qsize()} | "
            f"Seq gaps: {sequence_gaps} | "
            f"CRC: {crc_errors} | "
            f"PC drops: {pc_drops} | "
            f"ESP drops: {esp_drops} | "
            f"ESP pending: {esp_pending}"
        )

    def closeEvent(self, event) -> None:
        self.stop_serial()
        event.accept()


# ================= MAIN =================

def main() -> None:
    RAW_BIN_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_CSV_DIR.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    viewer = CSIViewer()
    viewer.showFullScreen()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()