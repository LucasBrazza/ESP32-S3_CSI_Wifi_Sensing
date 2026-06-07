import sys
import time
import queue
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

import serial
import serial.tools.list_ports
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore


# ================= PATHS =================

TOOLS_DIR = Path(__file__).resolve().parents[2]

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from acquisition.gui.csi_parser import parse_csi_line
from csi.csi_binary_io import write_packets


# ================= CONFIG =================

DEFAULT_BAUD = 115200

HISTORY_SIZE = 150
PLOT_UPDATE_MS = 100
MAX_LINES_PER_UPDATE = 10
QUEUE_MAX_SIZE = 500

SUBCARRIERS_TO_PLOT = [0, 5, 10, 15, 20, 25]

DATASET_DIR = TOOLS_DIR / "datasets"

RAW_BIN_DIR = DATASET_DIR / "raw_bin"
DEBUG_CSV_DIR = DATASET_DIR / "debug_csv"


# ================= SERIAL READER =================

class SerialReader:
    def __init__(self, port, baud, line_queue):
        self.port = port
        self.baud = baud
        self.line_queue = line_queue
        self.running = False
        self.thread = None
        self.serial_conn = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False

        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def _read_loop(self):
        try:
            self.serial_conn = serial.Serial(
                self.port,
                self.baud,
                timeout=0.05,
            )

            while self.running:
                try:
                    line = self.serial_conn.readline().decode(
                        errors="ignore"
                    ).strip()

                    if not line:
                        continue

                    if self.line_queue.full():
                        try:
                            self.line_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.line_queue.put_nowait(line)

                except Exception:
                    continue

        except Exception as exc:
            print(f"Serial error: {exc}")

        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()


# ================= MAIN VIEWER =================

class CSIViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ESP32-S3 CSI Viewer and Binary Collector")
        self.resize(1400, 850)

        self.line_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

        self.reader = None
        self.running = False

        self.amplitude_history = deque(maxlen=HISTORY_SIZE)
        self.rssi_history = deque(maxlen=HISTORY_SIZE)

        self.total_lines = 0
        self.total_packets = 0

        self.collecting = False
        self.waiting_offset = False

        self.collection_start_time = None
        self.collection_end_time = None

        self.collection_packets = []
        self.collection_packet_index = 0

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_lines_and_update_plots)
        self.timer.start(PLOT_UPDATE_MS)

    # ================= UI =================

    def _build_ui(self):
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()

        self.port_combo = QtWidgets.QComboBox()
        self.refresh_ports()

        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_ports)

        self.baud_input = QtWidgets.QLineEdit(str(DEFAULT_BAUD))
        self.baud_input.setFixedWidth(100)

        self.start_serial_button = QtWidgets.QPushButton("Start Serial")
        self.start_serial_button.clicked.connect(self.toggle_serial)

        self.status_label = QtWidgets.QLabel("Disconnected")

        self.stats_label = QtWidgets.QLabel("Lines: 0 | Packets: 0 | Queue: 0")

        controls.addWidget(QtWidgets.QLabel("Port"))
        controls.addWidget(self.port_combo)
        controls.addWidget(refresh_button)
        controls.addWidget(QtWidgets.QLabel("Baud"))
        controls.addWidget(self.baud_input)
        controls.addWidget(self.start_serial_button)
        controls.addWidget(self.status_label)
        controls.addStretch()
        controls.addWidget(self.stats_label)

        main_layout.addLayout(controls)

        grid = QtWidgets.QGridLayout()
        main_layout.addLayout(grid)

        self.collection_panel = self._create_collection_panel()
        grid.addWidget(self.collection_panel, 0, 0, 2, 1)

        # ================= WAVEFORM =================

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

        for i, subcarrier in enumerate(SUBCARRIERS_TO_PLOT):
            curve = self.waveform_widget.plot(
                pen=pg.mkPen(
                    color=colors[i % len(colors)],
                    width=1,
                ),
                name=f"SC {subcarrier}",
            )

            self.waveform_curves[subcarrier] = curve

        grid.addWidget(self.waveform_widget, 0, 1)

        # ================= RSSI =================

        self.rssi_widget = pg.PlotWidget(title="RSSI Over Time")

        self.rssi_curve = self.rssi_widget.plot(
            pen=pg.mkPen(
                color=(255, 255, 255),
                width=1,
            )
        )

        self.rssi_widget.setLabel("left", "RSSI dBm")
        self.rssi_widget.setLabel("bottom", "Packets")

        grid.addWidget(self.rssi_widget, 1, 1)

        self.setCentralWidget(central)

    def _create_collection_panel(self):
        group = QtWidgets.QGroupBox("Dataset Collection")
        layout = QtWidgets.QVBoxLayout(group)

        self.label_combo = QtWidgets.QComboBox()

        self.label_combo.addItems(
            [
                "empty",
                "static_presence",
                "movement",
            ]
        )

        self.offset_input = QtWidgets.QDoubleSpinBox()
        self.offset_input.setRange(0, 300)
        self.offset_input.setValue(0)
        self.offset_input.setSuffix(" s")

        self.duration_input = QtWidgets.QDoubleSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(60)
        self.duration_input.setSuffix(" s")

        self.output_dir_input = QtWidgets.QLineEdit(str(DATASET_DIR))

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.select_output_dir)

        output_layout = QtWidgets.QHBoxLayout()
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(browse_button)

        self.start_collection_button = QtWidgets.QPushButton("Start Collection")
        self.start_collection_button.clicked.connect(self.start_collection)

        self.stop_collection_button = QtWidgets.QPushButton("Stop and Save")
        self.stop_collection_button.clicked.connect(self.stop_collection)
        self.stop_collection_button.setEnabled(False)

        self.collection_status_label = QtWidgets.QLabel("Collection idle")

        self.collection_progress = QtWidgets.QProgressBar()
        self.collection_progress.setValue(0)

        form = QtWidgets.QFormLayout()
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
            "datasets/raw_bin/label_timestamp.bin\n\n"
            "Debug CSV conversion:\n"
            "csi/bin_to_csv.py\n\n"
            "Labels:\n"
            "empty | static_presence | movement"
        )

        info.setWordWrap(True)
        layout.addWidget(info)

        return group

    # ================= SERIAL CONTROL =================

    def refresh_ports(self):
        self.port_combo.clear()

        for port in serial.tools.list_ports.comports():
            self.port_combo.addItem(port.device)

    def toggle_serial(self):
        if self.running:
            self.stop_serial()
        else:
            self.start_serial()

    def start_serial(self):
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
            line_queue=self.line_queue,
        )

        self.reader.start()

        self.running = True

        self.start_serial_button.setText("Stop Serial")
        self.status_label.setText(f"Connected: {port}")

    def stop_serial(self):
        if self.reader:
            self.reader.stop()
            self.reader = None

        self.running = False

        self.start_serial_button.setText("Start Serial")
        self.status_label.setText("Disconnected")

    # ================= COLLECTION =================

    def select_output_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_dir_input.text(),
        )

        if folder:
            self.output_dir_input.setText(folder)

    def start_collection(self):
        if not self.running:
            self.collection_status_label.setText(
                "Start serial before collecting."
            )
            return

        self.collection_packets = []
        self.collection_packet_index = 0

        offset = float(self.offset_input.value())
        duration = float(self.duration_input.value())

        now = time.time()

        self.collection_start_time = now + offset
        self.collection_end_time = self.collection_start_time + duration

        self.waiting_offset = offset > 0
        self.collecting = offset == 0

        self.collection_progress.setValue(0)

        self.start_collection_button.setEnabled(False)
        self.stop_collection_button.setEnabled(True)

        if self.waiting_offset:
            self.collection_status_label.setText(
                f"Waiting offset: {offset:.1f} s"
            )
        else:
            self.collection_status_label.setText(
                f"Collecting: {self.label_combo.currentText()}"
            )

    def stop_collection(self):
        if not self.collecting and not self.waiting_offset:
            return

        self.collecting = False
        self.waiting_offset = False

        self.save_collection()

        self.start_collection_button.setEnabled(True)
        self.stop_collection_button.setEnabled(False)

    def save_collection(self):
        if not self.collection_packets:
            self.collection_status_label.setText("No data collected.")
            return

        base_output_dir = Path(self.output_dir_input.text())

        raw_bin_dir = base_output_dir / "raw_bin"

        raw_bin_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        label = self.label_combo.currentText()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        bin_file_path = raw_bin_dir / f"{label}_{timestamp}.bin"

        write_packets(
            bin_file_path,
            self.collection_packets,
        )

        print(f"BIN saved at: {bin_file_path}")

        self.collection_status_label.setText(
            f"Saved:\n{bin_file_path.name}"
        )

    # ================= PROCESSING =================

    def clear_runtime_data(self):
        self.amplitude_history.clear()
        self.rssi_history.clear()

        self.total_lines = 0
        self.total_packets = 0

        while not self.line_queue.empty():
            try:
                self.line_queue.get_nowait()

            except queue.Empty:
                break

        for curve in self.waveform_curves.values():
            curve.setData([])

        self.rssi_curve.setData([])

        self.update_stats_label()

    def process_lines_and_update_plots(self):
        lines_processed = 0
        new_packets = 0

        now = time.time()

        self.update_collection_timing(now)

        while (
            not self.line_queue.empty()
            and lines_processed < MAX_LINES_PER_UPDATE
        ):
            try:
                line = self.line_queue.get_nowait()

            except queue.Empty:
                break

            self.total_lines += 1
            lines_processed += 1

            parsed = parse_csi_line(line)

            if parsed is None:
                continue

            imag = parsed.get("imag")
            real = parsed.get("real")

            if imag is None or real is None:
                continue

            if len(imag) == 0 or len(real) == 0:
                continue

            amplitude = [
                (float(i) ** 2 + float(r) ** 2) ** 0.5
                for i, r in zip(imag, real)
            ]

            metadata = parsed.get("metadata", {})
            rssi = metadata.get("rssi", 0)

            self.amplitude_history.append(amplitude)
            self.rssi_history.append(rssi)

            self.total_packets += 1
            new_packets += 1

            if self.collecting:
                self.append_collection_packet(parsed)

        if new_packets > 0:
            self.update_plots()

        self.update_stats_label()

    def update_collection_timing(self, now):
        if self.waiting_offset:
            remaining = self.collection_start_time - now

            if remaining <= 0:
                self.waiting_offset = False
                self.collecting = True

                self.collection_status_label.setText(
                    f"Collecting: {self.label_combo.currentText()}"
                )

            else:
                self.collection_status_label.setText(
                    f"Waiting offset: {remaining:.1f} s"
                )

        if self.collecting:
            duration = self.collection_end_time - self.collection_start_time
            elapsed = now - self.collection_start_time

            progress = int(
                max(
                    0,
                    min(
                        100,
                        (elapsed / duration) * 100,
                    ),
                )
            )

            self.collection_progress.setValue(progress)

            if now >= self.collection_end_time:
                self.collecting = False

                self.save_collection()

                self.start_collection_button.setEnabled(True)
                self.stop_collection_button.setEnabled(False)

    def append_collection_packet(self, parsed):
        label = self.label_combo.currentText()
        pc_timestamp = time.time()

        metadata = parsed.get("metadata", {})

        imag = parsed.get("imag")
        real = parsed.get("real")

        if imag is None or real is None:
            return

        self.collection_packet_index += 1

        packet = {
            "label": label,
            "pc_timestamp": pc_timestamp,
            "packet_index": self.collection_packet_index,
            "rssi": int(metadata.get("rssi", 0) or 0),
            "rate": int(metadata.get("rate", 0) or 0),
            "channel": int(metadata.get("channel", 0) or 0),
            "csi_len": int(metadata.get("csi_len", 0) or 0),
            "imag": [int(value) for value in imag],
            "real": [int(value) for value in real],
        }

        self.collection_packets.append(packet)

    # ================= PLOTS =================

    def update_plots(self):
        if not self.amplitude_history:
            return

        for subcarrier, curve in self.waveform_curves.items():
            values = []

            for amplitude in self.amplitude_history:
                if subcarrier < len(amplitude):
                    values.append(float(amplitude[subcarrier]))

            curve.setData(values)

        self.rssi_curve.setData(list(self.rssi_history))

    def update_stats_label(self):
        self.stats_label.setText(
            f"Lines: {self.total_lines} | "
            f"Packets: {self.total_packets} | "
            f"Queue: {self.line_queue.qsize()}"
        )

    def closeEvent(self, event):
        self.stop_serial()
        event.accept()


# ================= MAIN =================

def main():
    RAW_BIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEBUG_CSV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    app = QtWidgets.QApplication(sys.argv)

    viewer = CSIViewer()
    viewer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()