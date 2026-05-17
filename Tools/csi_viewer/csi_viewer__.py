import sys
import time
from collections import deque

import serial
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from csi_parser import parse_csi_line


DEFAULT_BAUD = 115200
HISTORY_SIZE = 200


class CSIViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ESP32-S3 CSI Waveform Viewer")
        self.resize(1300, 800)

        self.serial_conn = None
        self.running = False

        self.amplitude_history = deque(maxlen=HISTORY_SIZE)
        self.rssi_history = deque(maxlen=HISTORY_SIZE)
        self.packet_count_history = deque(maxlen=HISTORY_SIZE)

        self.packet_counter = 0
        self.last_packet_time = time.time()

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(20)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()

        self.port_combo = QtWidgets.QComboBox()
        self.refresh_ports()

        self.refresh_button = QtWidgets.QPushButton("Refresh ports")
        self.refresh_button.clicked.connect(self.refresh_ports)

        self.baud_input = QtWidgets.QLineEdit(str(DEFAULT_BAUD))
        self.baud_input.setFixedWidth(100)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_serial)

        self.status_label = QtWidgets.QLabel("Disconnected")

        controls.addWidget(QtWidgets.QLabel("Port:"))
        controls.addWidget(self.port_combo)
        controls.addWidget(self.refresh_button)
        controls.addWidget(QtWidgets.QLabel("Baud:"))
        controls.addWidget(self.baud_input)
        controls.addWidget(self.start_button)
        controls.addWidget(self.status_label)
        controls.addStretch()

        layout.addLayout(controls)

        self.graphs = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphs)

        self.amp_plot = self.graphs.addPlot(title="Current CSI Amplitude per Subcarrier")
        self.amp_curve = self.amp_plot.plot()
        self.amp_plot.setLabel("left", "Amplitude")
        self.amp_plot.setLabel("bottom", "Subcarrier")

        self.graphs.nextRow()

        self.waveform_plot = self.graphs.addPlot(title="Amplitude Waveform Over Time")
        self.waveform_image = pg.ImageItem()
        self.waveform_plot.addItem(self.waveform_image)
        self.waveform_plot.setLabel("left", "Packet history")
        self.waveform_plot.setLabel("bottom", "Subcarrier")

        self.graphs.nextRow()

        self.rssi_plot = self.graphs.addPlot(title="RSSI Over Time")
        self.rssi_curve = self.rssi_plot.plot()
        self.rssi_plot.setLabel("left", "RSSI dBm")
        self.rssi_plot.setLabel("bottom", "Samples")

        self.graphs.nextRow()

        self.mean_amp_plot = self.graphs.addPlot(title="Mean Amplitude Over Time")
        self.mean_amp_curve = self.mean_amp_plot.plot()
        self.mean_amp_plot.setLabel("left", "Mean amplitude")
        self.mean_amp_plot.setLabel("bottom", "Samples")

        self.setCentralWidget(central)

    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()

        for port in ports:
            self.port_combo.addItem(port.device)

    def toggle_serial(self):
        if self.running:
            self.stop_serial()
        else:
            self.start_serial()

    def start_serial(self):
        port = self.port_combo.currentText()
        baud = int(self.baud_input.text())

        if not port:
            self.status_label.setText("No COM port selected")
            return

        try:
            self.serial_conn = serial.Serial(port, baud, timeout=0.01)
            self.running = True
            self.start_button.setText("Stop")
            self.status_label.setText(f"Connected to {port}")
        except Exception as exc:
            self.status_label.setText(f"Error: {exc}")

    def stop_serial(self):
        self.running = False
        self.start_button.setText("Start")

        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

        self.status_label.setText("Disconnected")

    def update_data(self):
        if not self.running or not self.serial_conn:
            return

        try:
            while self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode(errors="ignore").strip()
                parsed = parse_csi_line(line)

                if parsed is None:
                    continue

                amplitude = parsed["amplitude"]
                metadata = parsed["metadata"]

                if len(amplitude) == 0:
                    continue

                self.packet_counter += 1

                self.amplitude_history.append(amplitude)
                self.rssi_history.append(metadata.get("rssi", 0))

                self.amp_curve.setData(amplitude)

                self.rssi_curve.setData(list(self.rssi_history))

                mean_values = [float(np.mean(a)) for a in self.amplitude_history]
                self.mean_amp_curve.setData(mean_values)

                self.update_waveform()

        except Exception as exc:
            self.status_label.setText(f"Read error: {exc}")

    def update_waveform(self):
        if len(self.amplitude_history) < 2:
            return

        min_len = min(len(a) for a in self.amplitude_history)
        matrix = np.array([a[:min_len] for a in self.amplitude_history], dtype=np.float32)

        # Normalize for better visual contrast
        matrix = matrix - np.min(matrix)
        max_value = np.max(matrix)

        if max_value > 0:
            matrix = matrix / max_value

        self.waveform_image.setImage(matrix, autoLevels=True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = CSIViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()