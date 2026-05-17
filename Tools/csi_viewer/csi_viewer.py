import sys
from collections import deque

import serial
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from csi_parser import parse_csi_line


DEFAULT_BAUD = 115200
HISTORY_SIZE = 300
SUBCARRIERS_TO_PLOT = [0, 5, 10, 15, 20, 25]


class CSIViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ESP32-S3 CSI Waveform Viewer")
        self.resize(1300, 750)

        self.serial_conn = None
        self.running = False

        self.amplitude_history = deque(maxlen=HISTORY_SIZE)
        self.rssi_history = deque(maxlen=HISTORY_SIZE)

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

        self.current_amp_plot = self.graphs.addPlot(title="Current CSI Amplitude per Subcarrier")
        self.current_amp_curve = self.current_amp_plot.plot(pen=pg.mkPen("y", width=2))
        self.current_amp_plot.setLabel("left", "Amplitude")
        self.current_amp_plot.setLabel("bottom", "Subcarrier")

        self.graphs.nextRow()

        self.waveform_plot = self.graphs.addPlot(title="CSI Amplitude Waveform - Multiple Subcarriers")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setLabel("bottom", "Samples")
        self.waveform_plot.addLegend()

        self.waveform_curves = {}

        colors = ["r", "g", "b", "c", "m", "y"]

        for index, subcarrier in enumerate(SUBCARRIERS_TO_PLOT):
            curve = self.waveform_plot.plot(
                pen=pg.mkPen(colors[index % len(colors)], width=2),
                name=f"Subcarrier {subcarrier}"
            )
            self.waveform_curves[subcarrier] = curve

        self.graphs.nextRow()

        self.rssi_plot = self.graphs.addPlot(title="RSSI Over Time")
        self.rssi_curve = self.rssi_plot.plot(pen=pg.mkPen("w", width=2))
        self.rssi_plot.setLabel("left", "RSSI dBm")
        self.rssi_plot.setLabel("bottom", "Samples")

        self.setCentralWidget(central)

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

                self.amplitude_history.append(amplitude)
                self.rssi_history.append(metadata.get("rssi", 0))

                self.current_amp_curve.setData(amplitude)

                self.update_waveform()
                self.rssi_curve.setData(list(self.rssi_history))

        except Exception as exc:
            self.status_label.setText(f"Read error: {exc}")

    def update_waveform(self):
        if len(self.amplitude_history) < 2:
            return

        for subcarrier, curve in self.waveform_curves.items():
            values = []

            for amplitude in self.amplitude_history:
                if subcarrier < len(amplitude):
                    values.append(float(amplitude[subcarrier]))

            curve.setData(values)


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = CSIViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()