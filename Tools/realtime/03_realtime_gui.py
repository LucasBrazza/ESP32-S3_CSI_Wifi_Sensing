from __future__ import annotations

"""GUI de inferência CSI em tempo real com registro e TTS no Windows."""

import argparse
import csv
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import serial.tools.list_ports
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = TOOLS_DIR.parent

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from acquisition.gui.csi_viewer import SerialReader  # noqa: E402
from csi.csi_binary_io import write_packets  # noqa: E402
from realtime.realtime_inference_engine import (  # noqa: E402
    IncrementalRealtimeInferenceEngine,
    RealtimeInferenceResult,
    sample_event_to_packet,
)

DEFAULT_BAUD = 921600
DEFAULT_START_DELAY_SECONDS = 10
DEFAULT_OUTPUT_ROOT = TOOLS_DIR / "datasets" / "realtime_runs"
EVENT_QUEUE_MAX_SIZE = 5000
MAX_EVENTS_PER_UPDATE = 500
GUI_UPDATE_MS = 50
HISTORY_ROWS = 80

STATE_NAMES = {
    "empty": "AMBIENTE VAZIO",
    "static_presence": "PRESENÇA ESTÁTICA",
    "movement": "MOVIMENTO",
}

STATE_SPEECH = {
    "empty": "Ambiente vazio.",
    "static_presence": "Presença estática detectada.",
    "movement": "Movimento detectado.",
}

STATE_STYLES = {
    "empty": "background:#DDEEFF;color:#153A5B;border:2px solid #4C83B6;border-radius:12px;padding:18px;",
    "static_presence": "background:#E4F3E7;color:#205C2E;border:2px solid #5A9B69;border-radius:12px;padding:18px;",
    "movement": "background:#FFF0D8;color:#704312;border:2px solid #D59442;border-radius:12px;padding:18px;",
}

GROUND_TRUTH_OPTIONS = (
    ("empty", "Ambiente vazio"),
    ("transition", "Transição"),
    ("static_presence", "Presença estática"),
    ("movement", "Movimento"),
    ("", "Sem rótulo"),
)


class SpeechWorker:
    """Executa TTS sem bloquear a serial ou a interface."""

    def __init__(self) -> None:
        self.messages: queue.Queue[tuple[str, str] | None] = queue.Queue(maxsize=4)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def speak(self, state: str, text: str) -> None:
        if not self.running:
            return
        try:
            while True:
                self.messages.get_nowait()
        except queue.Empty:
            pass
        try:
            self.messages.put_nowait((state, text))
        except queue.Full:
            pass

    def stop(self) -> None:
        self.running = False
        try:
            self.messages.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self.running:
            item = self.messages.get()
            if item is None:
                return
            state, text = item
            if not self._speak_windows(text):
                self._beep(state)

    @staticmethod
    def _speak_windows(text: str) -> bool:
        executable = (
            shutil.which("powershell.exe")
            or shutil.which("powershell")
            or shutil.which("pwsh.exe")
            or shutil.which("pwsh")
        )
        if not executable:
            return False

        escaped = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Volume = 100; $s.Rate = 0; "
            f"$s.Speak('{escaped}');"
        )
        flags = (
            subprocess.CREATE_NO_WINDOW
            if sys.platform.startswith("win") and hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        try:
            result = subprocess.run(
                [executable, "-NoProfile", "-NonInteractive", "-Command", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
                creationflags=flags,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    @staticmethod
    def _beep(state: str) -> None:
        if winsound is None:
            return
        for _ in range({"empty": 1, "static_presence": 2, "movement": 3}.get(state, 1)):
            try:
                winsound.Beep(900, 130)
                time.sleep(0.08)
            except RuntimeError:
                return


class RunRecorder:
    def __init__(self, output_root: Path, save_raw: bool) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = output_root / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.predictions_path = self.run_dir / "realtime_predictions.csv"
        self.raw_path = self.run_dir / "raw_stream.bin"
        self.metadata_path = self.run_dir / "metadata.json"
        self.save_raw = save_raw
        self.raw_packets: list[dict[str, Any]] = []
        self.started_at = time.time()
        self.prediction_count = 0
        self._file = None
        self._writer: csv.DictWriter | None = None

    def append_packet(self, packet: Mapping[str, Any], truth: str) -> None:
        if self.save_raw:
            stored = dict(packet)
            stored["label"] = truth
            self.raw_packets.append(stored)

    def append_result(self, result: RealtimeInferenceResult, truth: str) -> None:
        row = {"logged_at_pc": time.time(), "ground_truth": truth, **result.to_dict()}
        if self._writer is None:
            self._file = self.predictions_path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        self.prediction_count += 1

    def finalize(
        self,
        serial_diagnostics: Mapping[str, Any],
        engine_diagnostics: Mapping[str, Any],
        artifacts: Mapping[str, str],
        final_state: str,
    ) -> None:
        if self._file is not None:
            self._file.close()
        elif not self.predictions_path.exists():
            self.predictions_path.write_text("logged_at_pc,ground_truth\n", encoding="utf-8")

        raw_error = ""
        if self.save_raw and self.raw_packets:
            try:
                write_packets(self.raw_path, self.raw_packets)
            except Exception as exc:  # pragma: no cover - filesystem path
                raw_error = str(exc)

        ended_at = time.time()
        metadata = {
            "schema_version": 1,
            "started_at_unix": self.started_at,
            "ended_at_unix": ended_at,
            "duration_seconds": ended_at - self.started_at,
            "prediction_count": self.prediction_count,
            "raw_packet_count": len(self.raw_packets),
            "raw_stream_saved": self.raw_path.exists(),
            "raw_stream_error": raw_error,
            "final_stable_state": final_state,
            "serial_diagnostics": dict(serial_diagnostics),
            "engine_diagnostics": dict(engine_diagnostics),
            "artifact_paths": dict(artifacts),
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class RealtimeWindow(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.event_queue: queue.Queue = queue.Queue(maxsize=EVENT_QUEUE_MAX_SIZE)
        self.reader: SerialReader | None = None
        self.engine: IncrementalRealtimeInferenceEngine | None = None
        self.recorder: RunRecorder | None = None
        self.speech = SpeechWorker()
        self.running = False
        self.start_pending = False
        self.pending_port = ""
        self.pending_baud = DEFAULT_BAUD
        self.start_delay_remaining = 0
        self.last_result: RealtimeInferenceResult | None = None
        self.latest_esp_stats: dict[str, Any] = {}

        self.countdown_timer = QtCore.QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._countdown_tick)

        self.setWindowTitle("ESP32-S3 CSI — Detecção Realtime de Presença")
        self.resize(1260, 820)
        self._build_ui()
        self._style_ui()
        self.refresh_ports()
        self._show_stable_state("empty")

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(GUI_UPDATE_MS)
        self.timer.timeout.connect(self._process_events)
        self.timer.start()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        controls = QtWidgets.QGroupBox("Conexão e execução")
        grid = QtWidgets.QGridLayout(controls)
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Atualizar portas")
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.setEditable(True)
        self.baud_combo.addItems(["921600", "460800", "115200"])
        self.baud_combo.setCurrentText(str(self.args.baud))
        self.start_button = QtWidgets.QPushButton("Iniciar")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QtWidgets.QPushButton("Parar")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        self.tts_checkbox = QtWidgets.QCheckBox("Anunciar estado final por voz")
        self.tts_checkbox.setChecked(not self.args.no_tts)
        self.tts_test_button = QtWidgets.QPushButton("Testar voz")
        self.tts_test_button.clicked.connect(self.test_tts)
        self.raw_checkbox = QtWidgets.QCheckBox("Salvar fluxo CSI bruto")
        self.raw_checkbox.setChecked(True)

        self.start_delay_spin = QtWidgets.QSpinBox()
        self.start_delay_spin.setRange(0, 300)
        self.start_delay_spin.setValue(int(self.args.start_delay))
        self.start_delay_spin.setSuffix(" s")
        self.start_delay_spin.setSpecialValueText("Sem atraso")
        self.start_delay_spin.setToolTip(
            "Tempo entre pressionar Iniciar e abrir a serial. "
            "Use esse intervalo para sair do ambiente."
        )

        self.truth_combo = QtWidgets.QComboBox()
        for value, label in GROUND_TRUTH_OPTIONS:
            self.truth_combo.addItem(label, value)

        grid.addWidget(QtWidgets.QLabel("Porta:"), 0, 0)
        grid.addWidget(self.port_combo, 0, 1)
        grid.addWidget(self.refresh_button, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Baud:"), 0, 3)
        grid.addWidget(self.baud_combo, 0, 4)
        grid.addWidget(self.start_button, 0, 5)
        grid.addWidget(self.stop_button, 0, 6)
        grid.addWidget(self.tts_checkbox, 1, 0, 1, 2)
        grid.addWidget(self.tts_test_button, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Atraso inicial:"), 1, 3)
        grid.addWidget(self.start_delay_spin, 1, 4)
        grid.addWidget(QtWidgets.QLabel("Estado real:"), 1, 5)
        grid.addWidget(self.truth_combo, 1, 6)
        grid.addWidget(self.raw_checkbox, 2, 0, 1, 2)
        layout.addWidget(controls)

        states = QtWidgets.QHBoxLayout()
        final_group = QtWidgets.QGroupBox("Saída final da máquina de estados")
        final_layout = QtWidgets.QVBoxLayout(final_group)
        self.stable_label = QtWidgets.QLabel(STATE_NAMES["empty"])
        self.stable_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.stable_label.font()
        font.setPointSize(28)
        font.setBold(True)
        self.stable_label.setFont(font)
        self.reason_label = QtWidgets.QLabel("Aguardando início")
        self.reason_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reason_label.setWordWrap(True)
        final_layout.addWidget(self.stable_label)
        final_layout.addWidget(self.reason_label)

        raw_group = QtWidgets.QGroupBox("Classificação bruta")
        raw_layout = QtWidgets.QVBoxLayout(raw_group)
        self.raw_label = QtWidgets.QLabel("—")
        self.raw_label.setAlignment(QtCore.Qt.AlignCenter)
        raw_font = self.raw_label.font()
        raw_font.setPointSize(20)
        raw_font.setBold(True)
        self.raw_label.setFont(raw_font)
        self.confidence_label = QtWidgets.QLabel("Confiança: —")
        self.confidence_label.setAlignment(QtCore.Qt.AlignCenter)
        raw_layout.addWidget(self.raw_label)
        raw_layout.addWidget(self.confidence_label)
        states.addWidget(final_group, 2)
        states.addWidget(raw_group, 1)
        layout.addLayout(states)

        probabilities = QtWidgets.QGroupBox("Probabilidades do classificador")
        pgrid = QtWidgets.QGridLayout(probabilities)
        self.bars: dict[str, QtWidgets.QProgressBar] = {}
        self.percent_labels: dict[str, QtWidgets.QLabel] = {}
        for row, (state, label) in enumerate((
            ("empty", "Vazio"),
            ("static_presence", "Presença estática"),
            ("movement", "Movimento"),
        )):
            pgrid.addWidget(QtWidgets.QLabel(label), row, 0)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            value = QtWidgets.QLabel("0,0%")
            value.setMinimumWidth(65)
            value.setAlignment(QtCore.Qt.AlignRight)
            pgrid.addWidget(bar, row, 1)
            pgrid.addWidget(value, row, 2)
            self.bars[state] = bar
            self.percent_labels[state] = value
        layout.addWidget(probabilities)

        diagnostics = QtWidgets.QGroupBox("Diagnósticos")
        dgrid = QtWidgets.QGridLayout(diagnostics)
        self.connection_label = QtWidgets.QLabel("Desconectado")
        self.buffer_label = QtWidgets.QLabel("Buffer: 0/—")
        self.rate_label = QtWidgets.QLabel("Taxa: 0,0 Hz")
        self.inference_label = QtWidgets.QLabel("Inferências: 0")
        self.packet_label = QtWidgets.QLabel("Pacotes: 0")
        self.error_label = QtWidgets.QLabel("Erros: 0")
        self.delay_label = QtWidgets.QLabel("Atraso: —")
        self.time_label = QtWidgets.QLabel("Inferência: —")
        self.output_label = QtWidgets.QLabel("Saída: —")
        self.output_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        for column, widget in enumerate((self.connection_label, self.buffer_label, self.rate_label, self.inference_label)):
            dgrid.addWidget(widget, 0, column)
        for column, widget in enumerate((self.packet_label, self.error_label, self.delay_label, self.time_label)):
            dgrid.addWidget(widget, 1, column)
        dgrid.addWidget(self.output_label, 2, 0, 1, 4)
        layout.addWidget(diagnostics)

        history = QtWidgets.QGroupBox("Histórico recente")
        history_layout = QtWidgets.QVBoxLayout(history)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Horário", "Real", "Bruto", "Final", "P(vazio)", "P(estática)", "P(mov.)", "Motivo"
        ])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in range(7):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QtWidgets.QHeaderView.Stretch)
        history_layout.addWidget(self.table)
        layout.addWidget(history, 1)

    def _style_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background:#F4F6F8; }
            QGroupBox { font-weight:bold; border:1px solid #BCC5CE; border-radius:8px; margin-top:10px; padding-top:8px; background:#FFFFFF; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
            QPushButton { padding:7px 14px; }
            QComboBox { padding:5px; }
            QProgressBar { border:1px solid #AAB4BE; border-radius:5px; background:#EDF0F3; min-height:19px; }
            QProgressBar::chunk { background:#4B82B5; border-radius:4px; }
        """)

    def refresh_ports(self) -> None:
        current = self.port_combo.currentData() or self.args.port
        self.port_combo.clear()
        for port in serial.tools.list_ports.comports():
            self.port_combo.addItem(f"{port.device} — {port.description}", port.device)
        for index in range(self.port_combo.count()):
            if self.port_combo.itemData(index) == current:
                self.port_combo.setCurrentIndex(index)
                return
        if self.port_combo.count() == 0 and self.args.port:
            self.port_combo.addItem(self.args.port, self.args.port)

    def current_truth(self) -> str:
        return str(self.truth_combo.currentData() or "")

    def test_tts(self) -> None:
        if not self.tts_checkbox.isChecked():
            QtWidgets.QMessageBox.information(self, "TTS", "Ative a opção de voz antes de testar.")
            return
        self.speech.speak("empty", "Teste de voz. Ambiente vazio.")

    def _set_controls_locked(self, locked: bool) -> None:
        self.start_button.setEnabled(not locked)
        self.stop_button.setEnabled(locked)
        self.port_combo.setEnabled(not locked)
        self.baud_combo.setEnabled(not locked)
        self.refresh_button.setEnabled(not locked)
        self.raw_checkbox.setEnabled(not locked)
        self.start_delay_spin.setEnabled(not locked)

    def start_monitoring(self) -> None:
        if self.running or self.start_pending:
            return

        port = str(
            self.port_combo.currentData()
            or self.port_combo.currentText()
        ).strip()
        if not port:
            QtWidgets.QMessageBox.warning(
                self,
                "Porta serial",
                "Selecione uma porta serial.",
            )
            return

        try:
            baud = int(self.baud_combo.currentText())
            self.engine = (
                IncrementalRealtimeInferenceEngine.from_artifacts(
                    pipeline_config_path=self.args.pipeline_config,
                    model_path=self.args.model,
                    state_machine_config_path=(
                        self.args.state_machine_config
                    ),
                    verify_model_hash=not self.args.skip_hash_check,
                )
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Falha ao carregar os artefatos",
                str(exc),
            )
            self.engine = None
            return

        self.pending_port = port
        self.pending_baud = baud
        self.start_delay_remaining = int(
            self.start_delay_spin.value()
        )
        self.start_pending = True
        self.last_result = None
        self.latest_esp_stats = {}

        self._set_controls_locked(True)
        self.stop_button.setText("Cancelar")
        self.table.setRowCount(0)
        self._show_stable_state("empty")
        self.raw_label.setText("—")
        self.confidence_label.setText("Confiança: —")

        for state in self.bars:
            self.bars[state].setValue(0)
            self.percent_labels[state].setText("0,0%")

        if self.start_delay_remaining <= 0:
            self._begin_monitoring_now()
            return

        self.connection_label.setText(
            f"Início em {self.start_delay_remaining} s"
        )
        self.reason_label.setText(
            "Contagem regressiva: saia do ambiente e deixe-o vazio."
        )
        self.output_label.setText(
            "Saída: será criada quando a aquisição começar"
        )

        if self.tts_checkbox.isChecked():
            self.speech.speak(
                "empty",
                (
                    f"Início em {self.start_delay_remaining} segundos. "
                    "Deixe o ambiente vazio."
                ),
            )

        self.countdown_timer.start()

    def _countdown_tick(self) -> None:
        if not self.start_pending:
            self.countdown_timer.stop()
            return

        self.start_delay_remaining -= 1

        if self.start_delay_remaining > 0:
            self.connection_label.setText(
                f"Início em {self.start_delay_remaining} s"
            )
            self.reason_label.setText(
                "Contagem regressiva: saia do ambiente e deixe-o vazio."
            )
            return

        self.countdown_timer.stop()
        self._begin_monitoring_now()

    def _begin_monitoring_now(self) -> None:
        if not self.start_pending or self.engine is None:
            return

        try:
            self.recorder = RunRecorder(
                self.args.output_root,
                self.raw_checkbox.isChecked(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Falha ao criar a pasta da execução",
                str(exc),
            )
            self._cancel_pending_start(
                status_message="Falha ao iniciar"
            )
            return

        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

        self.reader = SerialReader(
            self.pending_port,
            self.pending_baud,
            self.event_queue,
        )
        self.reader.start()

        self.running = True
        self.start_pending = False
        self.stop_button.setText("Parar")
        self.connection_label.setText(
            f"Conectando a {self.pending_port} "
            f"@ {self.pending_baud}"
        )
        self.output_label.setText(
            f"Saída: {self.recorder.run_dir}"
        )
        self.reason_label.setText(
            "Preenchendo o buffer inicial"
        )

        if self.tts_checkbox.isChecked():
            self.speech.speak(
                "empty",
                "Sistema iniciado. Ambiente vazio.",
            )

    def _cancel_pending_start(
        self,
        status_message: str = "Início cancelado",
    ) -> None:
        self.countdown_timer.stop()
        self.start_pending = False
        self.start_delay_remaining = 0
        self.pending_port = ""
        self.engine = None
        self.recorder = None
        self.reader = None

        self._set_controls_locked(False)
        self.stop_button.setText("Parar")
        self.connection_label.setText(status_message)
        self.reason_label.setText("Aguardando início")
        self.output_label.setText("Saída: —")

    def stop_monitoring(self) -> None:
        if self.start_pending:
            self._cancel_pending_start()
            return

        if not self.running and self.reader is None:
            return

        self.running = False

        if self.reader is not None:
            try:
                self.reader.stop()
            except (serial.SerialException, OSError, AttributeError):
                # O pyserial no Windows pode tentar fechar duas vezes
                # o mesmo identificador durante o encerramento.
                pass

        serial_diag = (
            self.reader.diagnostics()
            if self.reader is not None
            else {}
        )
        engine_diag = (
            self.engine.diagnostics().to_dict()
            if self.engine is not None
            else {}
        )

        if self.recorder is not None:
            artifacts = {
                "pipeline_config": str(
                    self.engine.pipeline_config_path
                    if self.engine
                    else ""
                ),
                "model": str(
                    self.engine.model_path
                    if self.engine
                    else ""
                ),
                "state_machine_config": str(
                    self.engine.state_machine_config_path
                    if self.engine
                    else ""
                ),
            }
            final_state = (
                self.last_result.stable_state
                if self.last_result
                else "empty"
            )
            try:
                self.recorder.finalize(
                    serial_diag,
                    engine_diag,
                    artifacts,
                    final_state,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Falha ao salvar",
                    str(exc),
                )

        self.reader = None
        self.engine = None
        self.recorder = None
        self.pending_port = ""

        self.connection_label.setText("Desconectado")
        self.reason_label.setText("Aquisição encerrada")
        self._set_controls_locked(False)
        self.stop_button.setText("Parar")

    def _process_events(self) -> None:
        if self.running and self.reader is not None and not self.reader.running:
            message = self.reader.last_error or "A leitura serial foi encerrada."
            self.connection_label.setText(f"Falha serial: {message}")
            self.stop_monitoring()
            return
        for _ in range(MAX_EVENTS_PER_UPDATE):
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._process_event(event)
        self._update_diagnostics()

    def _process_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "stats":
            metadata = event.get("metadata", {})
            if isinstance(metadata, Mapping):
                self.latest_esp_stats = dict(metadata)
            return
        if event_type != "sample" or not self.running or self.engine is None:
            return
        try:
            packet = sample_event_to_packet(event)
        except (TypeError, ValueError):
            return
        truth = self.current_truth()
        if self.recorder is not None:
            self.recorder.append_packet(packet, truth)
        result = self.engine.push_packet(packet)
        if result is None:
            return
        self.last_result = result
        if self.recorder is not None:
            self.recorder.append_result(result, truth)
        self._display_result(result, truth)
        if result.should_announce and self.tts_checkbox.isChecked():
            self.speech.speak(result.stable_state, STATE_SPEECH.get(result.stable_state, result.stable_state))

    def _display_result(self, result: RealtimeInferenceResult, truth: str) -> None:
        self._show_stable_state(result.stable_state)
        self.raw_label.setText(STATE_NAMES.get(result.raw_state, result.raw_state))
        self.confidence_label.setText(f"Confiança: {result.raw_confidence * 100:.1f}%")
        self.reason_label.setText(result.transition_reason)
        probs = {
            "empty": result.probability_empty,
            "static_presence": result.probability_static_presence,
            "movement": result.probability_movement,
        }
        for state, probability in probs.items():
            self.bars[state].setValue(int(round(probability * 1000)))
            self.percent_labels[state].setText(f"{probability * 100:.1f}%")
        self.delay_label.setText(f"Atraso: {result.pipeline_delay_seconds * 1000:.0f} ms")
        self.time_label.setText(f"Inferência: {result.inference_time_ms:.1f} ms")
        self._append_row(result, truth)

    def _show_stable_state(self, state: str) -> None:
        self.stable_label.setText(STATE_NAMES.get(state, state))
        self.stable_label.setStyleSheet(STATE_STYLES.get(state, ""))

    def _append_row(self, result: RealtimeInferenceResult, truth: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        stamp = datetime.fromtimestamp(
            result.latest_packet_timestamp if result.latest_packet_timestamp > 0 else time.time()
        ).strftime("%H:%M:%S.%f")[:-3]
        values = [
            stamp,
            truth or "—",
            result.raw_state,
            result.stable_state,
            f"{result.probability_empty:.3f}",
            f"{result.probability_static_presence:.3f}",
            f"{result.probability_movement:.3f}",
            result.transition_reason,
        ]
        for column, value in enumerate(values):
            self.table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        while self.table.rowCount() > HISTORY_ROWS:
            self.table.removeRow(0)
        self.table.scrollToBottom()

    def _update_diagnostics(self) -> None:
        if self.engine is None:
            return
        diag = self.engine.diagnostics()
        serial_diag = self.reader.diagnostics() if self.reader is not None else {}
        self.buffer_label.setText(f"Buffer: {diag.buffer_packets}/{diag.required_buffer_packets}")
        self.rate_label.setText(f"Taxa: {diag.packet_rate_hz:.1f} Hz")
        self.inference_label.setText(f"Inferências: {diag.inference_count}")
        self.packet_label.setText(f"Pacotes: {diag.accepted_packets}")
        errors = (
            int(serial_diag.get("crc_errors", 0) or 0)
            + int(serial_diag.get("invalid_frames", 0) or 0)
            + int(serial_diag.get("serial_errors", 0) or 0)
            + diag.rejected_packets
        )
        self.error_label.setText(f"Erros: {errors}")
        if self.running and self.reader is not None:
            self.connection_label.setText("Conectado" if self.reader.running else "Conexão encerrada")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_monitoring()
        self.speech.stop()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI de inferência CSI em tempo real.")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument(
        "--start-delay",
        type=int,
        default=DEFAULT_START_DELAY_SECONDS,
        help=(
            "Atraso em segundos entre pressionar Iniciar e abrir a serial."
        ),
    )
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--state-machine-config", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--check-artifacts", action="store_true")
    return parser.parse_args()


def check_artifacts(args: argparse.Namespace) -> None:
    engine = IncrementalRealtimeInferenceEngine.from_artifacts(
        pipeline_config_path=args.pipeline_config,
        model_path=args.model,
        state_machine_config_path=args.state_machine_config,
        verify_model_hash=not args.skip_hash_check,
    )
    print("\nRealtime GUI Artifact Check")
    print("=" * 72)
    print(f"Pipeline config:       {engine.pipeline_config_path}")
    print(f"Model:                 {engine.model_path}")
    print(f"State-machine config:  {engine.state_machine_config_path}")
    print(f"Expected subcarriers:  {engine.expected_subcarriers}")
    print(f"Window packets:        {engine.window_packets}")
    print(f"Step packets:          {engine.step_packets}")
    print(f"Required buffer:       {engine.required_buffer_packets}")
    print(f"Selected features:     {engine.selected_feature_indices.size}")
    print(f"Initial state:         {engine.state_machine.current_state}")


def main() -> None:
    args = parse_args()
    if args.output_root.is_absolute():
        output_root = args.output_root
    else:
        output_root = PROJECT_ROOT / args.output_root
    args.output_root = output_root

    if args.check_artifacts:
        check_artifacts(args)
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication(sys.argv)
    window = RealtimeWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()