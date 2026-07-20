from __future__ import annotations

"""Fullscreen entry point for the complete ESP32-S3 CSI desktop system."""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets


THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = TOOLS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = (
    TOOLS_DIR
    / "datasets/processed/realtime_model_extra_trees.joblib"
)
PIPELINE_CONFIG_PATH = (
    TOOLS_DIR
    / "datasets/processed/realtime_pipeline_config_extra_trees.json"
)
STATE_MACHINE_CONFIG_PATH = (
    TOOLS_DIR
    / "realtime/state_machine_config_candidate_v4.json"
)
REALTIME_RUNS_PATH = TOOLS_DIR / "datasets/realtime_runs"
RESULTS_PATH = TOOLS_DIR / "datasets/results"


class TrainingWindow(QtWidgets.QMainWindow):
    """Runs Dataset v2 training inside the desktop app."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Treinamento do Modelo — Dataset v2")
        self.process: QtCore.QProcess | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(24, 18, 24, 24)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Treinamento e exportação do modelo")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title_font = title.font()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        info = QtWidgets.QLabel(
            "Executa o pipeline completo do Dataset v2 e atualiza "
            "os artefatos usados pela detecção realtime."
        )
        info.setAlignment(QtCore.Qt.AlignCenter)
        info.setWordWrap(True)
        layout.addWidget(info)

        controls = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Iniciar treinamento")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QtWidgets.QPushButton("Interromper")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        controls.addStretch(1)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.status_label = QtWidgets.QLabel("Aguardando")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(
            QtWidgets.QPlainTextEdit.NoWrap
        )
        font = QtGui.QFont("Consolas")
        font.setPointSize(10)
        self.log.setFont(font)
        layout.addWidget(self.log, 1)

        self.setStyleSheet(
            """
            QMainWindow { background:#F2F5F8; }
            QLabel { color:#263746; }
            QPushButton {
                min-width:190px;
                min-height:44px;
                padding:8px 18px;
                border-radius:7px;
                background:#2E628F;
                color:white;
                font-weight:bold;
            }
            QPushButton:disabled { background:#9AA7B2; }
            QPlainTextEdit {
                background:#101820;
                color:#E7EEF5;
                border:1px solid #7C8995;
                border-radius:7px;
                padding:8px;
            }
            """
        )

    def start_training(self) -> None:
        if self.process is not None:
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "Iniciar treinamento",
            (
                "O treinamento completo pode demorar bastante e utilizar "
                "intensamente o processador. Deseja continuar?"
            ),
            QtWidgets.QMessageBox.Yes
            | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        self.log.clear()
        self.status_label.setText("Treinamento em execução")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProcessChannelMode(
            QtCore.QProcess.MergedChannels
        )
        process.readyReadStandardOutput.connect(
            self._read_process_output
        )
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process

        process.start(
            sys.executable,
            [
                "-m",
                "Tools.training.20_retrain_dataset_v2",
            ],
        )

    def stop_training(self) -> None:
        if self.process is None:
            return

        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self.status_label.setText("Treinamento interrompido")

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        data = bytes(
            self.process.readAllStandardOutput()
        ).decode("utf-8", errors="replace")
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.insertPlainText(data)
        self.log.moveCursor(QtGui.QTextCursor.End)

    def _process_finished(
        self,
        exit_code: int,
        exit_status: QtCore.QProcess.ExitStatus,
    ) -> None:
        self._read_process_output()
        success = (
            exit_status == QtCore.QProcess.NormalExit
            and exit_code == 0
        )
        self.status_label.setText(
            "Treinamento concluído"
            if success
            else f"Treinamento encerrado com código {exit_code}"
        )
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.process = None

    def _process_error(
        self,
        error: QtCore.QProcess.ProcessError,
    ) -> None:
        self.status_label.setText(
            f"Erro ao iniciar processo: {error}"
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.process is not None:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Treinamento em execução",
                "Interromper o treinamento e voltar ao menu?",
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.stop_training()
        event.accept()


class MenuWindow(QtWidgets.QMainWindow):
    """Single fullscreen menu for all user-facing system functions."""

    def __init__(self) -> None:
        super().__init__()
        self.active_window: QtWidgets.QMainWindow | None = None
        self.setWindowTitle("ESP32-S3 CSI Wi-Fi Sensing")
        self._build_ui()
        self._refresh_system_status()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(42, 30, 42, 30)
        root.setSpacing(20)

        title = QtWidgets.QLabel(
            "ESP32-S3 CSI Wi-Fi Sensing"
        )
        title.setAlignment(QtCore.Qt.AlignCenter)
        title_font = title.font()
        title_font.setPointSize(34)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Aquisição, treinamento e detecção passiva de presença"
        )
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle_font = subtitle.font()
        subtitle_font.setPointSize(15)
        subtitle.setFont(subtitle_font)
        root.addWidget(subtitle)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        cards = QtWidgets.QGridLayout()
        cards.setHorizontalSpacing(24)
        cards.setVerticalSpacing(24)

        acquisition = self._menu_card(
            "Aquisição do Dataset",
            (
                "Coleta manual ou programada de CSI, "
                "gráficos e armazenamento binário."
            ),
            self.open_acquisition,
        )
        realtime = self._menu_card(
            "Detecção Realtime",
            (
                "Classificação com os parâmetros exportados, "
                "máquina de estados, gravação e voz."
            ),
            self.open_realtime,
        )
        calibration = self._menu_card(
            "Calibração opcional",
            (
                "Registra a resposta do modelo a uma referência "
                "sem alterar os parâmetros treinados."
            ),
            self.open_calibration,
        )
        training = self._menu_card(
            "Treinar Modelo",
            (
                "Executa o treinamento completo do Dataset v2 "
                "e exporta os artefatos finais."
            ),
            self.open_training,
        )
        results = self._menu_card(
            "Resultados e Gravações",
            (
                "Abre as pastas de resultados experimentais "
                "e das execuções realtime."
            ),
            self.open_results_menu,
        )

        cards.addWidget(acquisition, 0, 0)
        cards.addWidget(realtime, 0, 1)
        cards.addWidget(calibration, 1, 0)
        cards.addWidget(training, 1, 1)
        cards.addWidget(results, 2, 0, 1, 2)
        root.addLayout(cards, 1)

        footer = QtWidgets.QHBoxLayout()
        verify_button = QtWidgets.QPushButton(
            "Verificar instalação"
        )
        verify_button.clicked.connect(
            self.verify_installation
        )
        exit_button = QtWidgets.QPushButton("Encerrar")
        exit_button.clicked.connect(self.close)
        footer.addWidget(verify_button)
        footer.addStretch(1)
        footer.addWidget(
            QtWidgets.QLabel(
                "Nas telas auxiliares, pressione Esc "
                "ou use “Voltar ao menu”."
            )
        )
        footer.addStretch(1)
        footer.addWidget(exit_button)
        root.addLayout(footer)

        self.setStyleSheet(
            """
            QMainWindow {
                background:#EAF0F5;
            }
            QLabel {
                color:#243746;
            }
            QPushButton {
                min-height:42px;
                padding:8px 18px;
                border-radius:7px;
                background:#3A6F9D;
                color:white;
                font-weight:bold;
            }
            QPushButton:hover {
                background:#2E5E86;
            }
            QPushButton:pressed {
                background:#214967;
            }
            QFrame#menuCard {
                background:#FFFFFF;
                border:1px solid #B6C3CE;
                border-radius:14px;
            }
            """
        )

    def _menu_card(
        self,
        title: str,
        description: str,
        action: Callable[[], None],
    ) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("menuCard")
        frame.setMinimumSize(380, 145)

        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(14)

        title_label = QtWidgets.QLabel(title)
        title_font = title_label.font()
        title_font.setPointSize(21)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(QtCore.Qt.AlignCenter)

        description_label = QtWidgets.QLabel(description)
        description_label.setAlignment(QtCore.Qt.AlignCenter)
        description_label.setWordWrap(True)
        description_font = description_label.font()
        description_font.setPointSize(12)
        description_label.setFont(description_font)

        button = QtWidgets.QPushButton("Abrir")
        button.clicked.connect(action)

        layout.addWidget(title_label)
        layout.addWidget(description_label, 1)
        layout.addWidget(button)
        return frame

    def _refresh_system_status(self) -> None:
        missing = [
            path.name
            for path in (
                MODEL_PATH,
                PIPELINE_CONFIG_PATH,
                STATE_MACHINE_CONFIG_PATH,
            )
            if not path.exists()
        ]

        if missing:
            self.status_label.setText(
                "Atenção: faltam artefatos do realtime: "
                + ", ".join(missing)
            )
            self.status_label.setStyleSheet(
                "color:#9A3B28;font-weight:bold;"
            )
        else:
            self.status_label.setText(
                "Sistema pronto — modelo e configurações encontrados"
            )
            self.status_label.setStyleSheet(
                "color:#27643B;font-weight:bold;"
            )

    def _show_child(
        self,
        window: QtWidgets.QMainWindow,
        title: str,
    ) -> None:
        if self.active_window is not None:
            self.active_window.raise_()
            self.active_window.activateWindow()
            return

        self.active_window = window
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QtCore.QSize(22, 22))

        back_action = QtWidgets.QAction(
            "Voltar ao menu (Esc)",
            window,
        )
        back_action.triggered.connect(window.close)
        toolbar.addAction(back_action)
        toolbar.addSeparator()

        title_label = QtWidgets.QLabel(title)
        title_font = title_label.font()
        title_font.setBold(True)
        title_label.setFont(title_font)
        toolbar.addWidget(title_label)
        window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

        escape = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Escape),
            window,
        )
        escape.activated.connect(window.close)
        window._menu_escape_shortcut = escape
        window._menu_toolbar = toolbar

        window.destroyed.connect(self._child_closed)
        self.hide()
        window.showFullScreen()

    def _child_closed(self) -> None:
        self.active_window = None
        self._refresh_system_status()
        self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def open_acquisition(self) -> None:
        try:
            module = importlib.import_module(
                "Tools.acquisition.gui.csi_viewer"
            )
            window = module.CSIViewer()
        except Exception as exc:
            self._show_error(
                "Não foi possível abrir a aquisição.",
                exc,
            )
            return

        self._show_child(window, "Aquisição do Dataset")

    def open_realtime(self) -> None:
        missing = [
            path
            for path in (
                MODEL_PATH,
                PIPELINE_CONFIG_PATH,
                STATE_MACHINE_CONFIG_PATH,
            )
            if not path.exists()
        ]
        if missing:
            QtWidgets.QMessageBox.critical(
                self,
                "Artefatos ausentes",
                "Arquivos necessários não encontrados:\n\n"
                + "\n".join(str(path) for path in missing),
            )
            return

        try:
            module = importlib.import_module(
                "Tools.realtime.03_realtime_gui"
            )
            args = argparse.Namespace(
                port="COM4",
                baud=921600,
                start_delay=10,
                calibration_seconds=8,
                start_mode="monitoring",
                pipeline_config=PIPELINE_CONFIG_PATH,
                model=MODEL_PATH,
                state_machine_config=STATE_MACHINE_CONFIG_PATH,
                output_root=REALTIME_RUNS_PATH,
                skip_hash_check=False,
                no_tts=False,
                check_artifacts=False,
            )
            REALTIME_RUNS_PATH.mkdir(
                parents=True,
                exist_ok=True,
            )
            window = module.RealtimeWindow(args)
        except Exception as exc:
            self._show_error(
                "Não foi possível abrir a detecção realtime.",
                exc,
            )
            return

        self._show_child(window, "Detecção Realtime")

    def open_calibration(self) -> None:
        missing = [
            path
            for path in (
                MODEL_PATH,
                PIPELINE_CONFIG_PATH,
                STATE_MACHINE_CONFIG_PATH,
            )
            if not path.exists()
        ]
        if missing:
            QtWidgets.QMessageBox.critical(
                self,
                "Artefatos ausentes",
                "Arquivos necessários não encontrados:\n\n"
                + "\n".join(str(path) for path in missing),
            )
            return

        try:
            module = importlib.import_module(
                "Tools.app.calibration_gui"
            )
            args = argparse.Namespace(
                port="COM4",
                baud=921600,
                start_delay=10,
                calibration_seconds=8,
                start_mode="calibration",
                pipeline_config=PIPELINE_CONFIG_PATH,
                model=MODEL_PATH,
                state_machine_config=STATE_MACHINE_CONFIG_PATH,
                output_root=REALTIME_RUNS_PATH,
                skip_hash_check=False,
                no_tts=False,
                check_artifacts=False,
            )
            REALTIME_RUNS_PATH.mkdir(
                parents=True,
                exist_ok=True,
            )
            window = module.CalibrationWindow(args)
        except Exception as exc:
            self._show_error(
                "Não foi possível abrir a calibração opcional.",
                exc,
            )
            return

        self._show_child(
            window,
            "Calibração opcional",
        )

    def open_training(self) -> None:
        self._show_child(
            TrainingWindow(),
            "Treinamento do Modelo",
        )

    def open_results_menu(self) -> None:
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Resultados e gravações")
        dialog.setText("Escolha a pasta que deseja abrir.")
        runs_button = dialog.addButton(
            "Execuções realtime",
            QtWidgets.QMessageBox.ActionRole,
        )
        results_button = dialog.addButton(
            "Resultados experimentais",
            QtWidgets.QMessageBox.ActionRole,
        )
        dialog.addButton(
            "Cancelar",
            QtWidgets.QMessageBox.RejectRole,
        )
        dialog.exec_()

        clicked = dialog.clickedButton()
        if clicked is runs_button:
            self._open_folder(REALTIME_RUNS_PATH)
        elif clicked is results_button:
            self._open_folder(RESULTS_PATH)

    def verify_installation(self) -> None:
        checks: list[tuple[str, bool, str]] = []

        for name, path in (
            ("Modelo realtime", MODEL_PATH),
            ("Configuração do pipeline", PIPELINE_CONFIG_PATH),
            ("Máquina de estados", STATE_MACHINE_CONFIG_PATH),
            (
                "GUI de aquisição",
                TOOLS_DIR / "acquisition/gui/csi_viewer.py",
            ),
            (
                "GUI realtime",
                TOOLS_DIR / "realtime/03_realtime_gui.py",
            ),
            (
                "Calibração opcional",
                TOOLS_DIR / "app/calibration_gui.py",
            ),
        ):
            checks.append(
                (name, path.exists(), str(path))
            )

        try:
            import serial.tools.list_ports
            ports = [
                port.device
                for port in serial.tools.list_ports.comports()
            ]
            checks.append(
                (
                    "PySerial e portas",
                    True,
                    ", ".join(ports) if ports else "nenhuma porta",
                )
            )
        except Exception as exc:
            checks.append(("PySerial", False, str(exc)))

        try:
            import numpy
            import joblib
            import pyqtgraph
            checks.append(
                (
                    "Bibliotecas principais",
                    True,
                    (
                        f"numpy {numpy.__version__}, "
                        f"joblib {joblib.__version__}, "
                        f"pyqtgraph {pyqtgraph.__version__}"
                    ),
                )
            )
        except Exception as exc:
            checks.append(
                ("Bibliotecas principais", False, str(exc))
            )

        lines = []
        all_ok = True
        for name, ok, detail in checks:
            all_ok = all_ok and ok
            marker = "OK" if ok else "FALHA"
            lines.append(f"[{marker}] {name}\n{detail}")

        message = QtWidgets.QMessageBox(self)
        message.setWindowTitle("Verificação da instalação")
        message.setIcon(
            QtWidgets.QMessageBox.Information
            if all_ok
            else QtWidgets.QMessageBox.Warning
        )
        message.setText(
            "Sistema pronto."
            if all_ok
            else "Foram encontradas pendências."
        )
        message.setDetailedText("\n\n".join(lines))
        message.exec_()
        self._refresh_system_status()

    @staticmethod
    def _open_folder(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(path))
            else:
                QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl.fromLocalFile(str(path))
                )
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                None,
                "Não foi possível abrir a pasta",
                str(exc),
            )

    def _show_error(
        self,
        message: str,
        error: Exception,
    ) -> None:
        QtWidgets.QMessageBox.critical(
            self,
            "Falha ao abrir módulo",
            f"{message}\n\n{error}",
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Encerrar sistema",
            "Deseja encerrar o programa?",
            QtWidgets.QMessageBox.Yes
            | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            event.accept()
            application = QtWidgets.QApplication.instance()
            if application is not None:
                application.quit()
        else:
            event.ignore()


def main() -> None:
    REALTIME_RUNS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(
        "ESP32-S3 CSI Wi-Fi Sensing"
    )
    app.setQuitOnLastWindowClosed(False)

    menu = MenuWindow()
    menu.showFullScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
