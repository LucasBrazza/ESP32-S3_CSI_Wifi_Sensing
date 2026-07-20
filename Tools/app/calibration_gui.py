from __future__ import annotations

"""
Optional reference verification.

The trained model, normalization parameters, selected subcarriers and
selected features are loaded from the exported artifacts. This screen does
not fit, recalibrate or modify any of them.
"""

import argparse
import importlib
import time
from typing import Any

from PyQt5 import QtWidgets


BASE = importlib.import_module("Tools.realtime.03_realtime_gui")

RealtimeWindow = BASE.RealtimeWindow


class CalibrationWindow(RealtimeWindow):
    """Records the model response to a reference condition."""

    def __init__(self, args: argparse.Namespace) -> None:
        args.start_mode = "calibration"
        super().__init__(args)

        self.setWindowTitle(
            "Calibração opcional — Verificação de referência"
        )
        self.start_button.setText("Iniciar verificação")
        self.stop_button.setText("Cancelar")
        self.raw_checkbox.setChecked(True)

        self._show_phase(
            "VERIFICAÇÃO OPCIONAL",
            (
                "Não altera o modelo. Execute somente após mudar "
                "a posição dos dispositivos, o ambiente ou os "
                "artefatos treinados."
            ),
        )

    def _finish_calibration(self) -> None:
        if self.engine is None or not self.calibration_results:
            return

        results = list(self.calibration_results)
        total = len(results)

        counts = {
            "empty": sum(
                item.raw_state == "empty"
                for item in results
            ),
            "static_presence": sum(
                item.raw_state == "static_presence"
                for item in results
            ),
            "movement": sum(
                item.raw_state == "movement"
                for item in results
            ),
        }

        means = {
            "empty": sum(
                item.probability_empty
                for item in results
            ) / total,
            "static_presence": sum(
                item.probability_static_presence
                for item in results
            ) / total,
            "movement": sum(
                item.probability_movement
                for item in results
            ) / total,
        }

        dominant_state = max(
            counts,
            key=counts.get,
        )
        diagnostics = self.engine.diagnostics()

        summary: dict[str, Any] = {
            "type": "optional_reference_check",
            "changes_model_parameters": False,
            "completed_at_unix": time.time(),
            "evaluated_windows": total,
            "dominant_raw_state": dominant_state,
            "raw_state_counts": counts,
            "mean_probabilities": means,
            "packet_rate_hz": diagnostics.packet_rate_hz,
            "note": (
                "This check does not approve, reject or modify "
                "the trained model."
            ),
        }

        self.calibration_summary = summary

        if self.recorder is not None:
            self.recorder.save_calibration(summary)
            run_dir = self.recorder.run_dir
        else:
            run_dir = None

        self.stop_monitoring()

        names = {
            "empty": "ambiente vazio",
            "static_presence": "presença estática",
            "movement": "movimento",
        }

        self._show_phase(
            "VERIFICAÇÃO CONCLUÍDA",
            (
                "Resposta predominante: "
                f"{names.get(dominant_state, dominant_state)}. "
                "Nenhum parâmetro foi alterado."
            ),
        )

        message = QtWidgets.QMessageBox(self)
        message.setWindowTitle("Verificação concluída")
        message.setIcon(QtWidgets.QMessageBox.Information)
        message.setText(
            "A referência foi registrada sem modificar o modelo."
        )
        message.setInformativeText(
            f"Janelas avaliadas: {total}\n"
            f"Vazio: {counts['empty']}\n"
            f"Presença estática: {counts['static_presence']}\n"
            f"Movimento: {counts['movement']}\n\n"
            f"Probabilidade média de vazio: "
            f"{means['empty'] * 100:.1f}%\n"
            f"Probabilidade média de presença estática: "
            f"{means['static_presence'] * 100:.1f}%\n"
            f"Probabilidade média de movimento: "
            f"{means['movement'] * 100:.1f}%\n"
            f"Taxa de pacotes: {diagnostics.packet_rate_hz:.1f} Hz"
            + (
                f"\n\nArquivos: {run_dir}"
                if run_dir is not None
                else ""
            )
        )
        message.exec_()
