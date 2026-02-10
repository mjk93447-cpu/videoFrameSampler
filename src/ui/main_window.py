from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from core.frame_extractor import extract_videos_sequentially, suggest_fast_mode_interval
from core.models import ExtractionOptions, ImageFormat, VideoExtractionResult

MAX_VIDEO_COUNT = 500


class ExtractionWorker(QObject):
    progress = Signal(str)
    progress_value = Signal(int)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, video_paths: list[Path], options: ExtractionOptions):
        super().__init__()
        self.video_paths = video_paths
        self.options = options

    def run(self) -> None:
        try:
            total = max(1, len(self.video_paths))
            handled = 0
            results: list[VideoExtractionResult] = []

            def callback(message: str) -> None:
                self.progress.emit(message)

            for result in extract_videos_sequentially(self.video_paths, self.options, progress_cb=callback):
                handled += 1
                results.append(result)
                self.progress_value.emit(int((handled / total) * 100))

            self.done.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("videoFrameSampler ver2")
        self.resize(960, 680)
        self.selected_files: list[Path] = []
        self.worker_thread: QThread | None = None
        self.worker: ExtractionWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Video Frame Sampler")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        top_row = QHBoxLayout()
        self.select_button = QPushButton("Select Videos (max 500)")
        self.select_button.clicked.connect(self._pick_videos)
        self.files_label = QLabel("No files selected.")
        top_row.addWidget(self.select_button)
        top_row.addWidget(self.files_label, 1)
        layout.addLayout(top_row)

        controls = QGroupBox("Extraction Options")
        controls_layout = QGridLayout(controls)

        self.interval_input = QLineEdit("1")
        self.interval_input.setPlaceholderText("Sampling interval (>= 1)")

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG"])
        self.format_combo.currentTextChanged.connect(self._update_jpg_ui_state)

        self.jpg_quality_slider = QSlider()
        self.jpg_quality_slider.setOrientation(Qt.Horizontal)
        self.jpg_quality_slider.setMinimum(1)
        self.jpg_quality_slider.setMaximum(100)
        self.jpg_quality_slider.setValue(95)
        self.jpg_quality_value = QLabel("95")
        self.jpg_quality_slider.valueChanged.connect(
            lambda value: self.jpg_quality_value.setText(str(value))
        )

        self.fast_mode_checkbox = QCheckBox("Fast mode (JPG default + suggested interval)")
        self.fast_mode_checkbox.stateChanged.connect(self._toggle_fast_mode)
        self.suggest_button = QPushButton("Suggest interval from first video")
        self.suggest_button.clicked.connect(self._suggest_interval)

        controls_layout.addWidget(QLabel("Interval"), 0, 0)
        controls_layout.addWidget(self.interval_input, 0, 1)
        controls_layout.addWidget(QLabel("Image format"), 1, 0)
        controls_layout.addWidget(self.format_combo, 1, 1)
        controls_layout.addWidget(QLabel("JPG quality"), 2, 0)
        controls_layout.addWidget(self.jpg_quality_slider, 2, 1)
        controls_layout.addWidget(self.jpg_quality_value, 2, 2)
        controls_layout.addWidget(self.fast_mode_checkbox, 3, 0, 1, 3)
        controls_layout.addWidget(self.suggest_button, 4, 0, 1, 3)
        layout.addWidget(controls)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("Start Extraction")
        self.start_button.clicked.connect(self._start_extraction)
        action_row.addWidget(self.start_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        layout.addWidget(self.progress)

        self.summary = QLabel("Ready.")
        layout.addWidget(self.summary)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)

        self.setCentralWidget(central)
        self._update_jpg_ui_state(self.format_combo.currentText())

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)

    def _pick_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm);;All files (*.*)",
        )
        if not files:
            return
        if len(files) > MAX_VIDEO_COUNT:
            QMessageBox.warning(
                self,
                "Too many files",
                f"Selected {len(files)} files. Maximum allowed is {MAX_VIDEO_COUNT}.",
            )
            return
        self.selected_files = [Path(f) for f in files]
        self.files_label.setText(f"{len(self.selected_files)} file(s) selected.")
        self._append_log(f"Selected {len(self.selected_files)} file(s).")

    def _toggle_fast_mode(self) -> None:
        if self.fast_mode_checkbox.isChecked():
            self.format_combo.setCurrentText("JPG")
            self._append_log("Fast mode enabled. JPG selected by default.")
        else:
            self._append_log("Fast mode disabled.")

    def _suggest_interval(self) -> None:
        if not self.selected_files:
            QMessageBox.information(self, "No file", "Select at least one video first.")
            return
        suggestion = suggest_fast_mode_interval(self.selected_files[0])
        self.interval_input.setText(str(suggestion))
        self._append_log(f"Suggested interval: {suggestion}")

    def _update_jpg_ui_state(self, selected_format: str) -> None:
        enabled = selected_format.upper() == "JPG"
        self.jpg_quality_slider.setEnabled(enabled)
        self.jpg_quality_value.setEnabled(enabled)

    def _validate_interval(self) -> int | None:
        value = self.interval_input.text().strip()
        try:
            interval = int(value)
            if interval < 1:
                raise ValueError
            return interval
        except ValueError:
            QMessageBox.warning(self, "Invalid interval", "Interval must be an integer >= 1.")
            return None

    def _collect_options(self) -> ExtractionOptions | None:
        interval = self._validate_interval()
        if interval is None:
            return None
        selected = self.format_combo.currentText().upper()
        image_format = ImageFormat.JPG if selected == "JPG" else ImageFormat.PNG
        return ExtractionOptions(
            interval=interval,
            image_format=image_format,
            jpg_quality=self.jpg_quality_slider.value(),
            fast_mode=self.fast_mode_checkbox.isChecked(),
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.select_button.setEnabled(enabled)
        self.start_button.setEnabled(enabled)
        self.suggest_button.setEnabled(enabled)
        self.interval_input.setEnabled(enabled)
        self.fast_mode_checkbox.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.jpg_quality_slider.setEnabled(enabled and self.format_combo.currentText() == "JPG")

    def _start_extraction(self) -> None:
        if not self.selected_files:
            QMessageBox.warning(self, "No input", "Please select at least one video file.")
            return

        options = self._collect_options()
        if options is None:
            return

        self.progress.setValue(0)
        self.summary.setText("Running...")
        self._append_log("Extraction started.")
        self._set_controls_enabled(False)

        self.worker_thread = QThread(self)
        self.worker = ExtractionWorker(self.selected_files, options)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._append_log)
        self.worker.progress_value.connect(self.progress.setValue)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.done.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(lambda: self._set_controls_enabled(True))
        self.worker_thread.start()

    def _on_done(self, results: list[VideoExtractionResult]) -> None:
        success = sum(1 for r in results if r.success)
        failures = [r for r in results if not r.success]
        self.summary.setText(f"Completed. Success: {success}, Failed: {len(failures)}")

        for result in failures:
            self._append_log(f"[ERROR] {result.video_path.name}: {result.message}")
        for result in results:
            if result.success:
                self._append_log(
                    f"[OK] {result.video_path.name}: {result.saved_count} frame(s) -> {result.output_dir}"
                )

        if failures:
            QMessageBox.warning(
                self,
                "Completed with errors",
                f"Success: {success}, Failed: {len(failures)}. Check logs.",
            )
        else:
            QMessageBox.information(self, "Done", "All videos were processed successfully.")

    def _on_failed(self, traceback_text: str) -> None:
        self.summary.setText("Failed due to unexpected error.")
        self._append_log(traceback_text)
        QMessageBox.critical(self, "Unexpected error", traceback_text)


def run_app() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
