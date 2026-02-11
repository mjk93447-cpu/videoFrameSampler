from __future__ import annotations

import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QPoint, QRect, QThread, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
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
    QFrame,
)

from core.frame_extractor import extract_videos_sequentially, load_first_frame_preview, suggest_fast_mode_interval
from core.models import ExtractionOptions, ImageFormat, MotionSamplingOptions, RoiBox, VideoExtractionResult

MAX_VIDEO_COUNT = 500


class RoiImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Box)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap: QPixmap | None = None
        self._start: QPoint | None = None
        self._current: QPoint | None = None
        self._selection = QRect()

    def set_preview(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self._selection = QRect()
        self.update()

    def clear_selection(self) -> None:
        self._selection = QRect()
        self._start = None
        self._current = None
        self.update()

    def get_selection(self) -> QRect:
        return self._selection.normalized()

    def mousePressEvent(self, event):  # type: ignore[override]
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        self._start = event.position().toPoint()
        self._current = self._start
        self._selection = QRect(self._start, self._current).normalized()
        self.update()

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._start is None:
            return
        self._current = event.position().toPoint()
        self._selection = QRect(self._start, self._current).normalized()
        self.update()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if self._start is None:
            return
        self._current = event.position().toPoint()
        self._selection = QRect(self._start, self._current).normalized()
        self._start = None
        self._current = None
        self.update()

    def paintEvent(self, event):  # type: ignore[override]
        super().paintEvent(event)
        if self._selection.isNull():
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#5DADE2"), 2))
        painter.drawRect(self._selection)
        painter.fillRect(self._selection, QColor(93, 173, 226, 40))


class RoiSelectorDialog(QDialog):
    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Selector")
        self._frame_h, self._frame_w = frame_bgr.shape[:2]
        self.selected_roi: RoiBox | None = None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(1200, 680, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        layout = QVBoxLayout(self)
        info = QLabel("Drag a rectangle on the preview. Only this region will be extracted.")
        layout.addWidget(info)

        self.preview = RoiImageLabel()
        self.preview.set_preview(scaled)
        layout.addWidget(self.preview, alignment=Qt.AlignCenter)

        row = QHBoxLayout()
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.preview.clear_selection)
        self.apply_button = QPushButton("Apply ROI")
        self.apply_button.clicked.connect(self._apply)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        row.addWidget(self.cancel_button)
        row.addWidget(self.apply_button)
        layout.addLayout(row)

    def _apply(self) -> None:
        rect = self.preview.get_selection()
        if rect.width() < 4 or rect.height() < 4:
            self.selected_roi = None
            self.accept()
            return
        scale_x = self._frame_w / self.preview.width()
        scale_y = self._frame_h / self.preview.height()
        roi = RoiBox(
            x=int(rect.x() * scale_x),
            y=int(rect.y() * scale_y),
            width=max(1, int(rect.width() * scale_x)),
            height=max(1, int(rect.height() * scale_y)),
        )
        self.selected_roi = roi.normalized()
        self.accept()


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
        self.setWindowTitle("Video Frame Sampler")
        self.resize(1680, 960)
        self.setMinimumSize(1440, 810)
        self.selected_files: list[Path] = []
        self.selected_roi: RoiBox | None = None
        self.worker_thread: QThread | None = None
        self.worker: ExtractionWorker | None = None
        self._build_ui()
        self._apply_dark_theme()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Video Frame Sampler")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel(
            "Professional frame extraction with ROI crop and motion-focused segment sampling."
        )
        layout.addWidget(subtitle)

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

        self.roi_checkbox = QCheckBox("Enable ROI crop")
        self.roi_button = QPushButton("Open ROI Selector (first frame preview)")
        self.roi_button.clicked.connect(self._open_roi_selector)
        self.roi_label = QLabel("ROI: full frame")

        self.motion_checkbox = QCheckBox("Enable motion segment sampling")
        self.motion_checkbox.stateChanged.connect(
            lambda _state: self.motion_duration_input.setEnabled(self.motion_checkbox.isChecked())
        )
        self.motion_duration_input = QLineEdit("2.0")
        self.motion_duration_input.setPlaceholderText("Target duration in seconds (e.g. 2.0)")
        self.motion_hint = QLabel(
            "Detect one major motion segment and save contiguous frames to output/<video>/motion_segment."
        )
        self.motion_hint.setWordWrap(True)

        controls_layout.addWidget(QLabel("Interval"), 0, 0)
        controls_layout.addWidget(self.interval_input, 0, 1)
        controls_layout.addWidget(QLabel("Image format"), 1, 0)
        controls_layout.addWidget(self.format_combo, 1, 1)
        controls_layout.addWidget(QLabel("JPG quality"), 2, 0)
        controls_layout.addWidget(self.jpg_quality_slider, 2, 1)
        controls_layout.addWidget(self.jpg_quality_value, 2, 2)
        controls_layout.addWidget(self.fast_mode_checkbox, 3, 0, 1, 3)
        controls_layout.addWidget(self.suggest_button, 4, 0, 1, 3)
        controls_layout.addWidget(self.roi_checkbox, 5, 0, 1, 3)
        controls_layout.addWidget(self.roi_button, 6, 0, 1, 3)
        controls_layout.addWidget(self.roi_label, 7, 0, 1, 3)
        controls_layout.addWidget(self.motion_checkbox, 8, 0, 1, 3)
        controls_layout.addWidget(QLabel("Motion target duration (sec)"), 9, 0)
        controls_layout.addWidget(self.motion_duration_input, 9, 1, 1, 2)
        controls_layout.addWidget(self.motion_hint, 10, 0, 1, 3)
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
        self.motion_duration_input.setEnabled(self.motion_checkbox.isChecked())

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
        self.selected_roi = None
        self.roi_label.setText("ROI: full frame")
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

    def _open_roi_selector(self) -> None:
        if not self.selected_files:
            QMessageBox.information(self, "No input", "Select at least one video file first.")
            return
        frame = load_first_frame_preview(self.selected_files[0])
        if frame is None:
            QMessageBox.warning(self, "Preview error", "Failed to load first-frame preview.")
            return
        dialog = RoiSelectorDialog(frame, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.selected_roi = dialog.selected_roi
        if self.selected_roi is None:
            self.roi_label.setText("ROI: full frame")
            self._append_log("ROI cleared. Full frame extraction is active.")
        else:
            roi = self.selected_roi
            self.roi_label.setText(f"ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")
            self._append_log(f"ROI updated: x={roi.x}, y={roi.y}, width={roi.width}, height={roi.height}")

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
        motion_sampling: MotionSamplingOptions | None = None
        if self.motion_checkbox.isChecked():
            try:
                duration = float(self.motion_duration_input.text().strip())
                if duration <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Invalid duration", "Motion duration must be a positive number.")
                return None
            motion_sampling = MotionSamplingOptions(enabled=True, expected_duration_sec=duration)

        roi = self.selected_roi if self.roi_checkbox.isChecked() else None

        return ExtractionOptions(
            interval=interval,
            image_format=image_format,
            jpg_quality=self.jpg_quality_slider.value(),
            fast_mode=self.fast_mode_checkbox.isChecked(),
            roi=roi,
            motion_sampling=motion_sampling,
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.select_button.setEnabled(enabled)
        self.start_button.setEnabled(enabled)
        self.suggest_button.setEnabled(enabled)
        self.interval_input.setEnabled(enabled)
        self.fast_mode_checkbox.setEnabled(enabled)
        self.roi_checkbox.setEnabled(enabled)
        self.roi_button.setEnabled(enabled)
        self.motion_checkbox.setEnabled(enabled)
        self.motion_duration_input.setEnabled(enabled and self.motion_checkbox.isChecked())
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

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background-color: #181A1F; color: #EAECEE; font-size: 11pt; }
            QGroupBox { border: 1px solid #30343B; border-radius: 8px; margin-top: 10px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #A9DFBF; }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #111317; border: 1px solid #3A3F47; border-radius: 6px; padding: 6px;
            }
            QPushButton {
                background-color: #2C3E50; border: 1px solid #3F5873; border-radius: 6px; padding: 7px 10px;
            }
            QPushButton:hover { background-color: #34495E; }
            QPushButton:disabled { color: #7F8C8D; background-color: #22262C; }
            QProgressBar { border: 1px solid #3A3F47; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #1ABC9C; border-radius: 4px; }
            """
        )


def run_app() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
