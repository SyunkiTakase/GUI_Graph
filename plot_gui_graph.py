import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog,
    QLabel, QLineEdit, QMessageBox, QCheckBox, QTabWidget
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
from matplotlib import rcParams

class LearningCurvePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Learning Curve Plotter")
        self.resize(1200, 700)
        self.data = None
        self.metrics = []
        self.color_map = {}
        # Uniform figure size for display and save
        self.fig_size = (8, 6)  # inches

        # Central widget and layouts
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left side: toolbar + tabs
        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        main_layout.addWidget(left_widget, stretch=3)

        # Tab widget for plots
        self.tabs = QTabWidget()
        self.left_layout.addWidget(self.tabs)
        self.toolbar = None  # will attach after plotting

        # Control panel
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        main_layout.addWidget(ctrl, stretch=1)

        # Load CSV
        btn_load = QPushButton("Load log.csv")
        btn_load.clicked.connect(self.load_csv)
        ctrl_layout.addWidget(btn_load)

        # Separate tabs option
        self.cb_sep_tabs = QCheckBox("Use Separate Tabs for Each Metric")
        ctrl_layout.addWidget(self.cb_sep_tabs)

        # Grid ON option
        self.cb_grid = QCheckBox("Grid ON")
        ctrl_layout.addWidget(self.cb_grid)

        # Metrics list label and select-all
        ctrl_layout.addWidget(QLabel("Select Metrics to Plot:"))
        btn_select_all = QPushButton("Plot All")
        btn_select_all.clicked.connect(self.select_all_metrics)
        ctrl_layout.addWidget(btn_select_all)

        # Metrics list widget
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.NoSelection)
        ctrl_layout.addWidget(self.list_widget)

        # Plot & Save buttons
        btn_plot = QPushButton("Plot Selected")
        btn_plot.clicked.connect(self.plot_selected)
        ctrl_layout.addWidget(btn_plot)

        ctrl_layout.addWidget(QLabel("Save As (base filename, e.g. plot.png):"))
        self.edit_filename = QLineEdit("plot.png")
        ctrl_layout.addWidget(self.edit_filename)
        btn_save = QPushButton("Save Plots")
        btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(btn_save)

        ctrl_layout.addStretch()

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV log file", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            self.data = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{e}")
            return

        # Extract base metric names, filter out 'epoch'
        bases = []
        for col in self.data.columns:
            col_lower = col.lower()
            if col_lower.startswith("train_"):
                base = col[6:]
            elif col_lower.startswith("val_"):
                base = col[4:]
            else:
                base = col
            bases.append(base)
        self.metrics = sorted({m for m in bases if m.lower() != "epoch"})

        # Create consistent color map
        default_colors = rcParams['axes.prop_cycle'].by_key()['color']
        self.color_map = {m: default_colors[i % len(default_colors)] for i, m in enumerate(self.metrics)}

        # Populate list widget
        self.list_widget.clear()
        for m in self.metrics:
            item = QListWidgetItem(m)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        QMessageBox.information(self, "Loaded", f"Loaded {path}\nDetected metrics: {', '.join(self.metrics)}")

    def select_all_metrics(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def plot_selected(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
            return
        selected = [self.list_widget.item(i).text() for i in range(self.list_widget.count()) if self.list_widget.item(i).checkState() == Qt.Checked]
        if not selected:
            QMessageBox.warning(self, "Warning", "Select at least one metric to plot.")
            return

        use_sep = self.cb_sep_tabs.isChecked()
        grid_on = self.cb_grid.isChecked()
        loss_metrics = [m for m in selected if 'acc' not in m.lower()]
        acc_metrics = [m for m in selected if 'acc' in m.lower()]

        # Clear existing tabs and toolbar
        self.tabs.clear()
        if self.toolbar:
            self.toolbar.setParent(None)

        # Plotting: use uniform figsize
        def new_fig(): return Figure(figsize=self.fig_size)

        if use_sep:
            for base in selected:
                fig = new_fig(); ax = fig.add_subplot(111)
                self._plot_metric(ax, base); ax.set_title(base); ax.grid(grid_on); ax.legend(); fig.tight_layout()
                canvas=FigureCanvas(fig); tab=QWidget(); layout=QVBoxLayout(tab); layout.addWidget(canvas); self.tabs.addTab(tab, base)
            if loss_metrics:
                fig = new_fig(); ax = fig.add_subplot(111)
                for m in loss_metrics: self._plot_metric(ax, m)
                ax.set_title('All Loss'); ax.grid(grid_on); ax.legend(); fig.tight_layout()
                canvas=FigureCanvas(fig); tab=QWidget(); layout=QVBoxLayout(tab); layout.addWidget(canvas); self.tabs.addTab(tab, 'All Loss')
            if acc_metrics:
                fig = new_fig(); ax = fig.add_subplot(111)
                for m in acc_metrics: self._plot_metric(ax, m)
                ax.set_title('All Accuracy'); ax.grid(grid_on); ax.legend(); fig.tight_layout()
                canvas=FigureCanvas(fig); tab=QWidget(); layout=QVBoxLayout(tab); layout.addWidget(canvas); self.tabs.addTab(tab, 'All Accuracy')
        else:
            if loss_metrics:
                fig = new_fig(); ax = fig.add_subplot(111)
                for m in loss_metrics: self._plot_metric(ax, m)
                ax.set_title('Loss'); ax.grid(grid_on); ax.legend(); fig.tight_layout()
                canvas=FigureCanvas(fig); tab=QWidget(); layout=QVBoxLayout(tab); layout.addWidget(canvas); self.tabs.addTab(tab, 'Loss')
            if acc_metrics:
                fig = new_fig(); ax = fig.add_subplot(111)
                for m in acc_metrics: self._plot_metric(ax, m)
                ax.set_title('Accuracy'); ax.grid(grid_on); ax.legend(); fig.tight_layout()
                canvas=FigureCanvas(fig); tab=QWidget(); layout=QVBoxLayout(tab); layout.addWidget(canvas); self.tabs.addTab(tab, 'Accuracy')

        # Add navigation toolbar
        first_canvas = self.tabs.widget(0).findChild(FigureCanvas)
        self.toolbar = NavigationToolbar(first_canvas, self)
        self.left_layout.insertWidget(0, self.toolbar)

    def _plot_metric(self, ax, base):
        color = self.color_map.get(base)
        train_col, val_col = f"train_{base}", f"val_{base}"
        if train_col in self.data.columns:
            ax.plot(self.data[train_col], label=train_col, color=color, linestyle='-')
        if val_col in self.data.columns:
            ax.plot(self.data[val_col], label=val_col, color=color, linestyle='--')
        if base in self.data.columns and train_col not in self.data.columns and val_col not in self.data.columns:
            ax.plot(self.data[base], label=base, color=color, linestyle='-')
        ax.set_xlabel('')

    def save_plot(self):
        base_name = self.edit_filename.text().strip()
        if not base_name:
            QMessageBox.warning(self, "Warning", "Enter a base filename.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Plots")
        if not directory:
            return
        root, ext = os.path.splitext(base_name)
        ext = ext if ext else '.png'
        for idx in range(self.tabs.count()):
            title = self.tabs.tabText(idx)
            canvas = self.tabs.widget(idx).findChild(FigureCanvas)
            # Ensure uniform size on save
            canvas.figure.set_size_inches(self.fig_size)
            save_path = os.path.join(directory, f"{title}_{root}{ext}")
            canvas.figure.savefig(save_path)
        QMessageBox.information(self, "Saved", f"Plots saved to {directory}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LearningCurvePlotter()
    win.show()
    sys.exit(app.exec_())
