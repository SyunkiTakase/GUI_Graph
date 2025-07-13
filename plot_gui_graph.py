import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog,
    QLabel, QLineEdit, QMessageBox, QCheckBox, QTabWidget
)
from PyQt5.QtCore import Qt, QFileSystemWatcher
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
from matplotlib import rcParams

class LearningCurvePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("学習曲線プロットツール")
        self.resize(1200, 700)
        self.logs = []
        self.metrics = []
        self.color_map = {}
        self.line_styles = ['-', '--', ':', '-.']
        self.fig_size = (8, 6)

        self.watcher = QFileSystemWatcher()
        self.watcher.fileChanged.connect(self.on_file_changed)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        main_layout.addWidget(left_widget, stretch=3)
        self.tabs = QTabWidget()
        self.left_layout.addWidget(self.tabs)
        self.toolbar = None

        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        main_layout.addWidget(ctrl, stretch=1)

        # ログ読み込みボタン
        btn_load = QPushButton("ログ読み込み")
        btn_load.clicked.connect(self.load_logs)
        ctrl_layout.addWidget(btn_load)

        # 各種オプション
        self.cb_sep = QCheckBox("メトリクスごとにタブ分割")
        ctrl_layout.addWidget(self.cb_sep)
        self.cb_grid = QCheckBox("グリッド表示")
        ctrl_layout.addWidget(self.cb_grid)
        self.cb_side = QCheckBox("横並びレイアウト")
        ctrl_layout.addWidget(self.cb_side)
        self.cb_connect = QCheckBox("ログ接続表示 (2つ以上)")
        ctrl_layout.addWidget(self.cb_connect)

        # メトリクス選択
        ctrl_layout.addWidget(QLabel("プロットするメトリクスを選択："))
        btn_all = QPushButton("全てプロット")
        btn_all.clicked.connect(self.select_all_metrics)
        ctrl_layout.addWidget(btn_all)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.NoSelection)
        ctrl_layout.addWidget(self.list_widget)

        # プロット・保存
        btn_plot = QPushButton("選択プロット")
        btn_plot.clicked.connect(self.plot_selected)
        ctrl_layout.addWidget(btn_plot)

        ctrl_layout.addWidget(QLabel("保存ファイル名 (例: plot.png)："))
        self.edit_filename = QLineEdit("plot.png")
        ctrl_layout.addWidget(self.edit_filename)
        btn_save = QPushButton("プロット保存")
        btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(btn_save)
        ctrl_layout.addStretch()

    def load_logs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "CSVログファイルを開く", "", "CSV ファイル (*.csv)"
        )
        if not paths:
            return
        self.logs.clear()
        self.watcher.removePaths(self.watcher.files())
        for p in paths:
            try:
                df = pd.read_csv(p)
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"{os.path.basename(p)} の読み込みに失敗しました：\n{e}")
                continue
            name = os.path.splitext(os.path.basename(p))[0]
            self.logs.append({'path': p, 'df': df, 'name': name})
            self.watcher.addPath(p)

        # メトリクス検出
        bases = []
        for log in self.logs:
            for col in log['df'].columns:
                cl = col.lower()
                if cl.startswith('train_'):
                    bases.append(col[6:])
                elif cl.startswith('val_'):
                    bases.append(col[4:])
                elif cl != 'epoch':
                    bases.append(col)
        self.metrics = sorted(set(bases))

        # カラーマップ設定
        colors = rcParams['axes.prop_cycle'].by_key()['color']
        self.color_map = {m: colors[i % len(colors)] for i, m in enumerate(self.metrics)}

        # メトリクス一覧に反映
        self.list_widget.clear()
        for m in self.metrics:
            item = QListWidgetItem(m)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        QMessageBox.information(
            self, "読み込み完了",
            f"{len(self.logs)} 件のログを読み込みました。\nメトリクス: {', '.join(self.metrics)}"
        )

    def on_file_changed(self, path):
        for log in self.logs:
            if log['path'] == path:
                try:
                    log['df'] = pd.read_csv(path)
                except:
                    pass
        self.plot_selected()

    def select_all_metrics(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def plot_selected(self):
        if not self.logs:
            QMessageBox.warning(self, "警告", "先にログを読み込んでください。")
            return
        selected = [
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        ]
        if not selected:
            QMessageBox.warning(self, "警告", "メトリクスを選択してください。")
            return

        grid = self.cb_grid.isChecked()
        side = self.cb_side.isChecked()
        sep = self.cb_sep.isChecked()
        conn = self.cb_connect.isChecked() and len(self.logs) >= 2

        self.tabs.clear()
        if self.toolbar:
            self.toolbar.setParent(None)

        def new_fig():
            return Figure(figsize=self.fig_size)

        # 横並びレイアウト
        if side:
            fig = new_fig()
            axes = fig.subplots(1, len(selected), squeeze=False)[0]
            for ax, m in zip(axes, selected):
                if conn and not sep:
                    self.plot_combined(ax, m)
                else:
                    if conn and sep:
                        self.plot_combined(ax, m)
                    else:
                        self.draw_metric(ax, m, conn)
                ax.set_title(m)
                ax.grid(grid)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels)
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(canvas)
            self.tabs.addTab(tab, '横並び')

        # メトリクスごとタブ
        elif sep:
            for m in selected:
                fig = new_fig()
                ax = fig.add_subplot(111)
                if conn:
                    self.plot_combined(ax, m)
                else:
                    self.draw_metric(ax, m, conn)
                ax.set_title(m)
                ax.grid(grid)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels)
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                tab = QWidget()
                layout = QVBoxLayout(tab)
                layout.addWidget(canvas)
                self.tabs.addTab(tab, m)

        # Loss & Accuracy タブ
        else:
            loss = [m for m in selected if 'acc' not in m.lower()]
            acc = [m for m in selected if 'acc' in m.lower()]
            if loss:
                fig = new_fig()
                ax = fig.add_subplot(111)
                for m in loss:
                    if conn:
                        self.plot_combined(ax, m)
                    else:
                        self.draw_metric(ax, m, conn)
                ax.set_title('Loss')
                ax.grid(grid)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels)
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                tab = QWidget()
                layout = QVBoxLayout(tab)
                layout.addWidget(canvas)
                self.tabs.addTab(tab, 'Loss')
            if acc:
                fig = new_fig()
                ax = fig.add_subplot(111)
                for m in acc:
                    if conn:
                        self.plot_combined(ax, m)
                    else:
                        self.draw_metric(ax, m, conn)
                ax.set_title('Accuracy')
                ax.grid(grid)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels)
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                tab = QWidget()
                layout = QVBoxLayout(tab)
                layout.addWidget(canvas)
                self.tabs.addTab(tab, 'Accuracy')

        # ナビゲーションツールバー追加
        first_canvas = self.tabs.widget(0).findChild(FigureCanvas)
        self.toolbar = NavigationToolbar(first_canvas, self)
        self.left_layout.insertWidget(0, self.toolbar)

    def draw_metric(self, ax, metric, conn):
        color = self.color_map.get(metric)
        for idx, log in enumerate(self.logs):
            df = log['df']
            style = self.line_styles[idx % len(self.line_styles)]
            tcol = f"train_{metric}"
            vcol = f"val_{metric}"
            if tcol in df.columns:
                ax.plot(df[tcol].values, label=f"{log['name']}:{tcol}", color=color, linestyle=style)
            if vcol in df.columns:
                ax.plot(df[vcol].values, label=f"{log['name']}:{vcol}", color=color, linestyle='--')
            if metric in df.columns and tcol not in df.columns and vcol not in df.columns:
                ax.plot(df[metric].values, label=f"{log['name']}:{metric}", color=color, linestyle='-')
        ax.set_xlabel('Epoch')

    def plot_combined(self, ax, metric):
        color = self.color_map.get(metric)
        train_series, val_series, base_series = [], [], []
        for log in self.logs:
            df = log['df']
            tcol, vcol = f"train_{metric}", f"val_{metric}"
            if tcol in df.columns:
                train_series.append(df[tcol].values)
            if vcol in df.columns:
                val_series.append(df[vcol].values)
            if metric in df.columns and tcol not in df.columns and vcol not in df.columns:
                base_series.append(df[metric].values)
        if train_series:
            y = np.concatenate(train_series); x = np.arange(len(y))
            ax.plot(x, y, label=f"train_{metric}", color=color, linestyle='-')
        if val_series:
            y = np.concatenate(val_series); x = np.arange(len(y))
            ax.plot(x, y, label=f"val_{metric}", color=color, linestyle='--')
        if not train_series and not val_series and base_series:
            y = np.concatenate(base_series); x = np.arange(len(y))
            ax.plot(x, y, label=metric, color=color, linestyle='-')
        ax.set_xlabel('Iterations')

    def save_plot(self):
        name = self.edit_filename.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "ファイル名を入力してください。")
            return
        directory = QFileDialog.getExistingDirectory(self, "プロット保存先を選択")
        if not directory:
            return
        base, ext = os.path.splitext(name)
        ext = ext or '.png'
        for i in range(self.tabs.count()):
            title = self.tabs.tabText(i)
            canvas = self.tabs.widget(i).findChild(FigureCanvas)
            canvas.figure.set_size_inches(self.fig_size)
            canvas.figure.savefig(os.path.join(directory, f"{title}_{base}{ext}"))
        QMessageBox.information(self, "保存完了", f"プロットを保存しました：\n{directory}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LearningCurvePlotter()
    win.show()
    sys.exit(app.exec_())
