## Created by: Myeongjin Lee
## Date: 2025-03-19
## Version: 1.0
## Description: ROBROS Imitation Learning dataset editor

import sys
import h5py
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QSlider, QDockWidget, QTabWidget, QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QSizePolicy, QHBoxLayout, QProgressDialog, QInputDialog, QCheckBox, QTreeWidget, QTreeWidgetItem, QDialog, QDialogButtonBox, QSplitter
from PyQt6.QtGui import QImage, QPixmap, QIcon, QColor
from PyQt6.QtCore import QTimer, Qt, QSize
import pyqtgraph as pg
import cv2
import os
import pyqtgraph.opengl as gl

class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        source_dir = os.path.dirname(os.path.abspath(__file__))
        resource_dir = os.path.join(source_dir, "resource")
        
        current_style = QApplication.style().objectName().lower()
        
        if current_style == "fusion":
            self.is_dark_theme = True
            icon = QIcon(os.path.join(resource_dir, "logo_white_square.png"))
            self.logo = QPixmap(os.path.join(resource_dir, "logo_white_long.png"))
        else:
            self.is_dark_theme = False
            icon = QIcon(os.path.join(resource_dir, "logo_black_square.png"))
            self.logo = QPixmap(os.path.join(resource_dir, "logo_black_long.png"))
            
        self.setWindowTitle("ROBROS IL Dataset Editor")
        self.setGeometry(100, 100, 1600, 900)
        self.setWindowIcon(icon)

        self.current_frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.hdf5_files = []
        self.should_plot_reward = False
        self.xpos_data = None
        self.gl_widget = None
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)

        self.logo_label = QLabel()
        self.logo_label.setMaximumHeight(100)
        scaled_logo = self.logo.scaled(self.logo_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.logo_label.setPixmap(scaled_logo)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.logo_label)

        self.dock = QDockWidget("Loaded HDF5 Files", self)
        self.dock.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        dock_widget_content = QWidget()
        dock_layout = QVBoxLayout(dock_widget_content)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(0)

        self.file_table_widget = QTableWidget()
        self.file_table_widget.setColumnCount(2)
        self.file_table_widget.setHorizontalHeaderLabels(["File Name", "Data Count"])
        self.file_table_widget.cellClicked.connect(self.load_selected_hdf5)
        
        header = self.file_table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        dock_layout.addWidget(self.file_table_widget)

        self.load_button = QPushButton("Load HDF5 File")
        self.load_button.clicked.connect(self.load_hdf5)
        dock_layout.addWidget(self.load_button)

        self.dock.setWidget(dock_widget_content)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |  
                              QDockWidget.DockWidgetFeature.DockWidgetFloatable | 
                              QDockWidget.DockWidgetFeature.DockWidgetClosable | 
                              QDockWidget.DockWidgetFeature.DockWidgetVerticalTitleBar)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        load_hdf5_action = file_menu.addAction("Load HDF5 File")
        load_hdf5_action.triggered.connect(self.load_hdf5)
        save_hdf5_action = file_menu.addAction("Save HDF5 File")
        save_hdf5_action.triggered.connect(self.save_hdf5)
        view_menu = menu_bar.addMenu("View")
        toggle_dock_action = view_menu.addAction("Toggle File List")
        toggle_dock_action.triggered.connect(self.toggle_dock_visibility)

        vertical_layout = QVBoxLayout()
        
        ## File Name
        self.file_name = QLabel("NO FILE SELECTED")
        self.file_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_name.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.file_name.setMaximumHeight(30)
        vertical_layout.addWidget(self.file_name)
        
        ## Image Tab
        main_horizontal_layout = QHBoxLayout()
        
        self.tab_widget = QTabWidget()
        
        no_image_tab = QWidget()
        no_image_layout = QVBoxLayout(no_image_tab)
        no_image_label = QLabel("NO IMAGE")
        no_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_image_layout.addWidget(no_image_label)
        
        self.tab_widget.addTab(no_image_tab, "NO IMAGE")
        
        main_horizontal_layout.addWidget(self.tab_widget)
        
        ## 3D View
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=1400)
        self.gl_widget.setMinimumSize(600, 400)
        main_horizontal_layout.addWidget(self.gl_widget)

        vertical_layout.addLayout(main_horizontal_layout)

        ## Image Info
        self.image_info_label = QLabel("W: 0 / H: 0")
        self.image_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.image_info_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        self.image_info_label.setMaximumHeight(30)
        vertical_layout.addWidget(self.image_info_label)
        
        ## Media Controls
        media_controls_layout = QHBoxLayout()
        self.fb_button = QPushButton("<<")
        self.fb_button.clicked.connect(self.toggle_one_frame_backward)
        self.fb_button.setMaximumHeight(30)
        media_controls_layout.addWidget(self.fb_button)
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setMaximumHeight(30)
        media_controls_layout.addWidget(self.play_button)
        
        self.ff_button = QPushButton(">>")
        self.ff_button.clicked.connect(self.toggle_one_frame_forward)
        self.ff_button.setMaximumHeight(30)
        media_controls_layout.addWidget(self.ff_button)
        vertical_layout.addLayout(media_controls_layout)
        
        ## Tick Label
        self.tick_label = QLabel("0 / 0")
        self.tick_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tick_label.setStyleSheet("font-size: 12px; font-weight: bold; border: 2px solid black; background-color: rgb(128, 128, 128);")
        self.tick_label.setMaximumHeight(30)
        vertical_layout.addWidget(self.tick_label)
        
        ## Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_widget.getPlotItem().showGrid(x=True, y=True)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMaximumHeight(50)
        vertical_layout.addWidget(self.plot_widget)

        self.frame_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.frame_line)
        
        ## Tick Control Layout
        self.tick_control_layout = QVBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)  
        self.slider.valueChanged.connect(self.slider_changed)
        self.tick_control_layout.addWidget(self.slider)
        vertical_layout.addLayout(self.tick_control_layout)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_click)
        
        self.layout.addLayout(vertical_layout)

    def load_hdf5(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open HDF5 Files", "", "HDF5 Files (*.hdf5)")
        
        if not file_names:
            return

        progress_dialog = QProgressDialog("Loading HDF5 Files...", "Cancel", 0, len(file_names), self)
        progress_dialog.setWindowTitle("Loading")
        progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()

        for i, file_name in enumerate(file_names):
            if file_name not in self.hdf5_files:
                QApplication.processEvents()  
                self.load_file(file_name)
                self.hdf5_files.append(file_name)
                
                self.add_file_to_table(file_name)
                
                progress_dialog.setValue(i + 1)
                
                if progress_dialog.wasCanceled():
                    break

        progress_dialog.close()
    
    def save_hdf5(self):
        if not hasattr(self, 'hdf5_file'):
            QMessageBox.warning(self, "Warning", "No HDF5 file loaded.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Save HDF5 File")
        dialog.setGeometry(100, 100, 500, 500)
        dialog.setLayout(QVBoxLayout())

        tree_widget = QTreeWidget(dialog)
        tree_widget.setHeaderLabels(["Dataset/Group", "Select"])
        dialog.layout().addWidget(tree_widget)

        header = tree_widget.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        def add_items(parent, hdf5_group, path=""):
            for key in hdf5_group.keys():
                item = QTreeWidgetItem(parent)
                item.setText(0, key)
                full_path = f"{path}/{key}" if path else key
                item.setData(0, Qt.ItemDataRole.UserRole, full_path)
                if isinstance(hdf5_group[key], h5py.Group):
                    add_items(item, hdf5_group[key], full_path)
                else:
                    item.setCheckState(1, Qt.CheckState.Checked)
                font_metrics = tree_widget.fontMetrics()
                row_height = font_metrics.height() + 10
                item.setSizeHint(0, QSize(item.sizeHint(0).width(), row_height))

        root_item = QTreeWidgetItem(tree_widget)
        root_item.setText(0, "Root")
        add_items(root_item, self.hdf5_file)
        tree_widget.expandAll()

        save_to_same_file_checkbox = QCheckBox("Save to the same file", dialog)
        save_to_same_file_checkbox.setChecked(True)
        dialog.layout().addWidget(save_to_same_file_checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
        dialog.layout().addWidget(button_box)

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_datasets = []

            def collect_selected_items(item):
                if item.checkState(1) == Qt.CheckState.Checked and not item.childCount():
                    selected_datasets.append(item.data(0, Qt.ItemDataRole.UserRole))
                for i in range(item.childCount()):
                    collect_selected_items(item.child(i))

            collect_selected_items(root_item)

            save_to_same_file = save_to_same_file_checkbox.isChecked()

            if not selected_datasets:
                QMessageBox.warning(self, "Warning", "No datasets selected to save.")
                return

            if save_to_same_file:
                file_name = self.file_name.text()
            else:
                file_name, _ = QFileDialog.getSaveFileName(self, "Save HDF5 File", "", "HDF5 Files (*.hdf5)")
                if not file_name:
                    return

            with h5py.File(file_name, 'w') as hdf5_file:
                for dataset_path in selected_datasets:
                    if dataset_path in self.hdf5_file:
                        data = self.hdf5_file[dataset_path][:]
                        hdf5_file.create_dataset(dataset_path, data=data)
                    else:
                        QMessageBox.warning(self, "Warning", f"Dataset '{dataset_path}' does not exist in the file.")

            QMessageBox.information(self, "Success", f"Data saved to {file_name}.")

    def add_file_to_table(self, file_name):
        row_position = self.file_table_widget.rowCount()
        self.file_table_widget.insertRow(row_position)
        
        file_item = QTableWidgetItem(os.path.basename(file_name))
        file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.file_table_widget.setItem(row_position, 0, file_item)
        
        data_count = self.get_data_count(file_name)
        data_count_item = QTableWidgetItem(str(data_count))
        data_count_item.setFlags(data_count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.file_table_widget.setItem(row_position, 1, data_count_item)

    def get_data_count(self, file_name):
        with h5py.File(file_name, 'r') as hdf5_file:
            total_data_length = sum(len(hdf5_file['observations/images'][key]) for key in hdf5_file['observations/images'])
            num_images = len(hdf5_file['observations/images'])
            return total_data_length // num_images

    def load_file(self, file_name):
        self.tab_widget.clear()

        self.file_name.setText(os.path.basename(file_name))

        self.hdf5_file = h5py.File(file_name, 'r')
        self.images_dict = {} 

        self.should_plot_reward = 'rewards/task' in self.hdf5_file
        if self.should_plot_reward:
            self.reward_data = self.hdf5_file['rewards/task'][:]
        
        self.total_data_length = sum(len(self.hdf5_file['observations/images'][key]) for key in self.hdf5_file['observations/images'])
        
        self.num_images = len(self.hdf5_file['observations/images'])
        
        self.total_frames = self.total_data_length // self.num_images
        
        for key in self.hdf5_file['observations/images']:
            self.images_dict[key] = self.hdf5_file[f'observations/images/{key}']
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            image_label = QLabel("No Image Loaded")
            image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tab_layout.addWidget(image_label)

            self.tab_widget.addTab(tab, key)
            self.tab_widget.currentChanged.connect(self.tab_changed)

        self.current_frame = 0
        self.show_frame(self.current_frame)
        self.slider.setMaximum(self.images_dict[self.tab_widget.tabText(0)].shape[0] - 1)
        self.slider.setValue(0)

        if 'observations/xpos' in self.hdf5_file:
            self.xpos_data = self.hdf5_file['observations/xpos'][:]
            self.update_3d_visualization(self.current_frame)
        else:
            self.xpos_data = None

        self.plot_reward()

    def load_selected_hdf5(self, row, column):
        file_name = self.file_table_widget.item(row, 0).text()
        full_path = next((f for f in self.hdf5_files if os.path.basename(f) == file_name), None)
        if full_path:
            self.load_file(full_path)

    def show_frame(self, frame_index):
        current_tab = self.tab_widget.currentWidget()
        
        if current_tab is None:
            return

        image_label = current_tab.layout().itemAt(0).widget()
        current_dataset = self.images_dict[self.tab_widget.tabText(self.tab_widget.currentIndex())]

        if current_dataset is not None and 0 <= frame_index < current_dataset.shape[0]:
            image_data = current_dataset[frame_index]
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = QImage(image_cv, image_cv.shape[1], image_cv.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            self.slider.setValue(frame_index)
            self.tick_label.setText(f"{frame_index} / {current_dataset.shape[0] - 1}")
            self.frame_line.setPos(frame_index)
            frame_width = scaled_pixmap.width()
            frame_height = scaled_pixmap.height()
            self.image_info_label.setText(f"W: {frame_width} / H: {frame_height}")

            self.update_3d_visualization(frame_index)

    def update_3d_visualization(self, frame_index):
        if self.xpos_data is None:
            return

        self.gl_widget.clear()

        num_objects = self.xpos_data.shape[1] // 3
        
        for i in range(num_objects):
            xyz_data = self.xpos_data[frame_index, i*3:(i+1)*3].reshape(1, 3)
            scatter = gl.GLScatterPlotItem(pos=xyz_data, color=(1, 1, 1, 1), size=10)
            self.gl_widget.addItem(scatter)


    def next_frame(self):
        if not hasattr(self, 'images_dict') or not self.images_dict:
            if self.timer.isActive():
                self.timer.stop()
                self.play_button.setText("Play")
            return

        current_tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())
        
        if current_tab_name not in self.images_dict:
            if self.timer.isActive():
                self.timer.stop()
                self.play_button.setText("Play")
            QMessageBox.warning(self, "Warning", "Current tab does not contain image data.")
            return

        self.current_frame = (self.current_frame + 1) % self.images_dict[current_tab_name].shape[0]
        self.show_frame(self.current_frame)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(10) 
            self.play_button.setText("Pause")

    def slider_changed(self, value):
        self.current_frame = value
        self.show_frame(self.current_frame)

    def tab_changed(self, index):
        if index < 0 or index >= self.tab_widget.count():
            return

        tab_name = self.tab_widget.tabText(index)
        
        if tab_name not in self.images_dict:
            return

        self.current_frame = 0
        self.show_frame(self.current_frame)
        self.slider.setMaximum(self.images_dict[tab_name].shape[0] - 1)
        self.slider.setValue(0)

    def toggle_dock_visibility(self):
        if self.dock.isVisible():
            self.dock.hide()
        else:
            self.dock.show()

    def plot_reward(self):
        self.plot_widget.clear()

        if not hasattr(self, 'reward_data') or not self.should_plot_reward:
            x_data = np.arange(self.total_frames)
            reward_data_flat = np.zeros_like(x_data)
            color = QColor('#FFFFFF')
        else:
            x_data = np.arange(self.reward_data.shape[0])
            reward_data_flat = self.reward_data.flatten()
            color = QColor('#FFFF00')
            color.setAlpha(100)

        self.plot_widget.setEnabled(True)

        label = 'Reward'
        curve = self.plot_widget.plot(x_data, reward_data_flat, pen=pg.mkPen(color=color, width=2), name=label)

        zero_curve = pg.PlotDataItem(x_data, np.zeros_like(x_data), pen=pg.mkPen(None))

        fill = pg.FillBetweenItem(curve, zero_curve, brush=pg.mkBrush(color))
        self.plot_widget.addItem(fill)

        self.plot_widget.addItem(self.frame_line)
        self.plot_widget.enableAutoRange()

        self.plot_widget.setXRange(0, len(x_data) - 1, padding=0)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_play()
        if event.key() == Qt.Key.Key_Right:
            self.toggle_one_frame_forward()
        if event.key() == Qt.Key.Key_Left:
            self.toggle_one_frame_backward()

    def toggle_one_frame_forward(self):
        if not hasattr(self, 'images_dict') or not self.images_dict:
            return

        current_tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())
        if current_tab_name in self.images_dict:
            self.current_frame = (self.current_frame + 1) % self.images_dict[current_tab_name].shape[0]
            self.show_frame(self.current_frame)

    def toggle_one_frame_backward(self):
        if not hasattr(self, 'images_dict') or not self.images_dict:
            return

        current_tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())
        if current_tab_name in self.images_dict:
            self.current_frame = (self.current_frame - 1) % self.images_dict[current_tab_name].shape[0]
            self.show_frame(self.current_frame)

    def on_plot_click(self, event):
        if not hasattr(self, 'images_dict') or not self.images_dict:
            return

        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        x_pos = int(mouse_point.x())

        current_tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())
        if current_tab_name in self.images_dict:
            max_frame = self.images_dict[current_tab_name].shape[0] - 1
            if 0 <= x_pos <= max_frame:
                self.current_frame = x_pos
                self.show_frame(self.current_frame)
                self.slider.setValue(self.current_frame)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec())








