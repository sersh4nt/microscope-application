# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designer.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from libs.canvas import Canvas


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        MainWindow.setBaseSize(QtCore.QSize(800, 600))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setLocale(QtCore.QLocale(QtCore.QLocale.Russian, QtCore.QLocale.Russia))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setContentsMargins(10, 10, 10, 0)
        self.horizontalLayout_3.setSpacing(10)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.componentList = QtWidgets.QListWidget(self.centralwidget)
        self.componentList.setObjectName("typeList")
        self.componentList.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.verticalLayout.addWidget(self.componentList)
        self.recordList = QtWidgets.QListWidget(self.centralwidget)
        self.recordList.setObjectName("componentList")
        self.recordList.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.verticalLayout.addWidget(self.recordList)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.canvas = Canvas()
        self.canvas.setGeometry(QtCore.QRect(0, 0, 432, 547))
        self.canvas.setObjectName("canvas")
        self.scrollArea.setWidget(self.canvas)
        self.horizontalLayout_3.addWidget(self.scrollArea)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.rectangleList = QtWidgets.QListWidget(self.centralwidget)
        self.rectangleList.setObjectName("objectList")
        self.verticalLayout_2.addWidget(self.rectangleList)
        self.shotButton = QtWidgets.QPushButton(self.centralwidget)
        self.shotButton.setObjectName("shotButton")
        self.verticalLayout_2.addWidget(self.shotButton)
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setObjectName("saveButton")
        self.verticalLayout_2.addWidget(self.saveButton)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_3.setStretch(0, 3)
        self.horizontalLayout_3.setStretch(1, 8)
        self.horizontalLayout_3.setStretch(2, 3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.fileMenu = QtWidgets.QMenu(self.menubar)
        self.fileMenu.setObjectName("fileMenu")
        self.redactorMenu = QtWidgets.QMenu(self.menubar)
        self.redactorMenu.setObjectName("redactorMenu")
        self.modeMenu = QtWidgets.QMenu(self.redactorMenu)
        self.modeMenu.setObjectName("modeMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.saveAll = QtWidgets.QAction(MainWindow)
        self.saveAll.setObjectName("saveAll")
        self.modeSelect = QtWidgets.QAction(MainWindow)
        self.modeSelect.setCheckable(True)
        self.modeSelect.setObjectName("modeSelect")
        self.modeEdit = QtWidgets.QAction(MainWindow)
        self.modeEdit.setCheckable(True)
        self.modeEdit.setObjectName("modeEdit")
        self.fileMenu.addAction(self.saveAll)
        self.modeMenu.addAction(self.modeSelect)
        self.modeMenu.addAction(self.modeEdit)
        self.redactorMenu.addAction(self.modeMenu.menuAction())
        self.menubar.addAction(self.fileMenu.menuAction())
        self.menubar.addAction(self.redactorMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Редактирование базы компонентов"))
        self.shotButton.setText(_translate("MainWindow", "Сделать снимок"))
        self.saveButton.setText(_translate("MainWindow", "Сохранить"))
        self.fileMenu.setTitle(_translate("MainWindow", "Файл"))
        self.redactorMenu.setTitle(_translate("MainWindow", "Редактор"))
        self.modeMenu.setTitle(_translate("MainWindow", "Режим"))
        self.saveAll.setText(_translate("MainWindow", "Сохранить изменения"))
        self.modeSelect.setText(_translate("MainWindow", "Выделение"))
        self.modeEdit.setText(_translate("MainWindow", "Редактирование"))
