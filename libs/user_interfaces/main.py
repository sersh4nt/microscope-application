# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from libs.camera import CameraWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1061, 712)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setLocale(QtCore.QLocale(QtCore.QLocale.Russian, QtCore.QLocale.Russia))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 0)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.microscopeView = CameraWidget(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.microscopeView.sizePolicy().hasHeightForWidth())
        self.microscopeView.setSizePolicy(sizePolicy)
        self.microscopeView.setFrameShape(QtWidgets.QFrame.Box)
        self.microscopeView.setObjectName("microscopeView")
        self.horizontalLayout.addWidget(self.microscopeView)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.listView = QtWidgets.QListWidget(self.centralwidget)
        self.listView.setObjectName("listView")
        self.verticalLayout_2.addWidget(self.listView)
        self.databaseComponentView = QtWidgets.QLabel(self.centralwidget)
        self.databaseComponentView.setFrameShape(QtWidgets.QFrame.Box)
        self.databaseComponentView.setObjectName("databaseComponentView")
        self.verticalLayout_2.addWidget(self.databaseComponentView)
        self.databaseEditButton = QtWidgets.QPushButton(self.centralwidget)
        self.databaseEditButton.setObjectName("databaseEditButton")
        self.verticalLayout_2.addWidget(self.databaseEditButton)
        self.operatorDataEditButton = QtWidgets.QPushButton(self.centralwidget)
        self.operatorDataEditButton.setObjectName("operatorDataEditButton")
        self.verticalLayout_2.addWidget(self.operatorDataEditButton)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout.setStretch(0, 8)
        self.horizontalLayout.setStretch(1, 3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.currentComponentMarking = QtWidgets.QLabel(self.centralwidget)
        self.currentComponentMarking.setFrameShape(QtWidgets.QFrame.Box)
        self.currentComponentMarking.setObjectName("currentComponentMarking")
        self.verticalLayout.addWidget(self.currentComponentMarking)
        self.verticalLayout.setStretch(0, 100)
        self.verticalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1061, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Инспекционный стенд входного контроля"))
        self.microscopeView.setText(_translate("MainWindow", "TextLabel"))
        self.databaseComponentView.setText(_translate("MainWindow", "TextLabel"))
        self.databaseEditButton.setText(_translate("MainWindow", "Изменение базы данных"))
        self.operatorDataEditButton.setText(_translate("MainWindow", "Занесение данных оператора"))
        self.currentComponentMarking.setText(_translate("MainWindow", "TextLabel"))
        self.action.setText(_translate("MainWindow", "Подключить"))
        self.action_2.setText(_translate("MainWindow", "Остановить"))
