# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitled.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1061, 712)
        font = QFont()
        font.setFamily(u"Times New Roman")
        font.setPointSize(14)
        MainWindow.setFont(font)
        MainWindow.setLocale(QLocale(QLocale.Russian, QLocale.Russia))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(20, 20, 20, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.microscopeView = QLabel(self.centralwidget)
        self.microscopeView.setObjectName(u"microscopeView")
        self.microscopeView.setFrameShape(QFrame.Box)

        self.horizontalLayout.addWidget(self.microscopeView)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.listView = QLabel(self.centralwidget)
        self.listView.setObjectName(u"listView")
        self.listView.setFrameShape(QFrame.Box)

        self.verticalLayout_2.addWidget(self.listView)

        self.databaseComponentView = QLabel(self.centralwidget)
        self.databaseComponentView.setObjectName(u"databaseComponentView")
        self.databaseComponentView.setFrameShape(QFrame.Box)

        self.verticalLayout_2.addWidget(self.databaseComponentView)

        self.databaseEditButton = QPushButton(self.centralwidget)
        self.databaseEditButton.setObjectName(u"databaseEditButton")

        self.verticalLayout_2.addWidget(self.databaseEditButton)

        self.operatorDataEditBitton = QPushButton(self.centralwidget)
        self.operatorDataEditBitton.setObjectName(u"operatorDataEditBitton")

        self.verticalLayout_2.addWidget(self.operatorDataEditBitton)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.horizontalLayout.setStretch(0, 8)
        self.horizontalLayout.setStretch(1, 5)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.currentComponentMarking = QLabel(self.centralwidget)
        self.currentComponentMarking.setObjectName(u"currentComponentMarking")
        self.currentComponentMarking.setFrameShape(QFrame.Box)

        self.verticalLayout.addWidget(self.currentComponentMarking)

        self.verticalLayout.setStretch(0, 100)
        self.verticalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1061, 27))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u0418\u043d\u0441\u043f\u0435\u043a\u0446\u0438\u043e\u043d\u043d\u044b\u0439 \u0441\u0442\u0435\u043d\u0434 \u0432\u0445\u043e\u0434\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044f", None))
        self.microscopeView.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.listView.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.databaseComponentView.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.databaseEditButton.setText(QCoreApplication.translate("MainWindow", u"\u0418\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435 \u0431\u0430\u0437\u044b \u0434\u0430\u043d\u043d\u044b\u0445", None))
        self.operatorDataEditBitton.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u043d\u0435\u0441\u0435\u043d\u0438\u0435 \u0434\u0430\u043d\u043d\u044b\u0445 \u043e\u043f\u0435\u0440\u0430\u0442\u043e\u0440\u0430", None))
        self.currentComponentMarking.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
    # retranslateUi

