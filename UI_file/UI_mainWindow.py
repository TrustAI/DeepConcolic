# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1207, 629)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/deepconcolic-logo2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mainWindow.setWindowIcon(icon)
        mainWindow.setStyleSheet("background-color:rgb(243, 253, 255)")
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Avenir Next")
        font.setPointSize(36)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setStyleSheet("color: rgb(42, 0, 255);\n"
"background-color: transparent;\n"
"")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 4)
        self.pushButton_TestRNN = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_TestRNN.setStyleSheet("QPushButton:focus{outline: none;}\n"
"QPushButton\n"
"{\n"
"    color: rgb(129, 255, 0);\n"
"    border:hide;\n"
"    border-radius:2px;\n"
"    padding:0px 0px;\n"
"    background-color: rgb(51,51,51);\n"
"}\n"
"QPushButton:!enabled\n"
"{\n"
"    color: gray;\n"
"    border:hide;\n"
"    border-radius:2px;\n"
"    padding:0px 0px;\n"
"    background-color: rgb(100,100,100);\n"
"}\n"
"QPushButton:pressed\n"
"{\n"
"    background-color:rgb(200 , 200 , 200);\n"
"    padding-left:2px;\n"
"    padding-top:2px;\n"
"}")
        self.pushButton_TestRNN.setObjectName("pushButton_TestRNN")
        self.gridLayout.addWidget(self.pushButton_TestRNN, 3, 1, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 4)
        self.pushButton_DeepConcolic = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_DeepConcolic.setStyleSheet("QPushButton:focus{outline: none;}\n"
"QPushButton\n"
"{\n"
"    color: rgb(255, 116, 219);\n"
"    border:hide;\n"
"    border-radius:2px;\n"
"    padding:0px 0px;\n"
"    background-color: rgb(51,51,51);\n"
"}\n"
"QPushButton:!enabled\n"
"{\n"
"    color: gray;\n"
"    border:hide;\n"
"    border-radius:2px;\n"
"    padding:0px 0px;\n"
"    background-color: rgb(100,100,100);\n"
"}\n"
"QPushButton:pressed\n"
"{\n"
"    background-color:rgb(200 , 200 , 200);\n"
"    padding-left:2px;\n"
"    padding-top:2px;\n"
"}")
        self.pushButton_DeepConcolic.setObjectName("pushButton_DeepConcolic")
        self.gridLayout.addWidget(self.pushButton_DeepConcolic, 3, 0, 1, 1)
        self.pushButton_GUAP = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_GUAP.setStyleSheet("QPushButton:focus{outline: none;}\n"
"QPushButton\n"
"{\n"
"    color: rgb(0, 255, 255);\n"
"    border:hide;\n"
"    border-radius:2px;\n"
"    padding:0px 0px;\n"
"    background-color: rgb(51,51,51);\n"
"}\n"
"QPushButton:!enabled\n"
"{\n"
"    color: gray;\n"
"    border:hide;\n"
"    border-radius:2px;\n"
"    padding:0px 0px;\n"
"    background-color: rgb(100,100,100);\n"
"}\n"
"QPushButton:pressed\n"
"{\n"
"    background-color:rgb(200 , 200 , 200);\n"
"    padding-left:2px;\n"
"    padding-top:2px;\n"
"}")
        self.pushButton_GUAP.setObjectName("pushButton_GUAP")
        self.gridLayout.addWidget(self.pushButton_GUAP, 3, 3, 1, 1)
        self.pushButton_EKiML = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_EKiML.setStyleSheet("QPushButton:focus{outline: none;}\n"
"QPushButton\n"
"{\n"
"    color: rgb(49, 131, 255);\n"
"    border:hide;\n"
"    border-radius:2px;\n"
"    padding:0px 0px;\n"
"    background-color: rgb(51,51,51);\n"
"}\n"
"QPushButton:!enabled\n"
"{\n"
"    color: gray;\n"
"    border:hide;\n"
"    border-radius:2px;\n"
"    padding:0px 0px;\n"
"    background-color: rgb(100,100,100);\n"
"}\n"
"QPushButton:pressed\n"
"{\n"
"    background-color:rgb(200 , 200 , 200);\n"
"    padding-left:2px;\n"
"    padding-top:2px;\n"
"}")
        self.pushButton_EKiML.setObjectName("pushButton_EKiML")
        self.gridLayout.addWidget(self.pushButton_EKiML, 3, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setItalic(True)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 4)
        self.pushButton_DeepConcolic.raise_()
        self.pushButton_TestRNN.raise_()
        self.pushButton_EKiML.raise_()
        self.pushButton_GUAP.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.graphicsView.raise_()
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1207, 22))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "DeepConcolic"))
        self.label.setText(_translate("mainWindow", "DeepConcolic (Testing for Deep Neural Networks)"))
        self.pushButton_TestRNN.setText(_translate("mainWindow", "TestRNN"))
        self.pushButton_DeepConcolic.setText(_translate("mainWindow", "DeepConcolic"))
        self.pushButton_GUAP.setText(_translate("mainWindow", "GUAP"))
        self.pushButton_EKiML.setText(_translate("mainWindow", "EKiML"))
        self.label_2.setText(_translate("mainWindow", "This repository includes a few software packages, all of which are dedicated for the analysis of deep neural netowrks (or tree ensembles) over its safety and/or security properties.\n"
"\n"
"DeepConcolic, a coverage-guided testing tool for convolutional neural networks. Now, it includes a major upgrade based on Bayesian Network based Abstraction.\n"
"testRNN, a coverage-guided testing tool for Long short-term memory models (LSTMs). LSTMs are a major class of recurrent neural networks.\n"
"EKiML, a tool for backdoor embedding and detection for tree ensembles.\n"
"GUAP: a generalised universal adversarial perturbation. It generates universersal adversarial perburbation that may be applied to many inputs at the same time."))

