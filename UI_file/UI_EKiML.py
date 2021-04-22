# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_EKiML.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_EKiML(object):
    def setupUi(self, EKiML):
        EKiML.setObjectName("EKiML")
        EKiML.resize(1105, 726)
        self.centralwidget = QtWidgets.QWidget(EKiML)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setStyleSheet("QPushButton:focus{outline: none;}\n"
"QPushButton\n"
"{\n"
"    color: white;\n"
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
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.pushButton, 4, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_Model = QtWidgets.QLabel(self.centralwidget)
        self.label_Model.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_Model.setObjectName("label_Model")
        self.gridLayout.addWidget(self.label_Model, 0, 3, 1, 1)
        self.label_SaveModel = QtWidgets.QLabel(self.centralwidget)
        self.label_SaveModel.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_SaveModel.setObjectName("label_SaveModel")
        self.gridLayout.addWidget(self.label_SaveModel, 0, 4, 1, 1)
        self.label_output = QtWidgets.QLabel(self.centralwidget)
        self.label_output.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_output.setObjectName("label_output")
        self.gridLayout.addWidget(self.label_output, 0, 8, 1, 1)
        self.label_Mode = QtWidgets.QLabel(self.centralwidget)
        self.label_Mode.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_Mode.setObjectName("label_Mode")
        self.gridLayout.addWidget(self.label_Mode, 0, 1, 1, 1)
        self.label_Pruning = QtWidgets.QLabel(self.centralwidget)
        self.label_Pruning.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_Pruning.setObjectName("label_Pruning")
        self.gridLayout.addWidget(self.label_Pruning, 0, 5, 1, 1)
        self.label_Dataset = QtWidgets.QLabel(self.centralwidget)
        self.label_Dataset.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_Dataset.setObjectName("label_Dataset")
        self.gridLayout.addWidget(self.label_Dataset, 0, 0, 1, 1)
        self.label_Embedding_Method = QtWidgets.QLabel(self.centralwidget)
        self.label_Embedding_Method.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_Embedding_Method.setObjectName("label_Embedding_Method")
        self.gridLayout.addWidget(self.label_Embedding_Method, 0, 2, 1, 1)
        self.comboBox_Dataset = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Dataset.setObjectName("comboBox_Dataset")
        self.gridLayout.addWidget(self.comboBox_Dataset, 1, 0, 1, 1)
        self.comboBox_Mode = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Mode.setObjectName("comboBox_Mode")
        self.gridLayout.addWidget(self.comboBox_Mode, 1, 1, 1, 1)
        self.comboBox_Embedding_Method = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Embedding_Method.setObjectName("comboBox_Embedding_Method")
        self.gridLayout.addWidget(self.comboBox_Embedding_Method, 1, 2, 1, 1)
        self.comboBox_Model = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Model.setObjectName("comboBox_Model")
        self.gridLayout.addWidget(self.comboBox_Model, 1, 3, 1, 1)
        self.comboBox_SaveModel = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_SaveModel.setObjectName("comboBox_SaveModel")
        self.gridLayout.addWidget(self.comboBox_SaveModel, 1, 4, 1, 1)
        self.comboBox_Pruning = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Pruning.setObjectName("comboBox_Pruning")
        self.gridLayout.addWidget(self.comboBox_Pruning, 1, 5, 1, 1)
        self.lineEdit_output = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_output.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_output.setObjectName("lineEdit_output")
        self.gridLayout.addWidget(self.lineEdit_output, 1, 8, 1, 1)
        self.label_workdir = QtWidgets.QLabel(self.centralwidget)
        self.label_workdir.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_workdir.setObjectName("label_workdir")
        self.gridLayout.addWidget(self.label_workdir, 0, 6, 1, 1)
        self.lineEdit_workdir = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_workdir.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_workdir.setObjectName("lineEdit_workdir")
        self.gridLayout.addWidget(self.lineEdit_workdir, 1, 6, 1, 1)
        self.label_datadir = QtWidgets.QLabel(self.centralwidget)
        self.label_datadir.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_datadir.setObjectName("label_datadir")
        self.gridLayout.addWidget(self.label_datadir, 0, 7, 1, 1)
        self.lineEdit_datadir = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_datadir.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_datadir.setObjectName("lineEdit_datadir")
        self.gridLayout.addWidget(self.lineEdit_datadir, 1, 7, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Avenir Next")
        font.setPointSize(36)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color:rgb(49, 131, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.pushButton_back = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_back.setStyleSheet("QPushButton:focus{outline: none;}\n"
"QPushButton\n"
"{\n"
"    color: white;\n"
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
        self.pushButton_back.setObjectName("pushButton_back")
        self.gridLayout_2.addWidget(self.pushButton_back, 6, 0, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout_2.addWidget(self.textEdit, 1, 0, 1, 1)
        EKiML.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(EKiML)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1105, 22))
        self.menubar.setObjectName("menubar")
        EKiML.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(EKiML)
        self.statusbar.setObjectName("statusbar")
        EKiML.setStatusBar(self.statusbar)

        self.retranslateUi(EKiML)
        QtCore.QMetaObject.connectSlotsByName(EKiML)

    def retranslateUi(self, EKiML):
        _translate = QtCore.QCoreApplication.translate
        EKiML.setWindowTitle(_translate("EKiML", "EKiML"))
        self.pushButton.setText(_translate("EKiML", "Confirm"))
        self.label_Model.setText(_translate("EKiML", "Model"))
        self.label_SaveModel.setText(_translate("EKiML", "SaveModel"))
        self.label_output.setText(_translate("EKiML", "output"))
        self.label_Mode.setText(_translate("EKiML", "Mode"))
        self.label_Pruning.setText(_translate("EKiML", "Pruning"))
        self.label_Dataset.setText(_translate("EKiML", "Dataset"))
        self.label_Embedding_Method.setText(_translate("EKiML", "Embedding_Method"))
        self.label_workdir.setText(_translate("EKiML", "workdir"))
        self.label_datadir.setText(_translate("EKiML", "datadir"))
        self.label.setText(_translate("EKiML", "EKiML"))
        self.pushButton_back.setText(_translate("EKiML", "Back"))

