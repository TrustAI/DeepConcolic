# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_testRNN.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_testRNN(object):
    def setupUi(self, testRNN):
        testRNN.setObjectName("testRNN")
        testRNN.resize(1146, 675)
        testRNN.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(testRNN)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_symbolsNum = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_symbolsNum.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_symbolsNum.setObjectName("lineEdit_symbolsNum")
        self.gridLayout.addWidget(self.lineEdit_symbolsNum, 4, 1, 1, 1)
        self.comboBox_mode = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_mode.setStyleSheet("")
        self.comboBox_mode.setObjectName("comboBox_mode")
        self.gridLayout.addWidget(self.comboBox_mode, 4, 2, 1, 1)
        self.lineEdit_outputPath = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_outputPath.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_outputPath.setObjectName("lineEdit_outputPath")
        self.gridLayout.addWidget(self.lineEdit_outputPath, 4, 3, 1, 1)
        self.label_mode = QtWidgets.QLabel(self.centralwidget)
        self.label_mode.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_mode.setObjectName("label_mode")
        self.gridLayout.addWidget(self.label_mode, 3, 2, 1, 1)
        self.lineEdit_seq = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_seq.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_seq.setObjectName("lineEdit_seq")
        self.gridLayout.addWidget(self.lineEdit_seq, 4, 0, 1, 1)
        self.lineEdit_SC = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_SC.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_SC.setObjectName("lineEdit_SC")
        self.gridLayout.addWidget(self.lineEdit_SC, 2, 3, 1, 1)
        self.label_SC = QtWidgets.QLabel(self.centralwidget)
        self.label_SC.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_SC.setObjectName("label_SC")
        self.gridLayout.addWidget(self.label_SC, 1, 3, 1, 1)
        self.label_Mutation_Method = QtWidgets.QLabel(self.centralwidget)
        self.label_Mutation_Method.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_Mutation_Method.setObjectName("label_Mutation_Method")
        self.gridLayout.addWidget(self.label_Mutation_Method, 1, 2, 1, 1)
        self.lineEdit_BC = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_BC.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_BC.setObjectName("lineEdit_BC")
        self.gridLayout.addWidget(self.lineEdit_BC, 2, 4, 1, 1)
        self.lineEdit_casesNum = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_casesNum.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_casesNum.setObjectName("lineEdit_casesNum")
        self.gridLayout.addWidget(self.lineEdit_casesNum, 2, 1, 1, 1)
        self.comboBox_Mutation_Method = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Mutation_Method.setStyleSheet("")
        self.comboBox_Mutation_Method.setObjectName("comboBox_Mutation_Method")
        self.gridLayout.addWidget(self.comboBox_Mutation_Method, 2, 2, 1, 1)
        self.comboBox_modelName = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_modelName.setStyleSheet("")
        self.comboBox_modelName.setObjectName("comboBox_modelName")
        self.gridLayout.addWidget(self.comboBox_modelName, 2, 0, 1, 1)
        self.label_casesNum = QtWidgets.QLabel(self.centralwidget)
        self.label_casesNum.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_casesNum.setObjectName("label_casesNum")
        self.gridLayout.addWidget(self.label_casesNum, 1, 1, 1, 1)
        self.label_modelName = QtWidgets.QLabel(self.centralwidget)
        self.label_modelName.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_modelName.setObjectName("label_modelName")
        self.gridLayout.addWidget(self.label_modelName, 1, 0, 1, 1)
        self.label_BC = QtWidgets.QLabel(self.centralwidget)
        self.label_BC.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_BC.setObjectName("label_BC")
        self.gridLayout.addWidget(self.label_BC, 1, 4, 1, 1)
        self.label_outPath = QtWidgets.QLabel(self.centralwidget)
        self.label_outPath.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_outPath.setObjectName("label_outPath")
        self.gridLayout.addWidget(self.label_outPath, 3, 3, 1, 1)
        self.label_seq = QtWidgets.QLabel(self.centralwidget)
        self.label_seq.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_seq.setObjectName("label_seq")
        self.gridLayout.addWidget(self.label_seq, 3, 0, 1, 1)
        self.label_symbolsNum = QtWidgets.QLabel(self.centralwidget)
        self.label_symbolsNum.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_symbolsNum.setObjectName("label_symbolsNum")
        self.gridLayout.addWidget(self.label_symbolsNum, 3, 1, 1, 1)
        self.label_CoverageStop = QtWidgets.QLabel(self.centralwidget)
        self.label_CoverageStop.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_CoverageStop.setObjectName("label_CoverageStop")
        self.gridLayout.addWidget(self.label_CoverageStop, 3, 4, 1, 1)
        self.lineEdit_CoverageStop = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_CoverageStop.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_CoverageStop.setObjectName("lineEdit_CoverageStop")
        self.gridLayout.addWidget(self.lineEdit_CoverageStop, 4, 4, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 2, 0, 1, 1)
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
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Avenir Next")
        font.setPointSize(36)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(0, 214, 0)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
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
        testRNN.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(testRNN)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1146, 22))
        self.menubar.setObjectName("menubar")
        testRNN.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(testRNN)
        self.statusbar.setObjectName("statusbar")
        testRNN.setStatusBar(self.statusbar)

        self.retranslateUi(testRNN)
        QtCore.QMetaObject.connectSlotsByName(testRNN)

    def retranslateUi(self, testRNN):
        _translate = QtCore.QCoreApplication.translate
        testRNN.setWindowTitle(_translate("testRNN", "TestRNN"))
        self.label_mode.setText(_translate("testRNN", "modeName"))
        self.label_SC.setText(_translate("testRNN", "SC threshold"))
        self.label_Mutation_Method.setText(_translate("testRNN", "Mutation Method"))
        self.label_casesNum.setText(_translate("testRNN", "Num. of Test Cases"))
        self.label_modelName.setText(_translate("testRNN", "modelName"))
        self.label_BC.setText(_translate("testRNN", "BC threshold"))
        self.label_outPath.setText(_translate("testRNN", "output file path"))
        self.label_seq.setText(_translate("testRNN", "seq in cells to test"))
        self.label_symbolsNum.setText(_translate("testRNN", "Num. of symbols"))
        self.label_CoverageStop.setText(_translate("testRNN", "CoverageStop"))
        self.pushButton_back.setText(_translate("testRNN", "Back"))
        self.label.setText(_translate("testRNN", "TestRNN"))
        self.pushButton.setText(_translate("testRNN", "Confirm"))

