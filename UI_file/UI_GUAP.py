# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_GUAP.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GUAP(object):
    def setupUi(self, GUAP):
        GUAP.setObjectName("GUAP")
        GUAP.resize(822, 600)
        self.centralwidget = QtWidgets.QWidget(GUAP)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Baloo")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color:rgb(0, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
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
        self.verticalLayout.addWidget(self.textEdit)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_gpulist = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_gpulist.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_gpulist.setObjectName("lineEdit_gpulist")
        self.gridLayout.addWidget(self.lineEdit_gpulist, 3, 3, 1, 1)
        self.label_limited = QtWidgets.QLabel(self.centralwidget)
        self.label_limited.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_limited.setObjectName("label_limited")
        self.gridLayout.addWidget(self.label_limited, 2, 6, 1, 1)
        self.lineEdit_outdir = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_outdir.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_outdir.setObjectName("lineEdit_outdir")
        self.gridLayout.addWidget(self.lineEdit_outdir, 3, 5, 1, 1)
        self.label_gpulist = QtWidgets.QLabel(self.centralwidget)
        self.label_gpulist.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_gpulist.setObjectName("label_gpulist")
        self.gridLayout.addWidget(self.label_gpulist, 2, 3, 1, 1)
        self.lineEdit_lr = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_lr.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_lr.setObjectName("lineEdit_lr")
        self.gridLayout.addWidget(self.lineEdit_lr, 1, 1, 1, 1)
        self.lineEdit_bs = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_bs.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_bs.setObjectName("lineEdit_bs")
        self.gridLayout.addWidget(self.lineEdit_bs, 1, 2, 1, 1)
        self.lineEdit_epochs = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_epochs.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_epochs.setObjectName("lineEdit_epochs")
        self.gridLayout.addWidget(self.lineEdit_epochs, 1, 3, 1, 1)
        self.lineEdit_l2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_l2.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_l2.setObjectName("lineEdit_l2")
        self.gridLayout.addWidget(self.lineEdit_l2, 1, 4, 1, 1)
        self.lineEdit_tau = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_tau.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_tau.setObjectName("lineEdit_tau")
        self.gridLayout.addWidget(self.lineEdit_tau, 1, 5, 1, 1)
        self.label_outdir = QtWidgets.QLabel(self.centralwidget)
        self.label_outdir.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_outdir.setObjectName("label_outdir")
        self.gridLayout.addWidget(self.label_outdir, 2, 5, 1, 1)
        self.label_lr = QtWidgets.QLabel(self.centralwidget)
        self.label_lr.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_lr.setObjectName("label_lr")
        self.gridLayout.addWidget(self.label_lr, 0, 1, 1, 1)
        self.label_tau = QtWidgets.QLabel(self.centralwidget)
        self.label_tau.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_tau.setObjectName("label_tau")
        self.gridLayout.addWidget(self.label_tau, 0, 5, 1, 1)
        self.label_l2 = QtWidgets.QLabel(self.centralwidget)
        self.label_l2.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_l2.setObjectName("label_l2")
        self.gridLayout.addWidget(self.label_l2, 0, 4, 1, 1)
        self.label_epochs = QtWidgets.QLabel(self.centralwidget)
        self.label_epochs.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_epochs.setObjectName("label_epochs")
        self.gridLayout.addWidget(self.label_epochs, 0, 3, 1, 1)
        self.label_bs = QtWidgets.QLabel(self.centralwidget)
        self.label_bs.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_bs.setObjectName("label_bs")
        self.gridLayout.addWidget(self.label_bs, 0, 2, 1, 1)
        self.label_dataset = QtWidgets.QLabel(self.centralwidget)
        self.label_dataset.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_dataset.setObjectName("label_dataset")
        self.gridLayout.addWidget(self.label_dataset, 0, 0, 1, 1)
        self.comboBox_dataset = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_dataset.setObjectName("comboBox_dataset")
        self.gridLayout.addWidget(self.comboBox_dataset, 1, 0, 1, 1)
        self.label_model = QtWidgets.QLabel(self.centralwidget)
        self.label_model.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_model.setObjectName("label_model")
        self.gridLayout.addWidget(self.label_model, 0, 6, 1, 1)
        self.comboBox_model = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_model.setObjectName("comboBox_model")
        self.gridLayout.addWidget(self.comboBox_model, 1, 6, 1, 1)
        self.comboBox_limited = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_limited.setObjectName("comboBox_limited")
        self.gridLayout.addWidget(self.comboBox_limited, 3, 6, 1, 1)
        self.label_epsilon = QtWidgets.QLabel(self.centralwidget)
        self.label_epsilon.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_epsilon.setObjectName("label_epsilon")
        self.gridLayout.addWidget(self.label_epsilon, 0, 7, 1, 1)
        self.lineEdit_epsilon = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_epsilon.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_epsilon.setObjectName("lineEdit_epsilon")
        self.gridLayout.addWidget(self.lineEdit_epsilon, 1, 7, 1, 1)
        self.label_beta1 = QtWidgets.QLabel(self.centralwidget)
        self.label_beta1.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_beta1.setObjectName("label_beta1")
        self.gridLayout.addWidget(self.label_beta1, 2, 0, 1, 1)
        self.lineEdit_beta1 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_beta1.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_beta1.setObjectName("lineEdit_beta1")
        self.gridLayout.addWidget(self.lineEdit_beta1, 3, 0, 1, 1)
        self.label_seed = QtWidgets.QLabel(self.centralwidget)
        self.label_seed.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_seed.setObjectName("label_seed")
        self.gridLayout.addWidget(self.label_seed, 2, 1, 1, 1)
        self.lineEdit_seed = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_seed.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_seed.setObjectName("lineEdit_seed")
        self.gridLayout.addWidget(self.lineEdit_seed, 3, 1, 1, 1)
        self.label_cuda = QtWidgets.QLabel(self.centralwidget)
        self.label_cuda.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_cuda.setObjectName("label_cuda")
        self.gridLayout.addWidget(self.label_cuda, 2, 2, 1, 1)
        self.comboBox_cuda = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_cuda.setObjectName("comboBox_cuda")
        self.gridLayout.addWidget(self.comboBox_cuda, 3, 2, 1, 1)
        self.label_resume = QtWidgets.QLabel(self.centralwidget)
        self.label_resume.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;\n"
"")
        self.label_resume.setObjectName("label_resume")
        self.gridLayout.addWidget(self.label_resume, 2, 4, 1, 1)
        self.comboBox_resume = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_resume.setObjectName("comboBox_resume")
        self.gridLayout.addWidget(self.comboBox_resume, 3, 4, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
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
        self.verticalLayout.addWidget(self.pushButton)
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
        self.verticalLayout.addWidget(self.pushButton_back)
        GUAP.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(GUAP)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 822, 22))
        self.menubar.setObjectName("menubar")
        GUAP.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(GUAP)
        self.statusbar.setObjectName("statusbar")
        GUAP.setStatusBar(self.statusbar)

        self.retranslateUi(GUAP)
        QtCore.QMetaObject.connectSlotsByName(GUAP)

    def retranslateUi(self, GUAP):
        _translate = QtCore.QCoreApplication.translate
        GUAP.setWindowTitle(_translate("GUAP", "MainWindow"))
        self.label.setText(_translate("GUAP", "GUAP"))
        self.label_limited.setText(_translate("GUAP", "limited"))
        self.label_gpulist.setText(_translate("GUAP", "GPU list"))
        self.label_outdir.setText(_translate("GUAP", "outdir"))
        self.label_lr.setText(_translate("GUAP", "learning Rate"))
        self.label_tau.setText(_translate("GUAP", "tau"))
        self.label_l2.setText(_translate("GUAP", "l2 Regularization"))
        self.label_epochs.setText(_translate("GUAP", "epochs"))
        self.label_bs.setText(_translate("GUAP", "batch size"))
        self.label_dataset.setText(_translate("GUAP", "dataset"))
        self.label_model.setText(_translate("GUAP", "model"))
        self.label_epsilon.setText(_translate("GUAP", "epsilon"))
        self.label_beta1.setText(_translate("GUAP", "beta1"))
        self.label_seed.setText(_translate("GUAP", "seed"))
        self.label_cuda.setText(_translate("GUAP", "cuda"))
        self.label_resume.setText(_translate("GUAP", "resume"))
        self.pushButton.setText(_translate("GUAP", "confirm"))
        self.pushButton_back.setText(_translate("GUAP", "back"))

