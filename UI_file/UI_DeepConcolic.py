# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_DeepConcolic.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DeepConcolic(object):
    def setupUi(self, DeepConcolic):
        DeepConcolic.setObjectName("DeepConcolic")
        DeepConcolic.resize(1172, 784)
        self.centralwidget = QtWidgets.QWidget(DeepConcolic)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Baloo")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 116, 219);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.pushButton_bfc = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_bfc.setStyleSheet("QPushButton:focus{outline: none;}\n"
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
        self.pushButton_bfc.setObjectName("pushButton_bfc")
        self.gridLayout_2.addWidget(self.pushButton_bfc, 5, 0, 1, 1)
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
        self.gridLayout_2.addWidget(self.pushButton_back, 7, 0, 1, 1)
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
        self.pushButton_bfdc = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_bfdc.setStyleSheet("QPushButton:focus{outline: none;}\n"
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
        self.pushButton_bfdc.setObjectName("pushButton_bfdc")
        self.gridLayout_2.addWidget(self.pushButton_bfdc, 6, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_top_classes = QtWidgets.QLabel(self.centralwidget)
        self.label_top_classes.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_top_classes.setObjectName("label_top_classes")
        self.gridLayout.addWidget(self.label_top_classes, 4, 2, 1, 1)
        self.lineEdit_feature_index = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_feature_index.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_feature_index.setObjectName("lineEdit_feature_index")
        self.gridLayout.addWidget(self.lineEdit_feature_index, 5, 4, 1, 1)
        self.label_dbnc_spec = QtWidgets.QLabel(self.centralwidget)
        self.label_dbnc_spec.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_dbnc_spec.setObjectName("label_dbnc_spec")
        self.gridLayout.addWidget(self.label_dbnc_spec, 4, 5, 1, 1)
        self.label_feature_index = QtWidgets.QLabel(self.centralwidget)
        self.label_feature_index.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_feature_index.setObjectName("label_feature_index")
        self.gridLayout.addWidget(self.label_feature_index, 4, 4, 1, 1)
        self.comboBox_layers = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_layers.setObjectName("comboBox_layers")
        self.gridLayout.addWidget(self.comboBox_layers, 5, 3, 1, 1)
        self.label_dbnc_or_bn = QtWidgets.QLabel(self.centralwidget)
        self.label_dbnc_or_bn.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_dbnc_or_bn.setObjectName("label_dbnc_or_bn")
        self.gridLayout.addWidget(self.label_dbnc_or_bn, 4, 7, 1, 1)
        self.lineEdit_top_classes = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_top_classes.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_top_classes.setObjectName("lineEdit_top_classes")
        self.gridLayout.addWidget(self.lineEdit_top_classes, 5, 2, 1, 1)
        self.label_layers = QtWidgets.QLabel(self.centralwidget)
        self.label_layers.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_layers.setObjectName("label_layers")
        self.gridLayout.addWidget(self.label_layers, 4, 3, 1, 1)
        self.lineEdit_dbnc_spec = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_dbnc_spec.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_dbnc_spec.setObjectName("lineEdit_dbnc_spec")
        self.gridLayout.addWidget(self.lineEdit_dbnc_spec, 5, 5, 1, 1)
        self.lineEdit_init = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_init.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_init.setObjectName("lineEdit_init")
        self.gridLayout.addWidget(self.lineEdit_init, 1, 7, 1, 1)
        self.label_init = QtWidgets.QLabel(self.centralwidget)
        self.label_init.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_init.setObjectName("label_init")
        self.gridLayout.addWidget(self.label_init, 0, 7, 1, 1)
        self.comboBox_dataset = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_dataset.setObjectName("comboBox_dataset")
        self.gridLayout.addWidget(self.comboBox_dataset, 1, 1, 1, 1)
        self.comboBox_model = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_model.setObjectName("comboBox_model")
        self.gridLayout.addWidget(self.comboBox_model, 1, 2, 1, 1)
        self.lineEdit_outputs = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_outputs.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_outputs.setObjectName("lineEdit_outputs")
        self.gridLayout.addWidget(self.lineEdit_outputs, 1, 3, 1, 1)
        self.label_model = QtWidgets.QLabel(self.centralwidget)
        self.label_model.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_model.setObjectName("label_model")
        self.gridLayout.addWidget(self.label_model, 0, 2, 1, 1)
        self.comboBox_filters = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_filters.setObjectName("comboBox_filters")
        self.gridLayout.addWidget(self.comboBox_filters, 3, 4, 1, 1)
        self.label_filters = QtWidgets.QLabel(self.centralwidget)
        self.label_filters.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_filters.setObjectName("label_filters")
        self.gridLayout.addWidget(self.label_filters, 2, 4, 1, 1)
        self.lineEdit_max_i = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_max_i.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_max_i.setObjectName("lineEdit_max_i")
        self.gridLayout.addWidget(self.lineEdit_max_i, 1, 8, 1, 1)
        self.lineEdit__extra_tests = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit__extra_tests.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit__extra_tests.setObjectName("lineEdit__extra_tests")
        self.gridLayout.addWidget(self.lineEdit__extra_tests, 3, 3, 1, 1)
        self.comboBox_norm = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_norm.setObjectName("comboBox_norm")
        self.gridLayout.addWidget(self.comboBox_norm, 1, 5, 1, 1)
        self.label_norm = QtWidgets.QLabel(self.centralwidget)
        self.label_norm.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_norm.setObjectName("label_norm")
        self.gridLayout.addWidget(self.label_norm, 0, 5, 1, 1)
        self.comboBox_save_all = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_save_all.setObjectName("comboBox_save_all")
        self.gridLayout.addWidget(self.comboBox_save_all, 3, 1, 1, 1)
        self.comboBox_criterion = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_criterion.setObjectName("comboBox_criterion")
        self.gridLayout.addWidget(self.comboBox_criterion, 1, 4, 1, 1)
        self.label_max_i = QtWidgets.QLabel(self.centralwidget)
        self.label_max_i.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_max_i.setObjectName("label_max_i")
        self.gridLayout.addWidget(self.label_max_i, 0, 8, 1, 1)
        self.label_criterion = QtWidgets.QLabel(self.centralwidget)
        self.label_criterion.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_criterion.setObjectName("label_criterion")
        self.gridLayout.addWidget(self.label_criterion, 0, 4, 1, 1)
        self.label_outputs = QtWidgets.QLabel(self.centralwidget)
        self.label_outputs.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_outputs.setObjectName("label_outputs")
        self.gridLayout.addWidget(self.label_outputs, 0, 3, 1, 1)
        self.label_dataset = QtWidgets.QLabel(self.centralwidget)
        self.label_dataset.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_dataset.setObjectName("label_dataset")
        self.gridLayout.addWidget(self.label_dataset, 0, 1, 1, 1)
        self.label_extra_tests = QtWidgets.QLabel(self.centralwidget)
        self.label_extra_tests.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_extra_tests.setObjectName("label_extra_tests")
        self.gridLayout.addWidget(self.label_extra_tests, 2, 3, 1, 1)
        self.label_seed = QtWidgets.QLabel(self.centralwidget)
        self.label_seed.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_seed.setObjectName("label_seed")
        self.gridLayout.addWidget(self.label_seed, 2, 2, 1, 1)
        self.label_save_all = QtWidgets.QLabel(self.centralwidget)
        self.label_save_all.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_save_all.setObjectName("label_save_all")
        self.gridLayout.addWidget(self.label_save_all, 2, 1, 1, 1)
        self.lineEdit_seed = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_seed.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_seed.setObjectName("lineEdit_seed")
        self.gridLayout.addWidget(self.lineEdit_seed, 3, 2, 1, 1)
        self.label_norm_factor = QtWidgets.QLabel(self.centralwidget)
        self.label_norm_factor.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_norm_factor.setObjectName("label_norm_factor")
        self.gridLayout.addWidget(self.label_norm_factor, 2, 5, 1, 1)
        self.lineEdit_norm_factor = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_norm_factor.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_norm_factor.setObjectName("lineEdit_norm_factor")
        self.gridLayout.addWidget(self.lineEdit_norm_factor, 3, 5, 1, 1)
        self.label_lb_hard = QtWidgets.QLabel(self.centralwidget)
        self.label_lb_hard.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_lb_hard.setObjectName("label_lb_hard")
        self.gridLayout.addWidget(self.label_lb_hard, 2, 7, 1, 1)
        self.lineEdit_lb_hard = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_lb_hard.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_lb_hard.setObjectName("lineEdit_lb_hard")
        self.gridLayout.addWidget(self.lineEdit_lb_hard, 3, 7, 1, 1)
        self.label_lb_noise = QtWidgets.QLabel(self.centralwidget)
        self.label_lb_noise.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_lb_noise.setObjectName("label_lb_noise")
        self.gridLayout.addWidget(self.label_lb_noise, 2, 8, 1, 1)
        self.lineEdit_lb_noise = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_lb_noise.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_lb_noise.setObjectName("lineEdit_lb_noise")
        self.gridLayout.addWidget(self.lineEdit_lb_noise, 3, 8, 1, 1)
        self.label_ratio = QtWidgets.QLabel(self.centralwidget)
        self.label_ratio.setStyleSheet("color: white;\n"
"background-color:rgb(51,51,51);\n"
"border:hide;\n"
"border-radius:4px;\n"
"padding-left: 10px;\n"
"padding-top: 8px;\n"
"padding-right: 7px;\n"
"padding-bottom: 9px;")
        self.label_ratio.setObjectName("label_ratio")
        self.gridLayout.addWidget(self.label_ratio, 4, 1, 1, 1)
        self.lineEdit_ratio = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_ratio.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_ratio.setObjectName("lineEdit_ratio")
        self.gridLayout.addWidget(self.lineEdit_ratio, 5, 1, 1, 1)
        self.lineEdit_dbnc_or_bn = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_dbnc_or_bn.setStyleSheet("border:1px solid rgb(79,170,227);border-radius:3px;margin:0;background-color:rgb(250,250,250);color: rgb(106,106,106);\n"
"")
        self.lineEdit_dbnc_or_bn.setObjectName("lineEdit_dbnc_or_bn")
        self.gridLayout.addWidget(self.lineEdit_dbnc_or_bn, 5, 7, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 3, 0, 1, 1)
        DeepConcolic.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(DeepConcolic)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1172, 22))
        self.menubar.setObjectName("menubar")
        DeepConcolic.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(DeepConcolic)
        self.statusbar.setObjectName("statusbar")
        DeepConcolic.setStatusBar(self.statusbar)

        self.retranslateUi(DeepConcolic)
        QtCore.QMetaObject.connectSlotsByName(DeepConcolic)

    def retranslateUi(self, DeepConcolic):
        _translate = QtCore.QCoreApplication.translate
        DeepConcolic.setWindowTitle(_translate("DeepConcolic", "MainWindow"))
        self.label.setText(_translate("DeepConcolic", "Deep Concolic"))
        self.pushButton_bfc.setText(_translate("DeepConcolic", "run Bayesian-network Feature Coverage (BFC)"))
        self.pushButton_back.setText(_translate("DeepConcolic", "back"))
        self.pushButton_bfdc.setText(_translate("DeepConcolic", "run Bayesian-network Feature-dependence Coverage (BFdC)"))
        self.label_top_classes.setText(_translate("DeepConcolic", "top-classes"))
        self.label_dbnc_spec.setText(_translate("DeepConcolic", "dbnc-spec"))
        self.label_feature_index.setText(_translate("DeepConcolic", "feature-index"))
        self.label_dbnc_or_bn.setText(_translate("DeepConcolic", "dbnc-abstr or bn-abstr"))
        self.label_layers.setText(_translate("DeepConcolic", "layers"))
        self.label_init.setText(_translate("DeepConcolic", "init"))
        self.label_model.setText(_translate("DeepConcolic", "model"))
        self.label_filters.setText(_translate("DeepConcolic", "filters"))
        self.label_norm.setText(_translate("DeepConcolic", "norm"))
        self.label_max_i.setText(_translate("DeepConcolic", "max-iterations"))
        self.label_criterion.setText(_translate("DeepConcolic", "criterion"))
        self.label_outputs.setText(_translate("DeepConcolic", "outputs"))
        self.label_dataset.setText(_translate("DeepConcolic", "dataset"))
        self.label_extra_tests.setText(_translate("DeepConcolic", "extra-tests"))
        self.label_seed.setText(_translate("DeepConcolic", "rng-seed"))
        self.label_save_all.setText(_translate("DeepConcolic", "save-all-tests"))
        self.label_norm_factor.setText(_translate("DeepConcolic", "norm-factor"))
        self.label_lb_hard.setText(_translate("DeepConcolic", "lb-hard"))
        self.label_lb_noise.setText(_translate("DeepConcolic", "lb-noise"))
        self.label_ratio.setText(_translate("DeepConcolic", "mcdc-cond-ratio"))

