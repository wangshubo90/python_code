import sys, os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from registration_naive import *

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class MainWindow(QDialog):
    def __init__(self,ui):
        super(MainWindow, self).__init__()
        loadUi(ui, self)
        self.browseRef.clicked.connect(self.browseFoldersRef)
        self.browseTar.clicked.connect(self.browseFoldersTar)
        self.browseOutput.clicked.connect(self.browseFoldersOutput)
        self.showRefSeq.itemSelectionChanged.connect(self.refSelectionChanged)
        self.showTarSeq.itemSelectionChanged.connect(self.tarSelectionChanged)
        self.runButton.clicked.connect(self.registerImage)

    def browseFoldersRef(self):
        dirname=QFileDialog.getExistingDirectory(self, "Choose Directory", r"//192.168.2.2/data_share/chuan_zhang/11.25.2021/KONG_CUN_GUANG_228246")
        self.showRefSeq.clear()
        self.showRef.setText(dirname)
        self.labelLog.setText("Looking for Dicom series")
        self.refDir = dirname
        series_IDs = findDicomSeries(dirname)
        if not series_IDs:
            self.showRefSeq.addItems(["No valid Dicom series"])
        else:
            self.showRefSeq.addItems([str(i) for i in series_IDs])
        self.labelLog.setText("Choose a reference Dicom series")

    def browseFoldersTar(self):
        dirname=QFileDialog.getExistingDirectory(self, "Choose Directory", r"//192.168.2.2/data_share/chuan_zhang/11.25.2021/KONG_CUN_GUANG_228246")
        self.showTarSeq.clear()
        self.showTar.setText(dirname)
        self.labelLog.setText("Looking for Dicom series")
        self.tarDir = dirname
        series_IDs = findDicomSeries(dirname)
        if not series_IDs:
            self.showTarSeq.addItems(["No valid Dicom series"])
        else:
            self.showTarSeq.addItems([str(i) for i in series_IDs])
        self.labelLog.setText("Choose a moving Dicom series")

    def browseFoldersOutput(self):
        dirname=QFileDialog.getExistingDirectory(self, "Choose Directory", r"//192.168.2.2/data_share/chuan_zhang/11.25.2021/KONG_CUN_GUANG_228246")
        self.showOutput.setText(dirname)
        self.outputdir = dirname

    def refSelectionChanged(self):
        self.refUID = self.showRefSeq.currentItem().text()
        self.labelRefSelected.setText("Row Selected: "+str(self.showRefSeq.currentRow()+1))
        self.labelLog.setText("Ref chosen: "+self.refUID)


    def tarSelectionChanged(self):
        self.tarUID = self.showTarSeq.currentItem().text()
        self.labelTarSelected.setText("Row Selected: "+str(self.showTarSeq.currentRow()+1))
        self.labelLog.setText("")
        self.labelLog.setText("Mov chosen: "+self.tarUID)

    def registerImage(self):
        
        self.outputdir = self.showOutput.text()
        self.labelLog.setText("Reading images ... ")
        reg=Registration(self.refDir, self.tarDir, ref_uid=self.refUID, tar_uid=self.tarUID)
        self.labelLog.setText("Executing co-registration ... ")
        reg.registration()
        reg.save_dicom(self.outputdir)
        self.labelLog.setText("Registered image has been saved!")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui=resource_path("registration_naive.ui")
    mainWindow = MainWindow(ui)
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainWindow)
    widget.setFixedHeight(400)
    widget.setFixedWidth(600)
    widget.show()
    sys.exit(app.exec_())