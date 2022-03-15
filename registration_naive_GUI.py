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
        self.resampleOnly=True
        self.coregTogether=False
        self.browseRef.clicked.connect(self.browseFoldersRef)
        self.browseTar.clicked.connect(self.browseFoldersTar)
        self.browseOutput.clicked.connect(self.browseFoldersOutput)
        self.showRefSeq.itemSelectionChanged.connect(self.refSelectionChanged)
        self.showTarSeq.itemSelectionChanged.connect(self.tarSelectionChanged)
        self.runButton.clicked.connect(self.registerImage)
        self.resampleOnlyCheck.stateChanged.connect(lambda: self.checkboxState(self.resampleOnlyCheck, "resampleOnly"))
        self.coRegTogCheck.stateChanged.connect(lambda:self.checkboxState(self.coRegTogCheck, "coregTogether"))
        self.outputdir = ""
        self.refUIDs = ""
        self.tarUIDs = ""

    def checkboxState(self, checkbox, attr):
        state = checkbox.isChecked()
        setattr(self, attr, state)
        self.labelLog.setText(checkbox.text()+" is checked:"+str(getattr(self, attr)))

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
        # self.refUID = self.showRefSeq.currentItem().text()
        self.refUIDs = [i.text() for i in self.showRefSeq.selectedItems()]
        self.labelRefSelected.setText("Row Selected: "+str(self.showRefSeq.currentRow()+1))
        self.labelLog.setText("")
        if len(self.refUIDs) == 0:
            self.labelLog.setText("")
        elif len(self.refUIDs) == 1:
            self.labelLog.setText("Mov chosen: "+self.refUIDs[0])
        else:
            self.labelLog.setText("Mov chosen: {} series selected".format(len(self.refUIDs)))


    def tarSelectionChanged(self):
        # self.tarUID = self.showTarSeq.currentItem().text()
        self.tarUIDs = [i.text() for i in self.showTarSeq.selectedItems()]
        self.labelTarSelected.setText("Row Selected: "+str(self.showTarSeq.currentRow()+1))
        self.labelLog.setText("")
        if len(self.tarUIDs) == 0:
            self.labelLog.setText("")
        elif len(self.tarUIDs) == 1:
            self.labelLog.setText("Mov chosen: "+self.tarUIDs[0])
        else:
            self.labelLog.setText("Mov chosen: {} series selected".format(len(self.tarUIDs)))

    def registerImage(self):
        if not(self.tarUIDs and self.refUIDs):
            self.labelLog.setText("Please choose Dicom series")
            return

        if not os.path.exists(self.outputdir):
            self.labelLog.setText("Please choose valid output director")
            return


        self.outputdir = self.showOutput.text()
        self.labelLog.setText("Reading images ... ")
        
        if self.coregTogether:
            reg=Registration(self.refDir, self.tarDir, ref_uid=self.refUIDs, tar_uid=self.tarUIDs)
        else:
            reg=Registration(self.refDir, self.tarDir, ref_uid=self.refUIDs[0], tar_uid=self.tarUIDs[0])
        self.labelLog.setText("Executing co-registration ... ")
        
        if self.resampleOnly:
            reg.resampleOnly()
            reg.save_dicom(reg.resampled_tar_img, self.outputdir)
            self.labelLog.setText("Resampled image has been saved!")
        else:
            reg.registration()
            reg.save_dicom(reg.reg_img, self.outputdir)
            self.labelLog.setText("Registered image has been saved!")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui=resource_path("registration_naive.ui")
    mainWindow = MainWindow(ui)
    mainWindow.showRefSeq.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
    mainWindow.showTarSeq.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainWindow)
    widget.setFixedHeight(400)
    widget.setFixedWidth(600)
    widget.show()
    sys.exit(app.exec_())