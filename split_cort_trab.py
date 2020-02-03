import os 
import SimpleITK as sitk
from shubow_tools import imreadseq_multithread
import re

masterdir = r"/media/spl/Seagate MicroCT/Shubo MicroCT data/CTimages for plotting"
os.chdir(masterdir)
pattern = re.compile(r"(\d{3}.week.\d).(left|right|runner|nonrunner).(.*)")

for folder in sorted(os.listdir(masterdir)): 
    pattern.search(folder)
    sampleID=pattern.search(folder)
    if sampleID:
        separator = "_"
        
        cort_imseq=os.path.join(masterdir,folder,"Cort-ROI-wx")
        trab_imseq=os.path.join(masterdir,folder,"Trab-ROI-wx")
        '''
        cort = imreadseq_multithread()
        trab = imreadseq_multithread()
        '''
        cort_nii= os.path.join(masterdir,separator.join([sampleID.group(1),sampleID.group(2),"Cort.nii"]))
        trab_nii= os.path.join(masterdir,separator.join([sampleID.group(1),sampleID.group(2),"Trab.nii"]))

        print(cort_imseq+"\n"+trab_imseq+"\n"+cort_nii+"\n"+trab_nii)