import os 
import SimpleITK as sitk
from shubow_tools import imreadseq_multithread
import re
import logging

masterdir = r"/media/spl/Seagate MicroCT/Shubo MicroCT data/CTimages for plotting"
pattern = re.compile(r"(\d{3}.week.\d).(left|right|runner|nonrunner).(.*)")

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

for folder in sorted(os.listdir(masterdir)): 
    pattern.search(folder)
    sampleID=pattern.search(folder)
    if sampleID:
        separator = "_"
        
        cort_imseq=os.path.join(masterdir,folder,"Cort-ROI-wx")
        trab_imseq=os.path.join(masterdir,folder,"Trab-ROI-wx")
        logging.info("Reading {}".format(sampleID.group(0)))
        cort = imreadseq_multithread(cort_imseq)
        trab = imreadseq_multithread(trab_imseq)
        
        cort_nii= os.path.join(masterdir,separator.join([sampleID.group(1),sampleID.group(2),"Cort.nii"]))
        trab_nii= os.path.join(masterdir,separator.join([sampleID.group(1),sampleID.group(2),"Trab.nii"]))
        logging.info("Writting {}".format(sampleID.group(0)))
        sitk.WriteImage(cort,cort_nii)
        sitk.WriteImage(trab,trab_nii)