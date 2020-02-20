import os 
import SimpleITK as sitk
from shubow_tools import imreadseq_multithread
import re
import logging

masterdir = r"F:\Shubo MicroCT data\CTimages for plotting\3rd batch loading"
pattern = re.compile(r"(\d{3}.week.\d).(left|right|runner|nonrunner).*composite")
#pattern = re.compile(r"(320 week 4) (right) w0w4composite")
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

for folder in sorted(os.listdir(masterdir)): 
    pattern.search(folder)
    sampleID=pattern.search(folder)
    if sampleID:
        separator = "_"
        logging.info("Processing {}".format(sampleID.group(0)))
        try:
            trab_imseq=os.path.join(masterdir,folder,"Trab-ROI-wx")
            trab = imreadseq_multithread(trab_imseq)
            trab_nii= os.path.join(masterdir,separator.join([sampleID.group(1),sampleID.group(2),"Trab.nii"]))
            sitk.WriteImage(trab,trab_nii)
            logging.info("Trabecular bone saved!")
        except Exception as ex:
            logging.info("Trabecular bone failed! {}".format(ex))
            pass

        try:           
            cort_imseq=os.path.join(masterdir,folder,"Cort-ROI-wx")
            cort = imreadseq_multithread(cort_imseq)
            cort_nii= os.path.join(masterdir,separator.join([sampleID.group(1),sampleID.group(2),"Cort.nii"]))
            sitk.WriteImage(cort,cort_nii)
            logging.info("Cortical bone saved!")
        except Exception as ex:
            logging.info("Cortical bone failed! {}".format(ex))
            pass