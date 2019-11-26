import os 
import shutil
import glob

masterdir = "D:\MicroCT data\\4th batch bone mets loading study\w0w1composite"
mvstldir = os.path.join(masterdir,"..","STL files")

for fd in os.listdir(masterdir):
    sampleId = fd[0:4]+fd[11:-14]
    sampledir = os.path.join(mvstldir,sampleId)
    if not os.path.exists(sampledir):
        os.mkdir(sampledir)
    stls = glob.glob(os.path.join(masterdir,fd,"*.stl"))
    newname = sampleID+" week 0"
    for stl in stls:
        shutil.move(stl,os.path.join(sampledir,sampleID+" week 0 trab st."))
