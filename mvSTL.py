import os 
import shutil
import glob

masterdir = "/media/spl/D/MicroCT data/4th batch bone mets loading study/w0w0composite"
mvstldir = os.path.join(masterdir,"..","STL files")
if not os.path.exists(mvstldir):
    os.mkdir(mvstldir)

for fd in os.listdir(masterdir):
    if "week" in fd:
        sampleID = fd[0:4]+fd[11:-14]
        sampledir = os.path.join(mvstldir,sampleID)
        if not os.path.exists(sampledir):
            os.mkdir(sampledir)
        stls = glob.glob(os.path.join(masterdir,fd,"*.stl"))
        stls.sort
        newname = [sampleID+" week 0 "+"trab.stl",
                    sampleID+" week 1 "+"trab.stl",
                    sampleID+" week 0 "+"cort.stl",
                    sampleID+" week 1 "+"cort.stl"]

        for i in range(-1,len(stls)-1):
            shutil.move(stls[i],os.path.join(sampledir,newname[i+1]))

