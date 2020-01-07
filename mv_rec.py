import os 
from glob import glob 
import shutil

cwd = "\\\\force.dbi.udel.edu\Lywang\Yoda1-loading study 12.8.2019"

for fd in glob(os.path.join(cwd,"*[0-9] week 4")):
    folder = os.path.basename(fd)
    recfd = os.path.join(fd,folder + "_Rec")
    for img in glob(os.path.join(fd,"*rec*.bmp")):
        shutil.move(img,recfd)



