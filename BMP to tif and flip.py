import os
import SimpleITK as sitk 
from shubow_tools import *
import logging

os.chdir(r'/media/spl/D/MicroCT_data/Machine learning/Jul 2018 1st batch L & R tibia/L & R tibia 7.9')

pattern = re.compile(r'.*(left|right) tibia')
format = "%(asctime)s:%(message)s"
logging.basicConfig(format = format, level = logging.INFO, datefmt="%H:%M:%S")

for fd in sorted(os.listdir('.'))[:]:
    logging.info("Conversion started: {}".format(fd))
    img = imreadseq_multithread(fd)
    side = pattern.search(fd).group(1)

    outfd = os.path.join('../1st batch LR tibia',fd)
    if not os.path.exists(outfd):
        os.mkdir(outfd)

    if side == 'left':
        imsaveseq(img,outfd,fd)
    elif side == 'right':
        img = img[::-1,:,:]
        imsaveseq(img,outfd,fd)
    logging.info("Conversion finished")