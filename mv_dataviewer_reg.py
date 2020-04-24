import shutil
import os 

origin = r'/media/spl/D/MicroCT_data/Machine learning/Jul 2018 1st batch L & R tibia/1st batch LR tibia'
destination = r'/media/spl/D/MicroCT_data/Machine learning/Dataviewer Registration'

os.chdir(origin)

for fd in os.listdir():
    if os.path.exists(os.path.join(fd,'Registration')):
        os.rename(os.path.join(fd, 'Registration'),
                 os.path.join(fd, fd))
        shutil.move(os.path.join(fd, fd),destination)