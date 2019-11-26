#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import os

Masterdir = 'D:\MicroCT data\\4th batch bone mets loading study\w0w4composite'
ctan_list =os.path.join(Masterdir,os.path.basename(Masterdir)+'_CTANlist.ctl')
info ='Info=0000000001000000FFFF0000000000000000000000000000000000000000000000000000000000000000000000000000E60E993B64EED03F00000000000000000000000000000000000000000000000004000000000000002C'
i=0

if os.path.exists(ctan_list):
    os.remove(ctan_list)

#method 1
with open(ctan_list,'a') as thefile:
    thefile.write('[Dataset list]\n')
    for folder in sorted(os.listdir(Masterdir)):
        if os.path.isdir(os.path.join(Masterdir,folder)) and "composite" in folder:
            file =os.path.join(Masterdir,folder,os.listdir(os.path.join(Masterdir,folder))[10])
            thefile.write('Next=@{:d}\n'.format(i))
            thefile.write('[@{:d}]\n'.format(i))
            thefile.write('File={:s}\n'.format(file))
            thefile.write('{:s}\n'.format(info))
            i=i+1

'''
#method 2
w = open(ctan_list,'a')
w.write('[Dataset list]\n')
for folder in sorted(os.listdir(Masterdir)):
        if os.path.isdir(os.path.join(Masterdir,folder)):
            file =os.path.join(Masterdir,folder,os.listdir(os.path.join(Masterdir,folder))[10])
            w.write('Next=@{:d}\n'.format(i))
            w.write('[@{:d}]\n'.format(i))
            w.write('File={:s}\n'.format(file))
            w.write('{:s}\n'.format(info))
            i=i+1
w.close()
'''

'''
#method 3
ctan = ['[Dataset list]\n']
for folder in sorted(os.listdir(Masterdir)):
        if os.path.isdir(os.path.join(Masterdir,folder)):
            file =os.path.join(Masterdir,folder,os.listdir(os.path.join(Masterdir,folder))[10])
            ctan.append('Next=@{:d}\n'.format(i))
            ctan.append('[@{:d}]\n'.format(i))
            ctan.append('File={:s}\n'.format(file))
            ctan.append('{:s}\n'.format(info))
            i=i+1
w = open(ctan_list,'w')
w.writelines(ctan)
w.close()
'''

print('Done!')


