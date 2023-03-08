import dataclasses
import os
import re
import threading
import pandas as pd
import numpy as np

'''
导入数据，查找指定分子所有id，
'''
class reax():
    def __init__(self,path):
        self.path=path
        files=os.listdir(path)
        for file in files:
            if re.search(r'\W', file.replace('.', '')) == None:
                if re.split("\W",file)[-1] == 'reaxff':
                    self.bond=os.path.join(path,file)
                elif re.split("\W",file)[-1] == 'lammpstrj':
                    self.trj=os.path.join(path,file)

    def blkup(self,molid,atomid):
        f=open(self.bond)
        molbond = []
        line = f.readline()
        molbond.append(line)
        for _ in range(6):
            line=f.readline()
            molbond.append(line)

        while line:
            line=f.readline()
            if 'Timestep' in line:
                molbond.append(line)
            if molid != '':
                if ' '+molid+'      ' in line:
                    molbond.append(line)
            if atomid != '':
                a='^ ' + atomid + ' '
                if re.match(a,line) != None:
                    molbond.append(line)
        f.close()
        f=open(os.path.join(self.path,'molbond.txt'),'w+')
        f.writelines(molbond)

    def trjlkp(self,molid):
        f = open(self.trj)
        moltrj = []
        line = f.readline()
        f.seek(0,0)
        count=96
        while line:
            line=f.readline()
            if 'TIMESTEP' in line:
                i = 0
                for _ in range(9):
                    i=i+1
                    if i !=4:
                        moltrj.append(line)
                    else:
                        line = str(count)+'\n'
                        moltrj.append(line)
                    line = f.readline()
            else:
                if molid !='':
                    if (' '+molid+' ' in line) or (' '+molid+' '+molid+' ' in line):
                        moltrj.append(line)
        f.close()
        f = open(os.path.join(self.path, 'moltrj.txt'), 'w+')
        f.writelines(moltrj)

    def sol(self,atoms):
        fb=open(os.path.join(self.path,'molbond.txt'),mode='r')
        ft=open(os.path.join(self.path, 'moltrj.txt'),mode='r')
        flag=True
        i0=0
        mol=[]
        mol0=[]
        mol1=[]
        mol2=[]
        bline = fb.readline()
        fb.seek(0, 0)
        while bline:
            if 'id' in bline:
                bline = fb.readline()
                for _ in range(atoms):
                    line=list(filter(None,re.split(r'[^\d.-]',bline)))
                    mol0.append(line)
                    bline=fb.readline()
                mol.append(mol0)
            elif 'Timestep' in bline:
                i0=i0+1
                timestep = list(filter(None, re.split(r'[^\d]', bline)))[0]
                mol2.append(timestep)
                bline = fb.readline()
                for i1 in range(atoms):
                    line = list(filter(None, re.split(r'[^\d.-]', bline)))
                    mol1.append(line)
                    for i2 in mol0:
                        if i2[0] == line[0]:
                            if len(i2) != len(line):
                                mol2.append(line[:2])
                                for i3 in line:
                                    if i3 not in i2[:3+i2[2]]:
                                        mol2.append(i3)
                                        mol2.append(line[line.index(i3)+line[2]+1])
                    bline=fb.readline()
                mol0=mol1
            break
        fb.close()
        ft.close()
        return

path='D:\Desktop\output\process'
molid='1'
atomid=''
atoms=97
print(reax(path).sol(atoms))



