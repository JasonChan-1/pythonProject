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
        # 查找指定分子或原子的键级
        f=open(self.bond)
        molbond = []
        line = f.readline()
        molbond.append(line)
        # for _ in range(6):
        #     line=f.readline()
        #     molbond.append(line)

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
        # 查找指定组成分子的所有原子轨迹
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
        f = open(os.path.join(self.path, 'mol'+molid+'.lammpstrj'), 'w+')
        f.writelines(moltrj)

    def reaxloc(self,atoms):
        #保存所有原子id
        fb=open(os.path.join(self.path,'molbond.txt'),mode='r')
        flag = True
        id=[]
        bline = fb.readline()
        fb.seek(0, 0)
        mol=[]
        while bline:
            bline = fb.readline()
            if 'Timestep' in bline:
                if flag:
                    flag=False
                    for _ in range(atoms):
                        bline = fb.readline()
                        line = list(filter(None, re.split(r'[^\d]', bline)))
                        id.append(line[0])
        #查找每个原子发生断裂的时间步
        i = -1
        timestep=[]
        for num in id:
            # mol.append('atomid  ' + num + '\n')
            fb.seek(0, 0)
            line = fb.readline()
            fb.seek(0, 0)
            flag = True
            # 对该分子键级文件进行循环
            while line:
                line = fb.readline()
                # line = re.split(r'[^\d.-]', line)
                if 'Timestep' in line:
                    tst = list(filter(None, re.split(r'[^\d]', line)))[0]
                    if tst not in timestep:
                        timestep.append(tst)
                a = '^ ' + num + ' '
                if re.match(a, line) != None:
                    i = i + 1
                    if flag:
                        flag=False
                        mol.append(tst+" "+line)
                    else:
                        line1 = list(filter(None, re.split(r'[\s\n]', line)))
                        moli_1=list(filter(None, re.split(r'[\s\n]', mol[i-1])))
                        if len(line1) != len(moli_1)-1:
                            mol.append(tst + " " + line)
                        else:
                            i=i-1

        mol1=[]
        for i0 in range(len(mol)):
            mol[i0]=list((filter(None,re.split('[^\d.-]',mol[i0]))))


        '''
        #按timesteps排序
        mol1=[]
        for ti in timestep:
            mol1.append('Timestep '+ti+'\n')
            for i1 in mol:
                if  list(filter(None, re.split(r'[^\d.-]', i1)))[0]== ti:
                    i1=re.sub(r'^\d*',' ',i1)
                    mol1.append(i1)
        '''
        '''
        f= open(os.path.join(self.path, 'reaxloc.txt'), 'w+')
        for i2 in mol:
            f.writelines(i2)
        '''
        return



path='D:\Desktop\output\process'
molid='1'
atomid=''
atoms=97
print(reax(path).reaxloc(atoms))