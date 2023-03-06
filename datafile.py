import os
import numpy as np
'''
处理data文件
'''
class change:
    #初始化，输入路径，读取文本
    def __init__(self, file_path, file_name, C, H, O):
        self.file=os.path.join(file_path,file_name)
        with open(self.file,'r') as f:
            self.lines=f.readlines()

    #查找位置
    def find(self):
        for line0 in self.lines:
            line=line0.split()
            if 'atoms' in line:
                self.Natoms=int(line[0])
            else:
                if 'Atoms' in line:
                    self.site=int(self.lines.index(line0)+2)
        return self.site,self.Natoms

    #提取内容
    def extrc(self):
        self.Atoms=[]
        for i in range(self.find()[0],self.find()[0]+self.find()[1]):
            self.Atoms.append(self.lines[i].split())
        Atoms_arr=np.array(self.Atoms)
        return Atoms_arr

    #处理内容
    def proce(self):
        di={1:C,2:H,3:O}    #CHO
        arr=self.extrc()
        for i in range(len(arr)):
            if arr[i,2] in di[1]:
                arr[i,2]='1'
            else:
                if arr[i, 2] in di[2]:
                    arr[i, 2] = '2'
                else:
                    if arr[i, 2] in di[3]:
                        arr[i, 2] = '3'
        return arr[:,0:7]

    #替换内容
    def alter(self):
        lines=self.lines
        ls0=self.proce().tolist()
        ls=[]
        #写入分隔符
        for i in ls0:
            k = '   '
            for j in i:
                k=k+j+'    '
            k=k+'\n'
            ls.append(k)
        #替换内容
        j=-1
        for i in range(self.find()[0],self.find()[0]+self.find()[1]):
            j=j+1
            lines[i]=ls[j]
        return lines

    #保存文件
    def save(self):
        lines=self.alter()
        with open(self.file+'.new','w+') as f:
            for i in range(len(lines)):
                for str in lines[i]:
                    f.write(str)

    def out(self):
        print('保存文件路径：',self.file+'.new')
        print('查找结果：',self.find()[0])
        self.save()

#测试
file_path='D:\lmp_file\\asphalt_o2'
file_name='iniP1.data'
C=['2','3','4','5']; H=['6']; O=['1']
data=change(file_path,file_name,C,H,O)
print(data.out())