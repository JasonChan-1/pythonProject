import pandas
import pandas as pd
import numpy as np
import os

'''
k-means算法
1）
'''
class clasify():
    # path，文件路径，N，随机选N个样本，
    def __init__(self,path,N):
        self.N=N
        df=pd.read_excel(path,sheet_name='随机')
        self.arr=np.array(df[['花序号','花瓣长度','花瓣宽度']].values)

    # 距离函数
    def Euler(self,x,y):
        x,y=np.array(x),np.array(y)
        return np.sqrt(sum(x-y)**2)

    def km(self):
        name={}
        order={}
        key=np.arange(0,self.N)
        j=0
        r=[]
        order1 = [0]
        # while True:
        j+=j
        while True:
            rand=np.random.choice(self.arr.shape[0],self.N,replace=False)   #生成随机数
            if 0 not in rand:
                break
        r.append(rand)
        # 排序分类
        for i in range(self.N):
            name[rand[i]] = key[i]+1    #识别类目
            order[key[i]+1]=[]          #储存类目的数目
        #开始分类
        # 随机选择花瓣
        randPetal=[]
        for petal in self.arr:
            if petal[0] in rand:
                randPetal.append(petal)
        while True:
            #计算距离
            for petal in self.arr:
                flag=True
                for rP in randPetal:
                    if flag:
                        flag=False
                        dist=self.Euler(petal[1:],rP[1:])
                        name0=name[rP[0]]       #距离最大值时的类目赋值
                        info=list(petal)                 #距离最大值时的花瓣信息赋值
                    else:
                        if self.Euler(petal[1:],rP[1:]) < dist:
                            dist=self.Euler(petal[1:],rP[1:])
                            name0 = name[rP[0]]
                            info=list(petal)
                order[name0].append(info)       #在指定类目里添加距离最大值时的花瓣信息
            for i in range(self.N):
                for j in range(len(randPetal[i])-1):
                    randPetal[i][j]=order[i+1][j]

        print('第{}次随机数是:{}'.format(i0 + 1, r[i0]))
        for i1 in range(3):
            print('第{}类是:{}'.format(i1 + 1, order[i1 + 1]))
        print('\n')


path='D:\Desktop\第一次大作业题目-花瓣.xlsx'
print(clasify(path,3).km())




