import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
k-means算法
'''
class kmeans():
    # 初始化
    # 输入参数
    # path，文件路径；N，聚类的数目；prec，精度；iter，迭代次数。默认prec=1e-4,iter=100，两者满足其一结束分类
    def __init__(self,path,N,prec=1e-4,iter=100):
        self.N=N
        self.prec=prec
        self.iter=iter
        df=pd.read_excel(path,sheet_name='随机')
        self.arr=np.array(df[['花序号','花瓣长度','花瓣宽度']].values)

    # 绘制初始数据分布图
    def graph(self):
        fig,ax=plt.subplots()
        ax.scatter(x=self.arr[:, 1], y=self.arr[:, 2], marker='*')
        plt.show()

    # 距离函数
    def Euler(self,x,y):
        x,y=np.array(x),np.array(y)
        return np.sqrt(sum((x-y)**2))

    # 分类函数
    def km(self,sampleid=''):
        # 随机选择样本作为几何中心
        if sampleid:
            rand=sampleid
        else:
            rand=np.random.choice(self.arr[:,0],self.N,replace=False)
        # rand=[3,23,31,32]     #可自行选择样本
        geoc=[]
        for ped in self.arr:
            if ped[0] in rand:
                geoc.append(ped)
        geoc = np.array(geoc)
        print('开始进行k-means聚类')
        print('随机抽选{}个样本的几何中心为\n{}'.format(self.N,geoc))
        count = 0   #统计迭代次数
        while True:
            cla = {}    #创建字典用于存储分类
            for i0 in range(self.N):
                cla[i0+1]=[]
            count=count+1
            for ped in self.arr:
                i = 0
                dist = float('inf')
                geoc = list(geoc)
                for geo in geoc:
                    if self.Euler(ped[1:],geo[1:])<=dist:
                        dist=self.Euler(ped[1:],geo[1:])
                        i=i+1       #确定距离最小时的类
                cla[i].append(list(ped))
            # 更新几何中心
            geoc = np.array(geoc)
            geoc[:, 0] = np.arange(1, self.N + 1, 1)
            geo0 = copy.deepcopy(geoc)      #储存上一次的几何中心
            for i0 in cla.keys():
                # if len(cla[i0])==1:         #排除只有一个或零个样本的情况
                #     arr=np.array(cla[i0])#[1:]
                #     geoc=arr
                #     geoc[i0-1,0]=i0
                #     for j in range(len(cla[i0])):
                #         geoc[i0-1,j+1]=arr[j]
                if len(cla[i0])>1:
                    arr = np.array(cla[i0])[:, 1:]
                    ave = np.average(arr, axis=0)  # 以平均数来更新几何中心
                    geoc[i0 - 1, 1:] = ave
                else:
                    return print('第{}次迭代出现空集，需重新运行'.format(count))
            print('第{}次迭代的几何中心是\n{}'.format(count,geoc))
            # 判断几何中心是否发生改变
            if np.sum((geo0-geoc)**2) <= self.prec or count==self.iter: #prec，收敛精度；iter，迭代次数两者满足其一结束分类
                break
        print('最终分类如下')
        fig, ax = plt.subplots()
        for i3 in cla.keys():
            arr_cla=np.array(cla[i3])
            print('第{}类是\n{}'.format(i3,arr_cla))
            arr_cla=np.array(cla[i3])
            ax.scatter(x=arr_cla[:,1],y=arr_cla[:,2],label='class'+str(i3),marker='*')
            plt.legend()
        plt.show()

'''
kemeans(self,path,N,prec=1e-4,iter=100).km(sampleid)
输入文件路径path，聚类数目N，迭代精度prec，迭代次数iter,sampleid可根据情况自定
默认值 prec='1e-4' iter='100'
选km函数，即可得到k-means的分类结果
km函数里的sampleid可按照N选取，默认取随机数
note:
1 样本数少时，N不易取太大
2 迭代精度越高，越不容易收敛
2 若迭代分类出现空集或只有一个样本，分析原因是选择的初始几何中心不佳，或精度过高，或N过大
'''
path='D:\Desktop\第一次大作业题目-花瓣.xlsx'
prec=1e-3
iter=100
sampleid=[11,28,31,36] #根据N选取
print(kmeans(path,3).km())




