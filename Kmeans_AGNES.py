import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
k-means算法
'''
class classify():
    # 初始化
    # 输入参数
    # path，文件路径；N，聚类的数目；prec，精度；iter，迭代次数。默认prec=1e-4,iter=1000，两者满足其一结束分类
    def __init__(self,path,N):
        self.N=N
        df=pd.read_excel(path,sheet_name='随机')
        self.arr=np.array(df[['花序号','花瓣长度','花瓣宽度']].values)

    # 绘制初始数据分布图
    def graph(self):
        plt.figure()
        plt.title('origin')
        plt.scatter(x=self.arr[:, 1], y=self.arr[:, 2], marker='*')
        plt.show()

    # 距离函数
    def Euler(self,x,y):
        x,y=np.array(x),np.array(y)
        return np.sqrt(sum((x-y)**2))

    # 分类函数
    def km(self,prec=1e-4,iter=1000):
        # 随机选择样本作为几何中心
        rand=np.random.choice(self.arr[:,0],self.N,replace=False)
        geoc=[]
        for ped in self.arr:
            if ped[0] in rand:
                geoc.append(ped)
        geoc = np.array(geoc)
        print('随机抽选{}个样本的几何中心为\n{}'.format(self.N, geoc))
        print('开始进行k-means聚类')
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
                        i=i+1       # 确定距离最小时的类
                cla[i].append(list(ped))
            l = []
            for val in list(cla.values()):
                l.append(len(val))
            if 0 not in l:          # 用于判断分类里是否含有空集
                # 更新几何中心
                geo0 = copy.deepcopy(geoc)  # 储存上一次的几何中心
                geoc = np.array(geoc)
                geoc[:, 0] = np.arange(1, self.N + 1, 1)
                for i0 in cla.keys():
                    arr = np.array(list(cla[i0]))[:, 1:]
                    ave = np.average(arr, axis=0)  # 以平均数来更新几何中心
                    geoc[i0 - 1, 1:] = ave
                    # else:
                    #     return print('第{}次迭代出现空集，需重新运行'.format(count))
                print('第{}次迭代的几何中心是\n{}'.format(count,geoc))
                # 判断几何中心是否发生改变
                if np.sum((geo0-geoc)**2) <= prec or count > iter:  #prec，前后值差距；iter，迭代次数两者满足其一结束分类
                    break
            else:
                rand = np.random.choice(self.arr[:, 0], self.N, replace=False)
                geoc = []
                for ped in self.arr:
                    if ped[0] in rand:
                        geoc.append(ped)
                geoc = np.array(geoc)
                print('含空集，再次抽选{}个样本，几何中心为\n{}'.format(self.N, geoc))

        print('最终分类如下')
        fig, ax = plt.subplots()
        for i3 in cla.keys():
            arr_cla=np.array(cla[i3])
            print('第{}类是\n{}'.format(i3,arr_cla))
            arr_cla=np.array(cla[i3])
            ax.scatter(x=arr_cla[:,1],y=arr_cla[:,2],label='class'+str(i3),marker='*')
            plt.title('Kmeans')
            plt.legend()
        plt.show()


    # 用AGNES分类
    # 计算距离矩阵
    def agmat(self,ls):
        l=len(ls)
        agmat=np.zeros([l,l])
        ls=np.array(ls)
        for i in range(l):
            for j in range(l):
                if i !=j:
                    agmat[i,j]=self.Euler(ls[i],ls[j])
                else:
                    agmat[i,j]=float('inf')     #设定正无穷，方便后续定位矩阵最小值位置
        return agmat
    # 进行分类
    def agnes(self):
        arr0 = self.arr
        count=0         # 迭代次数
        tag=len(arr0)   # 用于生成新的编号
        cla = {}        # 所有编号的容器
        for i in range(len(arr0)):
            cla[i+1]=[arr0[i,0]]
        # print('初始化分类编号\n{}'.format(cla))
        # print(arr0)
        while True:
            count = count + 1
            tag=tag+count
            # print('第{}次迭代'.format(count))
            # print('分类数{}'.format(len(list(cla.values()))-1))
            agmat=self.agmat(arr0[:,1:])
            # print('距离矩阵\n{}'.format(agmat))
            whe=np.where(agmat==np.min(agmat))
            loc=[whe[0][0],whe[1][0]]       # 定位最小值位置
            # print('矩阵最小值时的位置是{}'.format(loc))

            # 更新分类编号
            no=[]
            no1=cla.pop(arr0[loc[0],0])
            no2=cla.pop(arr0[loc[1],0])
            for n1 in no1:
                no.append(n1)
            for n2 in no2:
                no.append(n2)
            cla[tag]=no
            # print('样本编号是{}'.format(no))

            # 按顺序设定分类编号用于输出
            cl0 = {}
            for i in range(len(list(cla.values()))):
                cl0[i+1]=list(cla.values())[i]
                # print('第{}类是{}'.format(i+1,cl0[i+1]))
            # print('分类情况{}'.format(cl0))

            # 计算类的几何中心
            arr0 = np.delete(arr0, loc, axis=0)
            new=[]
            for ped in self.arr:
                for no0 in no:
                    if no0 == ped[0]:
                        new.append(ped)
            new=np.array(new)
            # print('距离最小时的样本为\n{}'.format(new))
            ave=np.average(new[:,1:],axis=0)
            # print('几何中心是{}'.format(ave))
            ave=np.concatenate([np.array([tag]),ave],axis=0)[None]
            # print('生成待添加列表{}'.format(ave))
            arr0=np.concatenate([arr0,ave],axis=0)
            # print('更新的样本数据是\n{}\n'.format(arr0))
            if len(cl0) == self.N:   #以分类数为终止条件
                break

        # 输出
        print('分类数{}'.format(len(list(cla.values())) - 1))
        print('第{}次分类的距离矩阵是(对角线元素值被替换成正无穷)\n{}'.format(self.N,agmat))
        print('第{}次分类的矩阵最小值位置是{}'.format(self.N,loc))
        print('分类情况:')
        for i in range(len(list(cl0.values()))):
            cl0[i + 1] = list(cl0.values())[i]
            print('第{}类是{}'.format(i+1,cl0[i+1]))

        return cl0

    def aggraph(self):
        res=self.agnes()
        fig,ax=plt.subplots()

        # 创建字典收集数据
        data = {}
        for i0 in res.keys():
            data[i0]=[]
        for i in range(self.arr.shape[0]):
            for j in res.keys():
                for r in res[j]:
                    if r == self.arr[i,0]:
                        data[j].append(self.arr[i])
        # 绘图
        for i1 in data.keys():
            dat=np.array(list(data[i1]))
            plt.title('AGNES')
            ax.scatter(x=dat[:,1],y=dat[:,2],marker='*',label='class'+str(i1))
            plt.legend()
        plt.show()


'''
(1)K-means
kmeans(self,path,N).km(prec=1e-4,iter=100)
输入文件路径path，聚类数目N，迭代精度prec，迭代次数iter。默认值 prec='1e-5' iter='2000'
选km函数，即可得到k-means的分类结果
note:
1 样本数少，N不易取太大
2 迭代精度越高，条件满足时间越长
3 若迭代分类出现空集，将自动重新选择随机样本
4 分类结果受初始几何中心的影响比较大

(2)AGNES层次聚类
输入分类数N，选择agnes函数用于计算，aggraph用于绘图
'''
N=5
path='D:\Desktop\第一次大作业题目-花瓣.xlsx'
print(classify(path,N).graph())
print(classify(path,N).km(iter=2000,prec=1e-4))
print(classify(path,N).aggraph())

'''
结果对比，AGNES分类效果明显比Kmeans好
'''





