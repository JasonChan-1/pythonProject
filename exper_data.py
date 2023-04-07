import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy import optimize as op
import math

    #GR参数计算函数
def predic(p,x):
    alpha, beta, gamma, delta, a5, a25 = p
    x=np.array(x)
    y = np.zeros_like(x)
    x1 = x[:16];x2 = x[16:-16];x3 = x[-16:]
    y[:16]=delta + alpha / (1 + np.exp(beta + gamma * (x1 - a5)))
    y[16:-16]= delta +alpha / (1 + np.exp(beta + gamma * x2))
    y[-16:]= delta + alpha / (1 + np.exp(beta + gamma * (x3 - a25)))
    return y # x表示logtime

def progsol(p, x, y):
    y0 = sum((predic(p, x)[:16] - y[:16]) ** 2 + (predic(p, x)[16:-16] - y[16:-16]) ** 2 + (predic(p, x)[-16:] - y[-16:]) ** 2)
    return y0

def logG(p):
    alpha, beta, gamma, delta = p
    logG = delta + alpha / (1 + np.exp(beta + gamma * np.log10([np.pi * 2 / 0.005])))
    return logG

    #复数模量计算函数
def cml(p,x,y):
    alpha, beta = p
    x, y = np.array(x), np.array(y)
    x0 = np.zeros_like(x)
    x1 = x[:13];x2 = x[13:-13];x3 = x[-13:]
    x0[:13] = x1*10**alpha
    x0[13:-13] = x2
    x0[-13:] = x3*10**(-beta)
    y0=y*1e-3
    return x0,y0

def cmfun(p,x,y):
    alpha,beta,a,b=p
    x,y=np.array(x),np.array(y)
    x0=np.zeros_like(x)
    x1 = x[:13];x2 = x[13:-13];x3 = x[-13:]
    x0[:13]=np.log10(x1)+alpha;x0[13:-13]=np.log10(x2);x0[-13:]=np.log10(x3)-beta
    y0=np.log10(y*1e-3)
    return sum((a*x0+b-y0)**2)

#计算主类
class cal():
    def __init__(self,path):
        self.files=os.listdir(path)
        self.sheets_name=[]
        self.path=path
        for file in self.files:
            if re.search(r'^\w', file) != None:
                file_path = os.path.join(path, file)
                sheets=pd.read_excel(file_path,sheet_name=None)
                for sheet in sheets:
                    self.sheets_name.append(sheet)
        self.fushu=[];self.che=[];self.gr=[];self.jin=[]
        for name in self.sheets_name:
            if ('fushu' == name[-5:]) or ('FuShu' == name[-5:]):
                self.fushu.append(name)
            elif ('che' == name[-3:]) or ('Che' == name[-3:]):
                self.che.append(name)
            elif 'GR' == name[-2:]:
                self.gr.append(name)
            elif ('jin' == name[-3:]) or ('Jin' == name[-3]):
                self.jin.append(name)
    #复数模量
    def graph(self):
        plt.figure(figsize=(8, 5))
        plt.loglog()
        plt.xticks(
            ticks=[0.01, 0.1, 1, 10, 100, 1000],
            labels=['0.01', '0.1', '1', '10', '100', '1000'])
        plt.yticks(
            ticks=[0.1, 1, 10, 100, 1000, 10000],
            labels=['0.1', '1', '10', '100', '1000', '10000'])
        plt.xlabel(r'$\Omega$(rad/s)')
        plt.ylabel(r'G*(kPa)')
        plt.xlim([0.01, 1000])
        plt.ylim([0.1, 10000])

    def cm(self):
        i = -1
        marker=['>','*','^','o','+','1','s','p','h']
        color=['#e9963e','#f23b27', '#65a9d7', '#304f9e','#83639f','#ea7827','#c22f2f','#449945']
        k=[[0.7,-0.7],[0.65,-0.65],[0.7,-0.7],[0.7,-0.7],[0.83,-0.9],[0.80,-0.8],[0.85,-0.9],[0.85, -0.9],
           [0.85, -0.9], [0.8, -0.8], [0.85, -0.9], [0.85, -0.9], [0.85, -0.9], [0.85, -0.9],]  #调参
        self.graph()
        for file in self.files:
            if re.search(r'^\w', file) != None:
                file_path = os.path.join(path, file)
                for fushu_name in self.fushu:
                    if fushu_name in pd.read_excel(file_path, sheet_name=None):
                        i = i + 1
                        df_fushu = pd.read_excel(file_path, sheet_name=fushu_name)          #读取表格
                        df_fushu=df_fushu[['Angular frequency', 'Complex modulus']][1:]     #提取角频率和复数模量
                        ar_fushu=pd.DataFrame(df_fushu,dtype=float).values                  #数据转为float
                        p0=[1]*4    #设置初始值,alpha,beta,a,b
                        res=op.least_squares(cmfun,p0,args=(ar_fushu[:,0],ar_fushu[:,1]))   #最小二乘法拟合
                        fit=res.x[:2]
                        x,y=cml(fit,ar_fushu[:,0],ar_fushu[:,1])
                        print("拟合值是：\n{}\n".format(fit))
                        # arr = np.concatenate((ar35, ar45, ar55), axis=0)
                        plt.scatter(x=x,y=y,label=fushu_name[:-5],marker=marker[i],alpha=0.8)
                        plt.legend()
        plt.show()
        # return fit

    # 车辙因子
    def rf(self):
        i=-1
        plt.xticks(np.arange(0,100,10))
        plt.yticks(np.arange(0,200,50))
        plt.tick_params(direction='in', axis='both',length=8,width=1)
        plt.xlabel(r'$T(℃)kPa$')
        plt.ylabel(r'$G*/sin\delta$')
        marker=['>','*','^','o','+','1','s','p','h','>','*','^','o','+','1','s','p','h']
        color = ['#e9963e', '#f23b27', '#65a9d7', '#304f9e', '#83639f','#ea7827', '#c22f2f', '#449945']
        for file in self.files:
            if re.search(r'^\w', file) != None:
                file_path = os.path.join(path, file)
                for name in self.che:
                    if name in pd.read_excel(file_path, sheet_name=None):
                        i=i+1
                        df = pd.read_excel(file_path, sheet_name=name)
                        ar = df[['Temperature', 'Phase angle', 'Complex modulus']][1:].values
                        ar[:, 2] = ar[:, 2] / 1e3
                        for j in range(len(ar)):
                            ar[j,1]=ar[j,2]/(np.sin(ar[j,1]/180*np.pi))
                        plt.plot(ar[:,0],ar[:,1],label=name[:-3],lw=1.5,
                                 marker='*', color=color[i],mec=color[i],mfc='none',ms=5)
                        plt.legend()
        plt.show()

    # GR参数
    def GR(self):
        arr=[]
        for file in self.files:
            if re.search(r'^\w', file) != None:
                file_path = os.path.join(path, file)
                for name in self.gr:
                    if name in pd.read_excel(file_path, sheet_name=None):
                        df0 = pd.read_excel(file_path, sheet_name=name)
                        print('正在处理{}'.format(name))
                        df = df0[['Storage modulus', 'Loss modulus', 'Angular frequency']][1:]
                        ar=df.values
                        logtime=np.zeros([len(ar),1])
                        logg1=np.zeros([len(ar),1])
                        logg2=np.zeros([len(ar), 1])
                        for j in range(len(ar)):
                            time=2*math.pi/ar[j,2]
                            logtime[j,0]=np.emath.logn(10,time)
                            logg1[j,0]=np.emath.logn(10,ar[j,0])
                            logg2[j,0]=np.emath.logn(10,ar[j,1])
                        # 共轭梯度局解
                        # p0 = [1]*6
                        # fit1 = op.minimize(progsol, p0, args=(logtime, logg1), method='CG')
                        # fit2 = op.minimize(progsol, p0, args=(logtime, logg2), method='CG')
                        ''''''
                        bound = [[-20, 20]] * 6
                        iter=2000
                        fit1 = op.dual_annealing(progsol, args=(logtime, logg1), bounds=bound,maxiter=iter)
                        fit2 = op.dual_annealing(progsol, args=(logtime, logg2), bounds=bound,maxiter=iter)
                        # rt = np.zeros_like(logtime)
                        print(fit1)
                        print(fit2)
                        p1 = fit1.x; prec1 = fit1.fun
                        p2 = fit2.x; prec2 = fit2.fun
                        # 计算各指标
                        G = np.sqrt((10 ** logG(p1[:4])) ** 2 + (10 ** logG(p2[:4])) ** 2)
                        Delta = np.arctan((10 ** logG(p2[:4])) / (10 ** logG(p1[:4])))
                        GR = G / 1e3 * (np.cos(Delta)) ** 2 / np.sin(Delta)
                        arr.append([name[:-2],p1,prec1,p2,prec2,logG(p1[:4]),logG(p2[:4]),G,Delta,GR])

        df1=pd.DataFrame(data=arr,columns=['name','p1','prec1','p2','prec2','logG1','logG2','G','Delta','GR'])
        df1.to_excel('/Users/jasonchan/Desktop/GRres.xlsx',sheet_name='res',index=False )

if __name__ == "__main__":
    path='/Users/jasonchan/Desktop/process'
    res=cal(path).GR()
    print(res)