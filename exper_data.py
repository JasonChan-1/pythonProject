import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy import optimize as op
import math
from sko.SA import SABoltzmann
    #GR参数计算函数
def progsol(p,x,y):
    alpha, beta, gamma, delta, a5, a25 = p
    x,y=np.array(x),np.array(y)
    x1=x[:16];x2=x[:16];x3=x[:16]
    y1 =y[:16];y2=y[:16];y3=y[:16]
    y0 = sum(((delta + alpha / (1 + np.exp(beta + gamma * (x1 - a5))))-y1) ** 2+(( delta +alpha / (1 + np.exp(beta + gamma * x2)))-y2) ** 2+(( delta + alpha / (1 + np.exp(beta + gamma * (x3 - a25))))-y3) ** 2)
    return y0  # x表示logtime

def predic(p,x):
    alpha, beta, gamma, delta, a5, a25 = p
    x=np.array(x)
    y = np.zeros_like(x)
    x1 = x[:16];x2 = x[16:-16];x3 = x[-16:]
    y[:16]=delta + alpha / (1 + np.exp(beta + gamma * (x1 - a5)))
    y[16:-16]= delta +alpha / (1 + np.exp(beta + gamma * x2))
    y[-16:]= delta + alpha / (1 + np.exp(beta + gamma * (x3 - a25)))
    return y

def grg(p):
    alpha, beta, gamma, delta = p
    logg=delta + alpha / (1 + np.exp(beta + gamma * np.emath.logn(10,[np.pi*2/0.005])))
    return logg

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
            if re.search(r'\W', file.replace('.', '')) == None:
                file_path=os.path.join(path,file)
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
            if re.search(r'\W', file.replace('.', '')) == None:
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
        plt.figure(figsize=(5,5))
        plt.xticks(np.arange(0,100,10))
        plt.yticks(np.arange(0,200,50))
        plt.tick_params(direction='in', axis='both',length=8,width=1)
        plt.xlabel(r'$T(℃)kPa$')
        plt.ylabel(r'$G*/sin\delta$')
        marker=['>','*','^','o','+','1','s','p','h','>','*','^','o','+','1','s','p','h']
        color = ['#e9963e', '#f23b27', '#65a9d7', '#304f9e', '#83639f','#ea7827', '#c22f2f', '#449945']
        for file in self.files:
            if re.search(r'\W', file.replace('.', '')) == None:
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
                                 marker=marker[i], color=color[i],mec=color[i+6],mfc='none',ms=8)
                        plt.legend()
        plt.show()

    # GR参数
    def GR(self):
        plt.figure()
        for file in self.files:
            if re.search(r'\W', file.replace('.', '')) == None:
                file_path = os.path.join(path, file)
                for name in self.gr:
                    if name in pd.read_excel(file_path, sheet_name=None):
                        df0 = pd.read_excel(file_path, sheet_name=name)
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
                        # 双重退火全解
                        p0=[1]*6
                        bound = [[-25, 25]] * 6
                        fit1=op.dual_annealing(progsol,args=(logtime,logg1),bounds=bound)
                        fit2=op.dual_annealing(progsol,args=(logtime,logg2),bounds=bound)

                        p1=fit1.x;prec1=fit1.fun
                        p2=fit2.x;prec2=fit2.fun
                        # 计算各指标
                        alpha, beta, gamma, delta = p1[:4]
                        logG1=delta + alpha / (1 + np.exp(beta + gamma * np.emath.logn(10, [np.pi * 2 / 0.005])))
                        alpha, beta, gamma, delta = p2[:4]
                        logG2 = delta + alpha / (1 + np.exp(beta + gamma * np.emath.logn(10, [np.pi * 2 / 0.005])))
                        G=np.sqrt( np.exp(10,logG1)**2+np.exp(10,logG2)**2 )
                        Delta=np.arctan(np.exp(10,logG2)/np.exp(10,logG1))
                        GR=G/10e3*(np.cos(Delta))**2/np.sin(Delta)
                        # print("\n{}\n{}\nGR参数是{}\n".format(p1, p2,GR))
                        print("{}的G1和G2的规划值alpha,beta,gamma,delta,a5,a25和求和值分别是\n{},{}\n{},{}".format(name[:-3],p1,prec1,p2,prec2))
                        print("logg1和logg2分别是 {},{}".format(logG1, logG2))
                        print("复数模量和Delta分别是 {}".format(G,Delta))
                        print("GR参数 {}".format(GR))

                        # 预测与实际拟合情况
                        predy1=predic(p1,logtime)
                        predy2 = predic(p2, logtime)
                        '''
                        x1=np.zeros_like(logtime)
                        x1[:16]=logtime[:16]-p1[4];x1[16:-16]=logtime[16:-16];x1[-16:]=logtime[-16:]-p1[5]
                        plt.scatter(x=x1,y=predy,label=name[:-3]+'fit',marker='*')
                        plt.scatter(x=x1,y=logg1,label=name[:-3]+'origin',marker='^')
                        plt.legend()
                        plt.show()
                        #   结果顺序：alpha, beta, gamma, delta, a5, a25
                        '''
        # 输出
                        r0=np.concatenate([logg1,predy1,logg2,predy2],axis=1)
        # r0=np.concatenate([p1,p2,logG1,logG2,G,Delta,GR],axis=0)
                        print("真实值和预测值拟合效果对比\n{}\n".format(r0))

        # r00=np.array(r00)
        # df1=pd.DataFrame(data=r00,columns=['name','p1','p2','prec1','prec2','logG1','logG2','G','Delta','GR'])
        # df1.to_excel('D:\Desktop\研一\课题组\原始数据\GRres.xlsx',sheet_name='res',index=False )
        # return r0

    #


path='/Users/jasonchan/Desktop/process'
res=cal(path).GR()
print(res)