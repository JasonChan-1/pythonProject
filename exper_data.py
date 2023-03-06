import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy import optimize as op
import math


def f_cm(x, k):
    a=x*np.emath.power(10,[k])[0]
    return a

def sigmoid(p,x,y):
    a5, alpha, beta, gamma, delta=p

    return (y-delta-( alpha/(1+np.exp(beta+gamma*(x-a5) )) ))**2 #x表示logtime


class cal():
    def __init__(self,path):
        self.files=os.listdir(path)
        self.sheets_name=[]
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
        marker=['>','*','^','o','+','<','>','*','^','o','+','<']
        color=['#e9963e','#f23b27', '#65a9d7', '#304f9e','#83639f','#ea7827','#c22f2f','#449945'
               '#e9963e', '#f23b27', '#65a9d7','#304f9e', '#83639f', '#ea7827', '#c22f2f', '#449945']
        k=[[0.7,-0.7],[0.65,-0.65],[0.7,-0.7],[0.7,-0.7],[0.85,-0.9],[0.8,-0.8],[0.85,-0.9],[0.85, -0.9],
           [0.85, -0.9], [0.8, -0.8], [0.85, -0.9], [0.85, -0.9], [0.85, -0.9], [0.85, -0.9],]  #调参
        self.graph()
        for file in self.files:
            if re.search(r'\W', file.replace('.', '')) == None:
                file_path = os.path.join(path, file)
                for fushu_name in self.fushu:
                    if fushu_name in pd.read_excel(file_path, sheet_name=None):
                        i = i + 1
                        df_fushu = pd.read_excel(file_path, sheet_name=fushu_name)
                        ar_fushu = df_fushu[['Temperature', 'Angular frequency', 'Complex modulus']][1:].values
                        ar_fushu[:, 2] = ar_fushu[:, 2] / 1e3
                        ar35 = []; ar45 = []; ar55 = []
                        for row in ar_fushu:
                            if round(row[0]) == 35:
                                ar35.append(row[-2:])
                            elif round(row[0]) == 45:
                                ar45.append(row[-2:])
                            elif round(row[0]) == 55:
                                ar55.append(row[-2:])
                        ar35 = np.array(ar35)
                        ar45 = np.array(ar45)
                        ar55 = np.array(ar55)
                        for j in range(len(ar35)):
                            ar35[j, 0] = f_cm(ar35[j, 0], k=k[i][0])
                        for j in range(len(ar55)):
                            ar55[j, 0] = f_cm(ar55[j, 0], k=k[i][1])
                        arr = np.concatenate((ar35, ar45, ar55), axis=0)
                        plt.scatter(x=arr[:, 0], y=arr[:, 1],
                                    label=fushu_name[:-5], marker=marker[i], color=color[i],
                                    alpha=0.8)
                        plt.legend()
        plt.show()
    # 车辙因子
    def rf(self):
        i=-1
        plt.figure(figsize=(5,5))
        plt.xticks(np.arange(0,100,10))
        plt.yticks(np.arange(0,200,50))
        plt.tick_params(direction='in', axis='both',length=8,width=1)
        plt.xlabel(r'$T(℃)kPa$')
        plt.ylabel(r'$G*/sin\delta$')
        marker = ['>', '*','v','x','^', 'o', '+', '<', '>', '*', '^', 'o', '+', '<']
        color = ['#e9963e', '#f23b27', '#65a9d7', '#304f9e', '#83639f',
                 '#ea7827', '#c22f2f', '#449945','#e9963e', '#f23b27',
                 '#65a9d7', '#304f9e', '#83639f', '#ea7827', '#c22f2f', '#449945']
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
                            logtime[j,0]=round(np.emath.logn(10,time),2)
                            logg1[j,0]=round(np.emath.logn(10,ar[j,0]),2)
                            logg2[j,0]=round(np.emath.logn(10,ar[j,1]),2)
                        p0=[15,0.1,-5,0.1,-1]
                        fit2=op.least_squares(sigmoid,p0,args=(logtime[:,0],logg2))
                        fit2=op.curve_fit(self.sigmoid,logtime[:,0],logg2,absolute_sigma=True)


        # return fit2



path='D:\Desktop\研一\课题组\原始数据'
res=cal(path).rf()
print(res)
