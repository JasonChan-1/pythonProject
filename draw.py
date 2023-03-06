import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

# read and process data
data = pd.read_csv(r'D:\Desktop\output\process\process.csv', sep=',', header=0, encoding='gbk').dropna()
x1 = data.values[1:, 0]; y1 = data.values[1:, 2]
x2 = data.values[1:, 0]; y2 = data.values[1:, 3]
# a1 = data[data['column'] == 'value1']
# a2 = data[data['column'] == 'value2']

# initialization
font = font_manager.FontProperties('Serif',size=12)
font_ticks=font_manager.FontProperties('Serif')
plt.rcParams['mathtext.fontset']='stix' #设置公式字体为近似罗马字体
# plt.rcParams['font.serif']='SimSun'，设置为宋体
plt.figure(figsize=(6, 4))
plt.scatter(x1, y1, s=3, marker='^', color=['#f23b27'], label='pe', alpha=0.8)# reference_color #e9963e,#f23b27, #65a9d7, #304f9e
plt.scatter(x2, y2, s=3, marker='*', color=['#e9963e'], label='te', alpha=0.8)
ax=plt.gca()
plt.title('energy-timesteps', fontproperties=font)
plt.legend(prop=font, bbox_to_anchor=(0.95, 0.7), frameon=1, markerscale=5, ncol=2)

# 坐标轴
"""
设置坐标轴位置及隐藏坐标轴
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',120))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',5e4))
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
"""
# 设定刻度范围及数据
plt.xlim(0, 101e3)
plt.ylim(0, 240)
plt.xticks(np.arange(0, 105e3, 5000),fontproperties=font_ticks,rotation=0,minor=True)
plt.yticks(np.arange(0, 245, 10),fontproperties=font_ticks,rotation=0,minor=True)

# 设定刻度间隔
xmajor_locator=MultipleLocator(1e4); xminor_locator=MultipleLocator(2000)
ymajor_locator=MultipleLocator(30); yminor_locator=MultipleLocator(10)
ax.xaxis.set_major_locator(xmajor_locator)
ax.xaxis.set_minor_locator(xminor_locator)
ax.yaxis.set_major_locator(ymajor_locator)
ax.yaxis.set_minor_locator(yminor_locator)

# 设定刻度格式
lbsize=10;      majorlen=10;     minorlen=5;     wd=0.5
plt.tick_params(axis='x', direction='in',which="minor",length=minorlen,width=wd)    #x轴次轴
plt.tick_params(axis='x', direction='in',which='major',labelsize=lbsize,length=majorlen,width=wd)   #x轴主轴
plt.tick_params(axis='y', direction='in',which='minor',length=minorlen,width=wd)    #y轴次轴
plt.tick_params(axis='y', direction='in',which='major',labelsize=lbsize,length=majorlen,width=wd)   #y轴主轴

plt.ticklabel_format(style='sci',scilimits=(5,6),axis='x')
plt.xlabel('timesteps(fs)', fontproperties=font)
plt.ylabel(r'$energy(Kcal/mol)$', fontproperties=font)
# 希腊字母为Latex格式，e.g.r'$\Delta$'

# picture and text
# plt.grid()
plt.show()