import pandas as pd
import lxml # pd.read_html需要安装的包
import html5lib # pd.read_html需要安装的包
import urllib.parse
import numpy as np



class markov():
    def get_data(self):
        citys = ['遵义', '北京',  '上海', '广州', '深圳', '南宁', '昆明', '长沙', '西安', '武汉']
        url0 = 'https://www.chalieche.com/search/range/?'
        statis=np.zeros([len(citys),len(citys)])
        # columns ['车次' '列车类型' '始发站' '始发时间' '经过站' '经过站 到达时间' '经过站 发车时间' '终点站' '到达时间']
        for city1 in citys:
            for city2 in citys:
                if city1 == city2:
                    statis[citys.index(city1), citys.index(city2)] = 1
                else:
                    payload={"from":city1,"to":city2}
                    url = url0 + str(urllib.parse.urlencode(payload))
                    data = pd.read_html(io=url, header=0)
                    df = pd.DataFrame(data[0])
                    arr=df.values
                    statis[citys.index(city1),citys.index(city2)]=arr.shape[0]
        with open("D:\Desktop\city_train_data.txt",mode='w+') as f:
            f.write(str(df.columns\n)+str(statis))

'''
data=np.array([[  1.   2.   2.  17.   1.  11.  14.  10.   5.   2.]
[  2.   1.  52.  17.  11.  11.   5.  47.  31.  33.]
 [  4.  50.   1.  17.  26.  10.   8.  46.  60.  24.]
 [ 16.  18.  18.   1. 451. 143.  51. 243.  32. 103.]
 [  1.  11.  25. 432.   1.  17.   4.  99.  18.  48.]
 [ 12.  12.  10. 162.  19.   1. 163.  30.   1.   9.]
 [ 15.   5.   7.  47.   5. 159.   1.  28.   3.   6.]
 [ 15.  52.  46. 235. 109.  28.  28.   1.  32. 148.]
 [  3.  30.  61.  33.  20.   1.   3.  33.   1.  30.]
 [  2.  33.  28. 107.  52.   9.   6. 150.  30.   1.]])
print(data)
'''
m=markov().get_data()
print(m)