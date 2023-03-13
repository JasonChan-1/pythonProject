import requests
import requests.utils
import json
head={'User-Agnet': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4882.400 QQBrowser/9.7.13059.400',
'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
'accept-language': "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
"Accept-Encoding": "gzip, deflate, br",
'Cache-Control':'max-age=0',
'Connection':'keep-alive',
'Referer':'http://www.baidu.com/'}   # 'Connection': 'keep-alive''content-type': 'application/json'

wd=['语文','数学','英语','物理','化学']
url='https://www.baidu.com/s?wd='+wd[3]     #网站域名
data = {"key1":"value1","key2":"value2"}    #post表单数据
s=requests.session()                        #创建会话
# res = s.get(url=url, headers=head)
# print(res.status_code)
# print(res.content)
# print(res.cookies)
res1=s.get(url,headers=head)                #get获取网站信息
# res1=s.post(url,headers=head)             #post~
c=requests.utils.dict_from_cookiejar(res1.cookies)  #cookieJar保存为字典
print('网页cookieshi是：{}'.format(c))
# print(res.text)
# s=requests.session()
# data=s.get(url=url,headers=head)
# path='D:\Desktop\data.txt'
# with open(path,'w+') as f:
#         f.writelines(data.cookies)
'''
'''

