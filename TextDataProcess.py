#python处理文本数据,正则表达式提取只含数字和制表位的数据
import numpy as np
import csv
import os

def csv_save(arr,save_path):
    """
    save_path, e.g.,'/Users/jasonchan/Desktop/1.csv'
    save np.array as 'csv' of the form for the file
    """
    with open(save_path, "w", newline='') as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow(row)

def extract(folder_path,folder_name,start,stop):
    """
    both start and stop is the str for readline, search by str.
    """
    file_path=os.path.join(folder_path,folder_name)
    files=os.listdir(file_path)
    for file_name in files:
        file=os.path.join(file_path,file_name)
        with open(file,'r') as f:
            f.seek(0,0)
            text=f.read()
            ls_data=re.findall(start+'(.*)'+stop,text,re.S)
            str_data=re.sub(r'[^\d\s]','',ls_data[0])
            data=str_data.strip().split('\n')
            first_ele = True
            for line in data:
                line=line.strip().split()
                if first_ele:
                    arr0=np.array(line)
                    first_ele=False
                else:
                    arr0=np.c_[arr0,line]
            arr0=arr0.transpose()
            a=[]
            for i in range(1,len(arr0)):
                result=[ float(j) for j in arr0[i]]
                a.append(result)
                arr=np.array(a,dtype=object)
        save_path = os.path.join(folder_path, file_name + '.csv')
        with open(save_path, "w", newline='') as f1:
            writer = csv.writer(f1)
            for row in arr:
                writer.writerow(row)
    return arr

if __name__ == '__main__':
    start='PotEng         TotEng'
    stop='WARNING: Bond/angle/dihedral extent'
    folder_path='D:\Desktop\output'
    folder_name='process'
    arr=extract(folder_path,folder_name,start,stop)
    print(arr)






