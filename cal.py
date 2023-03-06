import numpy as np
import scipy as sp
a=np.array([1,2])
print("求和",np.sum(a))
print("平方和",np.sum( np.square(a) ))
print("平均值",np.average(a))
print("样本方差",np.var(a,ddof=1))
print("总体方差",np.var(a,ddof=0))
print("")



