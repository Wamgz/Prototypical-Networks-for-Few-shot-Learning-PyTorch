import torch
import math
import numpy as np
from visdom import Visdom
import time
torch.__version__


env2 = Visdom()
pane1= env2.line(
    X=torch.FloatTensor([0]),
    Y=torch.FloatTensor([0]),
    opts=dict(title='dynamic data'))

x,y=0,0
for i in range(10):
    time.sleep(1) #每隔一秒钟打印一次数据
    x+=i
    y=(y+i)*1.5
    print(x,y)
    env2.line(
        X=torch.FloatTensor([x]),
        Y=torch.FloatTensor([y]),
        win=pane1,#win参数确认使用哪一个pane
        update='append') #我们做的动作是追加，除了追加意外还有其他方式，这里我们不做介绍了
