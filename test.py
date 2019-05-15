import numpy as np

patch_label =[];
for i in range(64):
    tttt=[]
    for l in range(2):
        ttt=[]
        for j in range(8):
            tt=[]
            for k in range(8):
                if(l==0): t = k
                else : t=l
                tt.append(t)
            ttt.append(tt)
        tttt.append(ttt)
    patch_label.append(tttt)
print(np.shape(patch_label))
