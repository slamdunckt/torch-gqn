import numpy as np

patch_label =[];
for i in range(64):
    ttt=[]
    for j in range(8):
        tt=[]
        for k in range(8):
            t = [i//8+1, i%8+1]
            tt.append(t)
        ttt.append(tt)
    patch_label.append(ttt)
print(np.shape(patch_label)[0])
