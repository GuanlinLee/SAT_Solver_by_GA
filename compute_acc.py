import numpy as np
import numba
@numba.jit
def acc_compute(size_of_C,k_sat,C_np,X):
    acc=[]
    #print(X)
    ones=np.ones(len(X))
    for i in range(size_of_C):
        x_list=[]
        not_list=[]
        ci_np=C_np[i]
        #print(ci_np)
        for k in range(k_sat):
            #print(ci_np[k]-1)
            x_list.append(int(X[int(ci_np[k]-1)]))
            if ci_np[k_sat+k]==1.0:
                not_list.append(k)
        ci_val=0
        for p in range(k_sat):
            if p in not_list:
                x_list[p]=not x_list[p]
        for x in x_list:
            ci_val=ci_val or x
        if ci_val!=1:
            for k in range(k_sat):
                ones[int(ci_np[k]-1)]=0 #(ones[int(ci_np[k]-1)]+1)%2
        acc.append(ci_val)

    acc_np=np.array(acc)
    return np.mean(ones)