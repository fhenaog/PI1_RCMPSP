import gurobipy as gp
import numpy as np
import time

start=time.time()
opt=0
gaps=[]
#objs=[]
for m in range(48,49):
    for n in range(1,11):
        inst=str(m)+str(n)

        core=np.loadtxt("Instancias/rcpsp/Datos30/core"+inst+".txt", dtype='int')
        pred=np.loadtxt("Instancias/rcpsp/Datos30/pred"+inst+".txt", dtype='int')
        dur=np.loadtxt("Instancias/rcpsp/Datos30/dura"+inst+".txt", dtype='int')
        recu=np.loadtxt("Instancias/rcpsp/Datos30/recu"+inst+".txt", dtype='int')

        J=[dur[i][0] for i in range(len(dur))]
        n=len(J)
        H=pred-1
        d=[dur[i][1] for i in range(len(dur))]
        K=recu[-1][0]
        R=[recu[i][1] for i in range(len(recu))]
        r=[[core[4*i+j][2] for j in range(K)] for i in range(n)]
        es=[0 for i in range(n)]
        Cmax=sum(d)
        ls=[Cmax for i in range(n)]

        P=np.zeros((n,n))
        for h in range(len(H)):
            P[H[h,0],H[h,1]]=1

        for j in range(n):
            es[j] = 0
            for i in range(j):
                if P[i,j] == 1:
                    if es[j] < es[i] + d[i]:
                        es[j] = es[i] + d[i]

        for i in range(n-1,-1,-1):
            ls[i] = Cmax
            for j in range(i+1,n):
                if P[i,j] == 1:
                    if ls[i] > ls[j] - d[i]:
                        ls[i] = ls[j] - d[i]

        RTC=gp.Model("Resource Constrained Project Scheduling Problem")

        x=RTC.addVars(n,n, vtype=gp.GRB.BINARY, name='x')
        y=RTC.addVars(n,n, vtype=gp.GRB.BINARY, name='y')
        S=RTC.addVars(n, vtype=gp.GRB.CONTINUOUS, name='S')

        RTC.setObjective(S[n-1], gp.GRB.MINIMIZE)

        RTC.addConstrs((x[i,j]+x[j,i] <= 1  for i in range(n)
                                            for j in range(n)
                                            if i<j))

        RTC.addConstrs((S[j]>= S[i] + d[i]-Cmax*(1-x[i,j])   for i in range(n)
                                                            for j in range(n)))

        RTC.addConstrs((x[i,j] == 1 for i,j in H))

        RTC.addConstr(S[0] == 0)

        RTC.addConstrs((S[i] >= es[i] for i in range(n)))
        RTC.addConstrs((S[i] <= ls[i] for i in range(n)))

        RTC.addConstrs((y[i,j]<=1-x[i,j]-x[j,i] for i in range(n)
                                                for j in range(n)))

        RTC.addConstrs((S[i]<=S[j]+Cmax*(1-y[i,j])  for i in range(n)
                                                    for j in range(n)))

        RTC.addConstrs((S[i]+d[i]>=S[j]-Cmax*(1-y[i,j]) for i in range(n)
                                                        for j in range(n)))

        RTC.addConstrs((y[i,j]+y[j,i]+x[i,j]+x[j,i]>=1  for i in range(n)
                                                        for j in range(n)
                                                        if i<j))

        RTC.addConstrs((r[i][k]+sum(r[j][k]*y[j,i] for j in range(n) if j!=i)<=R[k] for i in range(n)
                                                                                    for k in range(K)))
        
        RTC.update
        RTC.setParam('OutputFlag',False)
        RTC.setParam(gp.GRB.Param.TimeLimit,600)
        RTC.optimize()
        if RTC.Status == gp.GRB.OPTIMAL:
            opt+=1
        else:
            gaps=gaps+[RTC.MIPGap]

print("Number of optimal results = ",str(opt))
print("Number of non-optimal results = ",str(480-opt))
print("Running time = ", time.time()-start)
print("Gaps ", gaps)
        