import gurobipy as gp
import numpy as np

class RCPSP:
    def __init__(self, inst, sample):
        core=np.loadtxt("Instancias/rcpsp/Datos30/core"+inst+sample+".txt", dtype='int')
        pred=np.loadtxt("Instancias/rcpsp/Datos30/pred"+inst+sample+".txt", dtype='int')
        dur=np.loadtxt("Instancias/rcpsp/Datos30/dura"+inst+sample+".txt", dtype='int')
        recu=np.loadtxt("Instancias/rcpsp/Datos30/recu"+inst+sample+".txt", dtype='int')
        
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

        self.RTC=gp.Model("Resource Constrained Project Scheduling Problem")

        x=self.RTC.addVars(n,n, vtype=gp.GRB.BINARY, name='x')
        y=self.RTC.addVars(n,n, vtype=gp.GRB.BINARY, name='y')
        S=self.RTC.addVars(n, vtype=gp.GRB.CONTINUOUS, name='S')

        self.RTC.setObjective(S[n-1], gp.GRB.MINIMIZE)

        self.RTC.addConstrs((x[i,j]+x[j,i] <= 1  for i in range(n)
                                            for j in range(n)
                                            if i<j))

        self.RTC.addConstrs((S[j]>= S[i] + d[i]-Cmax*(1-x[i,j])   for i in range(n)
                                                            for j in range(n)))

        self.RTC.addConstrs((x[i,j] == 1 for i,j in H))

        self.RTC.addConstr(S[0] == 0)

        self.RTC.addConstrs((S[i] >= es[i] for i in range(n)))
        self.RTC.addConstrs((S[i] <= ls[i] for i in range(n)))

        self.RTC.addConstrs((y[i,j]<=1-x[i,j]-x[j,i] for i in range(n)
                                                for j in range(n)))

        self.RTC.addConstrs((S[i]<=S[j]+Cmax*(1-y[i,j])  for i in range(n)
                                                    for j in range(n)))

        self.RTC.addConstrs((S[i]+d[i]>=S[j]-Cmax*(1-y[i,j]) for i in range(n)
                                                        for j in range(n)))

        self.RTC.addConstrs((y[i,j]+y[j,i]+x[i,j]+x[j,i]>=1  for i in range(n)
                                                        for j in range(n)
                                                        if i<j))

        self.RTC.addConstrs((r[i][k]+sum(r[j][k]*y[j,i] for j in range(n) if j!=i)<=R[k] for i in range(n)
                                                                                    for k in range(K)))
        
        self.RTC.update
        self.RTC.setParam('OutputFlag',False)
        self.RTC.setParam(gp.GRB.Param.TimeLimit,600)

    def solve(self):
        self.RTC.optimize()
        self.objVal=self.RTC.ObjVal
        self.status=self.RTC.Status
        self.gap=self.RTC.MIPGap
        self.runtime=self.RTC.Runtime