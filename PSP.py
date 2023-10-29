import gurobipy as gp
import numpy as np

class RCPSP:
    def __init__(self, nodes, inst, sample):
        core=np.loadtxt("Instancias/rcpsp/Datos"+nodes+"/core"+inst+sample+".txt", dtype='int')
        pred=np.loadtxt("Instancias/rcpsp/Datos"+nodes+"/pred"+inst+sample+".txt", dtype='int')
        dur=np.loadtxt("Instancias/rcpsp/Datos"+nodes+"/dura"+inst+sample+".txt", dtype='int')
        recu=np.loadtxt("Instancias/rcpsp/Datos"+nodes+"/recu"+inst+sample+".txt", dtype='int')
        
        J=[dur[i][0] for i in range(len(dur))]
        n=len(J)
        H=pred-1
        d=[dur[i][1] for i in range(len(dur))]
        K=recu[-1][0]
        R=[recu[i][1] for i in range(len(recu))]
        r=[[core[K*i+j][2] for j in range(K)] for i in range(n)]
        es=[0 for i in range(n)]
        Cmax=sum(d)
        ls=[Cmax for i in range(n)]

        self.nP=len(H)
        #Resource factor
        s=0
        for j in range(1,n-1):
            for k in range(K):
                if r[j][k]>0:
                    s+=1

        self.RF=(1/(n-2))*(1/K)*s

        #Resource Strength
        s=0
        self.RS=0
        for k in range(K):
            for j in range(1,n-1):
                s=s+r[j][k]
            s=(1/(n-2))*s
            self.RS=self.RS+R[k]/s

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
        self.gap=self.RTC.MIPGap*100
        self.runtime=self.RTC.Runtime

class RCMPSP_Resource:
    def __init__(self, inst, P, alpha, ResCostType):
        core=[]
        pred=[]
        dur=[]
        recu=[]
        samp=1
        count=0
        Proj=[]
        while count<P:
            sample=str(samp)
            for i in inst:
                if count>=P:
                    break
                instance=str(i)
                Proj.append([i, samp])            
                core.append(np.loadtxt("Instancias/rcpsp/Datos30/core"+instance+sample+".txt", dtype='int'))
                pred.append(np.loadtxt("Instancias/rcpsp/Datos30/pred"+instance+sample+".txt", dtype='int'))
                dur.append(np.loadtxt("Instancias/rcpsp/Datos30/dura"+instance+sample+".txt", dtype='int'))
                recu.append(np.loadtxt("Instancias/rcpsp/Datos30/recu"+instance+sample+".txt", dtype='int'))
                count+=1                
            samp+=1
        CmaxAll=np.loadtxt("Cmax.txt")

        J=[]
        n=[]
        H=[]
        d=[]
        K=0
        r=[]
        es=[]
        ls=[]
        Tmax=0
        for p in range(P):
            K=max(K,recu[p][-1][0]) 
        for p in range(P):
            J.append([dur[p][i][0] for i in range(len(dur[p]))])
            n.append(len(J[p]))
            Hr=pred[p]-1
            d.append([dur[p][i][1] for i in range(len(dur[p]))])    
            r.append([[core[p][K*i+j][2] for j in range(K)] for i in range(n[p])])
            es.append([0 for i in range(n[p])])
            Tmax=Tmax+sum(d[p]) 
            Pr=np.zeros((n[p],n[p]))
            for h in range(len(Hr)):
                Pr[Hr[h,0],Hr[h,1]]=1
            H.append(Pr)
        K=int(K)
        R=[0 for _ in range(K)]
        for k in range(K):
            for p in range(P):        
                R[k]=max(R[k],recu[p][k][1])

        for p in range(P):
            ls.append([Tmax for i in range(n[p])])
        
        Cmax=[]
        for p in Proj:
            Cmax.append(CmaxAll[10*(p[0]-1)+p[1]-1])
        c=[0 for _ in range(K)]
        w=[0 for _ in range(P)]
        Rkp=[[0 for _ in range(K)] for _ in range(P)]
        for p in range(P):
            for i in range(len(r[p])):
                Rkp[p]=[sum(x) for x in zip(Rkp[p],r[p][i])]
        ckp=[[round(10*Rkp[p][k]/min(Rkp[p])) for k in range(K)] for p in range(P)]
        for p in range(P):
            w[p]=sum(x*y for x,y in zip(Rkp[p],ckp[p]))
        
        if ResCostType==1:
            c=[10 for k in range(K)]
        elif ResCostType==2:
            Rk=[0 for _ in range(K)]
            for p in range(P):
                for k in range(K):
                    Rk[k]=Rk[k]+Rkp[p][k]
            c=[round(10*Rk[k]/min(Rkp[p])) for k in range(K)]

        for p in range(P):
            Pr=H[p]

            for j in range(n[p]):
                es[p][j] = 0
                for i in range(j):
                    if Pr[i,j] == 1:
                        if es[p][j] < es[p][i] + d[p][i]:
                            es[p][j] = es[p][i] + d[p][i]

            for i in range(n[p]-1,-1,-1):
                ls[p][i] = Cmax[p]
                for j in range(i+1,n[p]):
                    if Pr[i,j] == 1:
                        if ls[p][i] > ls[p][j] - d[p][i]:
                            ls[p][i] = ls[p][j] - d[p][i]

        Cmax=max([es[p][n[p]-1] for p in range(P)])+alpha*(sum(Cmax)-max([es[p][n[p]-1] for p in range(P)]))

        self.RTC=gp.Model("2_RCMPSP")

        x=self.RTC.addVars([(i,p,j,p2) for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])], vtype=gp.GRB.BINARY, name='x')
        y=self.RTC.addVars([(i,p,j,p2) for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])], vtype=gp.GRB.BINARY, name='y')
        S=self.RTC.addVars([(i,p) for p in range(P)
                                for i in range(n[p])], vtype=gp.GRB.CONTINUOUS, name='S')
        h=self.RTC.addVars(K, vtype=gp.GRB.CONTINUOUS, name='h')

        self.RTC.setObjective(sum(w[p] for p in range(P)) - sum(c[k]*h[k] for k in range(K)), gp.GRB.MAXIMIZE)

        self.RTC.addConstrs((x[i,p,j,p2]+x[j,p2,i,p] <= 1  for p in range(P)
                                            for p2 in range(P)
                                            for i in range(n[p])
                                            for j in range(n[p2])
                                            if (H[p][i,j]==0 and p==p2) or p!=p2))

        self.RTC.addConstrs((S[j,p2]>= S[i,p] + d[p][i]-Cmax*(1-x[i,p,j,p2]) for p in range(P)
                                                            for p2 in range(P)
                                                            for i in range(n[p])
                                                            for j in range(n[p2])
                                                            if i!=j or p!=p2))

        self.RTC.addConstrs((x[i,p,j,p] == 1 for p in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p])
                                        if i<j and H[p][i,j]==1))

        self.RTC.addConstrs((S[0,p] == 0 for p in range(P)))
        self.RTC.addConstrs((S[n[p]-1,p] <= Cmax for p in range(P)))
        self.RTC.addConstrs((S[i,p] + d[p][i] <= Cmax for p in range(P)
                                                    for i in range(n[p])))

        self.RTC.addConstrs((S[i,p] >= es[p][i] for p in range(P)
                                        for i in range(n[p])))
        self.RTC.addConstrs((S[i,p] <= ls[p][i] for p in range(P)
                                        for i in range(n[p])))

        self.RTC.addConstrs((y[i,p,j,p2]<=1-x[i,p,j,p2]-x[j,p2,i,p] for p in range(P)
                                                for p2 in range(P)
                                                for i in range(n[p])
                                                for j in range(n[p2])
                                                if i!=j or p!=p2))

        self.RTC.addConstrs((S[j,p2]>=S[i,p]-Cmax*(1-y[i,p,j,p2])  for p in range(P)
                                                                for p2 in range(P)
                                                                for i in range(n[p])
                                                                for j in range(n[p2])
                                                                if i!=j or p!=p2))

        self.RTC.addConstrs((S[j,p2]<=S[i,p]+d[p][i]+Cmax*(1-y[i,p,j,p2]) for p in range(P)
                                                                    for p2 in range(P)
                                                                    for i in range(n[p])
                                                                    for j in range(n[p2])
                                                                    if i!=j or p!=p2))

        self.RTC.addConstrs((y[i,p,j,p2]+y[j,p2,i,p]+x[i,p,j,p2]+x[j,p2,i,p]>=1  for p in range(P)
                                                        for p2 in range(P)
                                                        for i in range(n[p])
                                                        for j in range(n[p2])
                                                        if i!=j or p!=p2))

        self.RTC.addConstrs((r[p][i][k]+sum(r[p][j][k]*y[j,p,i,p] for j in range(n[p]) if j!=i)
                        + sum(sum(r[p2][j][k]*y[j,p2,i,p] for j in range(n[p2]))for p2 in range(P) if p2!=p)
                        <=R[k] + h[k] for p in range(P)
                                                                                    for i in range(n[p])
                                                                                    for k in range(K)))

        self.RTC.update
        self.RTC.setParam('OutputFlag',False)
        self.RTC.setParam(gp.GRB.Param.TimeLimit,600)

    def solve(self):
        self.RTC.optimize()
        self.objVal=self.RTC.ObjVal
        self.status=self.RTC.Status
        self.gap=self.RTC.MIPGap*100
        self.runtime=self.RTC.Runtime
        self.resources=[]
        for v in self.RTC.getVars():
            if "h" in v.VarName:
                self.resources.append(v.x)

    def write(self, n):
        name=self.RTC.ModelName
        self.RTC.write(name+"_"+str(n)+'.mps')

class RCMPSP_Selection:
    def __init__(self, inst, P, alpha):
        core=[]
        pred=[]
        dur=[]
        recu=[]
        samp=1
        count=0
        Proj=[]
        while count<P:
            sample=str(samp)
            for i in inst:
                if count>=P:
                    break
                instance=str(i)
                Proj.append([i, samp])            
                core.append(np.loadtxt("Instancias/rcpsp/Datos30/core"+instance+sample+".txt", dtype='int'))
                pred.append(np.loadtxt("Instancias/rcpsp/Datos30/pred"+instance+sample+".txt", dtype='int'))
                dur.append(np.loadtxt("Instancias/rcpsp/Datos30/dura"+instance+sample+".txt", dtype='int'))
                recu.append(np.loadtxt("Instancias/rcpsp/Datos30/recu"+instance+sample+".txt", dtype='int'))
                count+=1                
            samp+=1
        CmaxAll=np.loadtxt("Cmax.txt")

        J=[]
        n=[]
        H=[]
        d=[]
        K=0
        r=[]
        es=[]
        ls=[]
        Tmax=0
        for p in range(P):
            K=max(K,recu[p][-1][0]) 
        for p in range(P):
            J.append([dur[p][i][0] for i in range(len(dur[p]))])
            n.append(len(J[p]))
            Hr=pred[p]-1
            d.append([dur[p][i][1] for i in range(len(dur[p]))])    
            r.append([[core[p][4*i+j][2] for j in range(K)] for i in range(n[p])])
            es.append([0 for i in range(n[p])])
            Tmax=Tmax+sum(d[p]) 
            Pr=np.zeros((n[p],n[p]))
            for h in range(len(Hr)):
                Pr[Hr[h,0],Hr[h,1]]=1
            H.append(Pr)
        K=int(K)
        R=[0 for _ in range(K)]
        for k in range(K):
            for p in range(P):        
                R[k]=max(R[k],recu[p][k][1])


        for p in range(P):
            ls.append([Tmax for i in range(n[p])])
                
        Cmax=[]
        for p in Proj:
            Cmax.append(CmaxAll[10*(p[0]-1)+p[1]-1])
        c=[0 for _ in range(K)]
        w=[0 for _ in range(P)]
        Rkp=[[0 for _ in range(K)] for _ in range(P)]
        for p in range(P):
            for i in range(len(r[p])):
                Rkp[p]=[sum(x) for x in zip(Rkp[p],r[p][i])]
        ckp=[[round(10*Rkp[p][k]/min(Rkp[p])) for k in range(K)] for p in range(P)]
        for p in range(P):
            w[p]=sum(x*y for x,y in zip(Rkp[p],ckp[p]))

        Rk=[0 for _ in range(K)]
        for p in range(P):
            for k in range(K):
                Rk[k]=Rk[k]+Rkp[p][k]
        c=[round(10*Rk[k]/min(Rkp[p])) for k in range(K)]

        for p in range(P):
            Pr=H[p]

            for j in range(n[p]):
                es[p][j] = 0
                for i in range(j):
                    if Pr[i,j] == 1:
                        if es[p][j] < es[p][i] + d[p][i]:
                            es[p][j] = es[p][i] + d[p][i]

            for i in range(n[p]-1,-1,-1):
                for j in range(i+1,n[p]):
                    if Pr[i,j] == 1:
                        if ls[p][i] > ls[p][j] - d[p][i]:
                            ls[p][i] = ls[p][j] - d[p][i]

        Cmax=max(Cmax)+alpha*(sum(Cmax)-max(Cmax))

        self.RTC=gp.Model("3_RCMPSP")

        x=self.RTC.addVars([(i,p,j,p2) for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])], vtype=gp.GRB.BINARY, name='x')
        y=self.RTC.addVars([(i,p,j,p2) for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])], vtype=gp.GRB.BINARY, name='y')
        S=self.RTC.addVars([(i,p) for p in range(P)
                                for i in range(n[p])], vtype=gp.GRB.CONTINUOUS, name='S')
        z=self.RTC.addVars(P, vtype=gp.GRB.BINARY, name='z')

        self.RTC.setObjective(sum(w[p]*z[p] for p in range(P)) - sum(c[k] for k in range(K)), gp.GRB.MAXIMIZE)

        self.RTC.addConstrs((x[i,p,j,p2]+x[j,p2,i,p] <= z[p]  for p in range(P)
                                            for p2 in range(P)
                                            for i in range(n[p])
                                            for j in range(n[p2])
                                            if (H[p][i,j]==0 and p==p2) or p!=p2))
        self.RTC.addConstrs((x[i,p,j,p2]+x[j,p2,i,p] <= z[p2]  for p in range(P)
                                            for p2 in range(P)
                                            for i in range(n[p])
                                            for j in range(n[p2])
                                            if (H[p][i,j]==0 and p==p2) or p!=p2))

        self.RTC.addConstrs((S[j,p2]>= S[i,p] + d[p][i]-Cmax*(1-x[i,p,j,p2]) for p in range(P)
                                                            for p2 in range(P)
                                                            for i in range(n[p])
                                                            for j in range(n[p2])
                                                            if i!=j or p!=p2))

        self.RTC.addConstrs((x[i,p,j,p] == z[p] for p in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p])
                                        if i<j and H[p][i,j]==1))

        self.RTC.addConstrs((S[0,p] == 0 for p in range(P)))
        self.RTC.addConstrs((S[n[p]-1,p] <= Cmax for p in range(P)))
        self.RTC.addConstrs((S[i,p] + d[p][i] <= Cmax for p in range(P)
                                                    for i in range(n[p])))

        self.RTC.addConstrs((S[i,p] >= es[p][i]*z[p] for p in range(P)
                                        for i in range(n[p])))
        self.RTC.addConstrs((S[i,p] <= ls[p][i]*z[p] for p in range(P)
                                        for i in range(n[p])))

        self.RTC.addConstrs((y[i,p,j,p2]<=z[p]-x[i,p,j,p2]-x[j,p2,i,p] for p in range(P)
                                                for p2 in range(P)
                                                for i in range(n[p])
                                                for j in range(n[p2])
                                                if i!=j or p!=p2))

        self.RTC.addConstrs((y[i,p,j,p2]<=z[p] for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])
                                        if i!=j or p!=p2))
        self.RTC.addConstrs((y[i,p,j,p2]<=z[p2] for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])
                                        if i!=j or p!=p2))

        self.RTC.addConstrs((S[j,p2]>=S[i,p]-Cmax*(1-y[i,p,j,p2])  for p in range(P)
                                                                for p2 in range(P)
                                                                for i in range(n[p])
                                                                for j in range(n[p2])
                                                                if i!=j or p!=p2))

        self.RTC.addConstrs((S[j,p2]<=S[i,p]+d[p][i]+Cmax*(1-y[i,p,j,p2]) for p in range(P)
                                                                    for p2 in range(P)
                                                                    for i in range(n[p])
                                                                    for j in range(n[p2])
                                                                    if i!=j or p!=p2))

        self.RTC.addConstrs((y[i,p,j,p2]+y[j,p2,i,p]+x[i,p,j,p2]+x[j,p2,i,p]>=-1+z[p]+z[p2]  for p in range(P)
                                                        for p2 in range(P)
                                                        for i in range(n[p])
                                                        for j in range(n[p2])
                                                        if i!=j or p!=p2))

        self.RTC.addConstrs((r[p][i][k]+sum(r[p][j][k]*y[j,p,i,p] for j in range(n[p]) if j!=i)
                        + sum(sum(r[p2][j][k]*y[j,p2,i,p] for j in range(n[p2]))for p2 in range(P) if p2!=p)
                        <=R[k] for p in range(P)
                                                                                    for i in range(n[p])
                                                                                    for k in range(K)))

        self.RTC.update
        self.RTC.setParam('OutputFlag',False)
        self.RTC.setParam(gp.GRB.Param.TimeLimit,600)

    def solve(self):
        self.RTC.optimize()
        self.objVal=self.RTC.ObjVal
        self.status=self.RTC.Status
        self.gap=self.RTC.MIPGap*100
        self.runtime=self.RTC.Runtime
        self.projects=[]
        for v in self.RTC.getVars():
            if "z" in v.VarName:
                self.projects.append(v.x)
    
    def write(self, n):
        name=self.RTC.ModelName
        self.RTC.write(name+"_"+str(n)+'.mps')

class RCMPSP_Mix:
    def __init__(self, inst, P, alpha, ResCostType):
        core=[]
        pred=[]
        dur=[]
        recu=[]
        for i in range(P):
            sample=str(i+1)
            core.append(np.loadtxt("Instancias/rcpsp/Datos30/core"+inst+sample+".txt", dtype='int'))
            pred.append(np.loadtxt("Instancias/rcpsp/Datos30/pred"+inst+sample+".txt", dtype='int'))
            dur.append(np.loadtxt("Instancias/rcpsp/Datos30/dura"+inst+sample+".txt", dtype='int'))
            recu.append(np.loadtxt("Instancias/rcpsp/Datos30/recu"+inst+sample+".txt", dtype='int'))
        CmaxAll=np.loadtxt("Cmax.txt")

        J=[]
        n=[]
        H=[]
        d=[]
        K=0
        r=[]
        es=[]
        ls=[]
        Tmax=0
        for p in range(P):
            K=max(K,recu[p][-1][0]) 
        for p in range(P):
            J.append([dur[p][i][0] for i in range(len(dur[p]))])
            n.append(len(J[p]))
            Hr=pred[p]-1
            d.append([dur[p][i][1] for i in range(len(dur[p]))])    
            r.append([[core[p][4*i+j][2] for j in range(K)] for i in range(n[p])])
            es.append([0 for i in range(n[p])])
            Tmax=Tmax+sum(d[p]) 
            Pr=np.zeros((n[p],n[p]))
            for h in range(len(Hr)):
                Pr[Hr[h,0],Hr[h,1]]=1
            H.append(Pr)
        K=int(K)
        R=[0 for _ in range(K)]
        for k in range(K):
            for p in range(P):        
                R[k]=max(R[k],recu[p][k][1])

        Cmax=[]
        for p in range(P):
            ls.append([Tmax for i in range(n[p])])

        Cmax=[]
        for p in range(P):
            Cmax.append(CmaxAll[10*(int(inst)-1)+p])
        c=[0 for _ in range(K)]
        w=[0 for _ in range(P)]
        Rkp=[[0 for _ in range(K)] for _ in range(P)]
        for p in range(P):
            for i in range(len(r[p])):
                Rkp[p]=[sum(x) for x in zip(Rkp[p],r[p][i])]
        ckp=[[round(10*Rkp[p][k]/min(Rkp[p])) for k in range(K)] for p in range(P)]
        for p in range(P):
            w[p]=sum(x*y for x,y in zip(Rkp[p],ckp[p]))
        if ResCostType==1:
            c=[10 for k in range(K)]
        elif ResCostType==2:
            Rk=[0 for _ in range(K)]
            for p in range(P):
                for k in range(K):
                    Rk[k]=Rk[k]+Rkp[p][k]
            c=[round(10*Rk[k]/min(Rkp[p])) for k in range(K)]


        for p in range(P):
            Pr=H[p]

            for j in range(n[p]):
                es[p][j] = 0
                for i in range(j):
                    if Pr[i,j] == 1:
                        if es[p][j] < es[p][i] + d[p][i]:
                            es[p][j] = es[p][i] + d[p][i]

            for i in range(n[p]-1,-1,-1):
                for j in range(i+1,n[p]):
                    if Pr[i,j] == 1:
                        if ls[p][i] > ls[p][j] - d[p][i]:
                            ls[p][i] = ls[p][j] - d[p][i]

        Cmax=max([es[p][n[p]-1] for p in range(P)])+alpha*(sum(Cmax)-max([es[p][n[p]-1] for p in range(P)]))

        self.RTC=gp.Model("Resource Constrained Project Scheduling Problem")

        x=self.RTC.addVars([(i,p,j,p2) for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])], vtype=gp.GRB.BINARY, name='x')
        y=self.RTC.addVars([(i,p,j,p2) for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])], vtype=gp.GRB.BINARY, name='y')
        S=self.RTC.addVars([(i,p) for p in range(P)
                                for i in range(n[p])], vtype=gp.GRB.CONTINUOUS, name='S')
        z=self.RTC.addVars(P, vtype=gp.GRB.BINARY, name='z')
        h=self.RTC.addVars(K, vtype=gp.GRB.CONTINUOUS, name='h')

        self.RTC.setObjective(sum(w[p]*z[p] for p in range(P)) - sum(c[k]*h[k] for k in range(K)), gp.GRB.MAXIMIZE)

        self.RTC.addConstrs((x[i,p,j,p2]+x[j,p2,i,p] <= z[p]  for p in range(P)
                                            for p2 in range(P)
                                            for i in range(n[p])
                                            for j in range(n[p2])
                                            if (H[p][i,j]==0 and p==p2) or p!=p2))
        self.RTC.addConstrs((x[i,p,j,p2]+x[j,p2,i,p] <= z[p2]  for p in range(P)
                                            for p2 in range(P)
                                            for i in range(n[p])
                                            for j in range(n[p2])
                                            if (H[p][i,j]==0 and p==p2) or p!=p2))

        self.RTC.addConstrs((S[j,p2]>= S[i,p] + d[p][i]-Cmax*(1-x[i,p,j,p2]) for p in range(P)
                                                            for p2 in range(P)
                                                            for i in range(n[p])
                                                            for j in range(n[p2])
                                                            if i!=j or p!=p2))

        self.RTC.addConstrs((x[i,p,j,p] == z[p] for p in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p])
                                        if i<j and H[p][i,j]==1))

        self.RTC.addConstrs((S[0,p] == 0 for p in range(P)))
        self.RTC.addConstrs((S[n[p]-1,p] <= Cmax for p in range(P)))
        self.RTC.addConstrs((S[i,p] + d[p][i] <= Cmax for p in range(P)
                                                    for i in range(n[p])))

        self.RTC.addConstrs((S[i,p] >= es[p][i]*z[p] for p in range(P)
                                        for i in range(n[p])))
        self.RTC.addConstrs((S[i,p] <= ls[p][i]*z[p] for p in range(P)
                                        for i in range(n[p])))

        self.RTC.addConstrs((y[i,p,j,p2]<=z[p]-x[i,p,j,p2]-x[j,p2,i,p] for p in range(P)
                                                for p2 in range(P)
                                                for i in range(n[p])
                                                for j in range(n[p2])
                                                if i!=j or p!=p2))

        self.RTC.addConstrs((y[i,p,j,p2]<=z[p] for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])
                                        if i!=j or p!=p2))
        self.RTC.addConstrs((y[i,p,j,p2]<=z[p2] for p in range(P)
                                        for p2 in range(P)
                                        for i in range(n[p])
                                        for j in range(n[p2])
                                        if i!=j or p!=p2))

        self.RTC.addConstrs((S[j,p2]>=S[i,p]-Cmax*(1-y[i,p,j,p2])  for p in range(P)
                                                                for p2 in range(P)
                                                                for i in range(n[p])
                                                                for j in range(n[p2])
                                                                if i!=j or p!=p2))

        self.RTC.addConstrs((S[j,p2]<=S[i,p]+d[p][i]+Cmax*(1-y[i,p,j,p2]) for p in range(P)
                                                                    for p2 in range(P)
                                                                    for i in range(n[p])
                                                                    for j in range(n[p2])
                                                                    if i!=j or p!=p2))

        self.RTC.addConstrs((y[i,p,j,p2]+y[j,p2,i,p]+x[i,p,j,p2]+x[j,p2,i,p]>=-1+z[p]+z[p2]  for p in range(P)
                                                        for p2 in range(P)
                                                        for i in range(n[p])
                                                        for j in range(n[p2])
                                                        if i!=j or p!=p2))

        self.RTC.addConstrs((r[p][i][k]+sum(r[p][j][k]*y[j,p,i,p] for j in range(n[p]) if j!=i)
                        + sum(sum(r[p2][j][k]*y[j,p2,i,p] for j in range(n[p2]))for p2 in range(P) if p2!=p)
                        <=R[k] + h[k] for p in range(P)
                                                                                    for i in range(n[p])
                                                                                    for k in range(K)))

        self.RTC.update
        self.RTC.setParam('OutputFlag',False)
        self.RTC.setParam(gp.GRB.Param.TimeLimit,600)
        
    def solve(self):
        self.RTC.optimize()
        self.objVal=self.RTC.ObjVal
        self.status=self.RTC.Status
        self.gap=self.RTC.MIPGap*100
        self.runtime=self.RTC.Runtime
        self.projects=[]
        self.resources=[]
        for v in self.RTC.getVars():
            if "z" in v.VarName:
                self.projects.append(v.x)
            if "h" in v.VarName:
                self.resources.append(v.x)
    
