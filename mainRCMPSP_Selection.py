import time
from PSP import RCMPSP_Selection

start=time.time()
opt=0
gaps=[]
nonOpt=[]
nonOptStatus=[]
objs=[]
NoProj=[2,3,5,10] #Number of projects
alphaVal=[0.0,0.25,0.5,0.75,1.0] #alpha values
instances=[[13],[48],[29],[24],[13,48,29,24]] #instances (last one is for mixing)
result="w" # "s" --- "w" to write the problem, "s" to solve the problem
n=1
with open("resultsSelection.txt", "w") as res:
    res.write(f"{'Instance' : ^10}{'No. Proj' : ^10}{'Alpha' : ^10}{'Objective' : ^15}{'Gap' : ^10}{'Runtime' : ^10}{'Projects' : ^30}")
    res.write('\n')
    res.write(''.join(['-' for _ in range(95)]))
    res.write('\n')
    for inst in instances:
        for p in NoProj:
                for alpha in alphaVal:
                    
                    problem=RCMPSP_Selection(inst,p,alpha)
                    if result=="s":
                        problem.solve()
                        res.write(f"{inst : ^10}{NoProj[p] : ^10}{str(alpha) : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}{str(problem.projects) : ^30}")
                        res.write('\n') 
                    elif result=="w":
                        problem.write(n)
                        n+=1