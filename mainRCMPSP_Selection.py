import time
from PSP import RCMPSP_Selection

start=time.time()
opt=0
gaps=[]
nonOpt=[]
nonOptStatus=[]
objs=[]
NoProj=[2,3,5]
alphaVal=[0.0,0.5,1.0]
with open("resultsSelection.txt", "w") as res:
    res.write(f"{'Instance' : ^10}{'No. Proj' : ^10}{'Alpha' : ^10}{'Objective' : ^15}{'Gap' : ^10}{'Runtime' : ^10}{'Projects' : ^30}")
    res.write('\n')
    res.write(''.join(['-' for _ in range(95)]))
    res.write('\n')
    for m in range(1,2):
        inst=str(m)
        for p in range(len(NoProj)-1):
                for a in range(len(alphaVal)):

                    alpha=alphaVal[a] 
                    
                    problem=RCMPSP_Selection(inst,NoProj[p],alpha)
                    problem.solve()
                    res.write(f"{inst : ^10}{NoProj[p] : ^10}{str(alpha) : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}{str(problem.projects) : ^30}")
                    res.write('\n') 