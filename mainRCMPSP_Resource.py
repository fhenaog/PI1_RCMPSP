import time
from PSP import RCMPSP_Resource

start=time.time()
opt=0
gaps=[]
nonOpt=[]
nonOptStatus=[]
objs=[]
alphaVal=[1.0,0.0,0.5]
with open("resultsResource.txt", "w") as res:
    res.write(f"{'Instance' : ^10}{'No. Proj' : ^10}{'Res Cost Type' : ^15}{'Alpha' : ^10}{'Objective' : ^15}{'Gap' : ^10}{'Runtime' : ^10}")
    res.write('\n')
    res.write(''.join(['-' for _ in range(80)]))
    res.write('\n')
    for m in range(1,49):
        inst=str(m)
        for P in range(2,4):
            problem=RCMPSP_Resource(inst,P,1,1,1)
            problem.solve()

            res.write(f"{inst : ^10}{P : ^10}{'Constant' : ^15}{'NA' : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}")
            res.write('\n')
            for a in range(3):
                alpha=alphaVal[a]                

                problem=RCMPSP_Resource(inst,P,alpha,1,2)
                problem.solve()

                res.write(f"{inst : ^10}{P : ^10}{'Variable' : ^15}{str(alpha) : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}")
                res.write('\n')