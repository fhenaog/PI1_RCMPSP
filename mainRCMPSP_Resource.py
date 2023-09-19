import time
from PSP import RCMPSP_Resource

start=time.time()
opt=0
gaps=[]
nonOpt=[]
nonOptStatus=[]
objs=[]
NoProj=[1,2,3]
alphaVal=[0.0,0.5,1.0]
with open("resultsResource.txt", "w") as res:
    res.write(f"{'Instance' : ^10}{'No. Proj' : ^10}{'Res Cost Type' : ^15}{'Alpha' : ^10}{'Objective' : ^15}{'Gap' : ^10}{'Runtime' : ^10}{'Resources' : ^40}")
    res.write('\n')
    res.write(''.join(['-' for _ in range(120)]))
    res.write('\n')
    for m in range(1,49):
        inst=str(m)
        for p in range(len(NoProj)):
            for type in range(1,3):
                for a in range(len(alphaVal)):
                    alpha=alphaVal[a] 

                    problem=RCMPSP_Resource(inst,NoProj[p],alpha,type)
                    problem.solve()

                    if type==1:
                        res.write(f"{inst : ^10}{NoProj[p] : ^10}{'Constant' : ^15}{str(alpha) : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}{str(problem.resources) : ^40}")
                        res.write('\n') 
                    else:
                        res.write(f"{inst : ^10}{NoProj[p] : ^10}{'Variable' : ^15}{str(alpha) : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}{str(problem.resources) : ^40}")
                        res.write('\n')