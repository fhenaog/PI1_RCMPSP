import time
from PSP import RCMPSP_Resource

start=time.time()
opt=0
gaps=[]
nonOpt=[]
nonOptStatus=[]
objs=[]
NoProj=[2,3,5,10] #Number of projects
alphaVal=[0.0,0.25,0.5,0.75,1.0] #alpha values
instances=[[13],[48],[29],[24],[13,48,29,24]] #instances (last one is for mixing)
result="write" # "solve" --- "write" to write the problem, "solve" to solve the problem
n=1
with open("resultsResource.txt", "w") as res:
    res.write(f"{'Instance' : ^10}{'No. Proj' : ^10}{'Res Cost Type' : ^15}{'Alpha' : ^10}{'Objective' : ^15}{'Gap' : ^10}{'Runtime' : ^10}{'Resources' : ^40}")
    res.write('\n')
    res.write(''.join(['-' for _ in range(120)]))
    res.write('\n')
    for inst in instances:
        for p in NoProj:
            for type in range(1,3-1): #For constant or variable resource values 1 - constant, 2 - variable
                for alpha in alphaVal:
                    
                    problem=RCMPSP_Resource(inst,p,alpha,type)
                    if result=="solve":
                        problem.solve()

                        if type==1:
                            res.write(f"{inst : ^10}{p : ^10}{'Constant' : ^15}{str(alpha) : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}{str(problem.resources) : ^40}")
                            res.write('\n') 
                        else:
                            res.write(f"{inst : ^10}{p : ^10}{'Variable' : ^15}{str(alpha) : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}{str(problem.resources) : ^40}")
                            res.write('\n')
                    elif result=="write":
                        problem.write(n)
                        n+=1