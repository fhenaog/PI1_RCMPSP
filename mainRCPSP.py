import time
from PSP import RCPSP

start=time.time()
opt=0
gaps=[]
nonOpt=[]
nonOptStatus=[]
objs=[]
with open("results.txt", "w") as res:
    res.write(f"{'Instance' : ^10}{'Sample' : ^10}{'Objective' : ^15}{'Gap' : ^10}{'Runtime' : ^10}")
    res.write('\n')
    res.write(''.join(['-' for _ in range(55)]))
    res.write('\n')
    for m in range(1,49):
        for n in range(1,11):
            inst=str(m)
            sample=str(n)

            problem=RCPSP(inst,sample)
            problem.solve()
            if problem.status == 2:
                opt+=1
                objs.append(problem.objVal)
            else:
                gaps.append(problem.gap*100)
                nonOpt.append(inst+sample)
                nonOptStatus.append(problem.status)
            res.write(f"{inst : ^10}{sample : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}")
            res.write('\n')