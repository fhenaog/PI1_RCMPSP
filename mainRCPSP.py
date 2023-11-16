import time
from PSP import RCPSP

start=time.time()
opt=0
gaps=[]
nonOpt=[]
nonOptStatus=[]
objs=[]
with open("results.txt", "w") as res:
    res.write(f"{'Nodes' : ^10}{'Instance' : ^10}{'Sample' : ^10}{'|P|' : ^10}{'RS' : ^10}{'RF' : ^10}{'Objective' : ^15}{'Gap' : ^10}{'Runtime' : ^10}")
    res.write('\n')
    res.write(''.join(['-' for _ in range(95)]))
    res.write('\n')
    for s in range(2):
        nodes=str((s+1)*30)            
        for m in range(1,5):
            for n in range(1,3):
                inst=str(m)
                sample=str(n)

                problem=RCPSP(nodes,inst,sample)
                problem.solve()
                if problem.status == 2:
                    opt+=1
                    objs.append(problem.objVal)
                else:
                    gaps.append(problem.gap*100)
                    nonOpt.append(inst+sample)
                    nonOptStatus.append(problem.status)
                res.write(f"{nodes : ^10}{inst : ^10}{sample : ^10}{str(problem.nP) : ^10}{str(round(problem.RS,2)) : ^10}{str(round(problem.RF,2)) : ^10}{str(round(problem.objVal,1)) : ^15}{str(round(problem.gap,2)) : ^10}{str(round(problem.runtime,2)) : ^10}")
                res.write('\n')
        res.write(''.join(['-' for _ in range(95)]))
        res.write('\n')