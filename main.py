import time
from PSP import RCPSP

start=time.time()
opt=0
gaps=[]
#objs=[]
for m in range(1,49):
    for n in range(1,11):
        inst=str(m)
        sample=str(n)

        problem=RCPSP(inst,sample)
        problem.solve()
        print(problem.objVal)
        print(problem.status)
        print(problem.gap)
        if problem.status == 2:
            opt+=1
        else:
            gaps=gaps+[problem.gap]

print("Number of optimal results = ",str(opt))
print("Number of non-optimal results = ",str(480-opt))
print("Running time = ", time.time()-start)
print("Gaps ", gaps)
        