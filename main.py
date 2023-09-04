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
        if problem.status == 2:
            opt+=1
        else:
            gaps=gaps+[problem.gap]

with open("results.txt", "w") as res:
    res.write("Number of optimal results = "+str(opt))
    res.write('\n')
    res.write("Number of non-optimal results = "+str(480-opt))
    res.write('\n')
    res.write("Running time = "+ str(time.time()-start))
    res.write('\n')
    res.write("Gaps: ")
    res.write(str(gaps))
        