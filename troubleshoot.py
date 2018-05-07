# -*- coding: utf-8 -*-

import sys

failed = sys.argv[1]

troubles = open(failed).read().splitlines()

run = []
add = ''
ct = 0
for i in troubles:
    add += str('%s '%(i)); ct +=1
    if ct%50 == 0:
        run.append(str('bsub -R "rusage[mem=8192]" -W 24:00 "%s"' % add))
        add = ''
else:
    run.append(str('bsub -R "rusage[mem=8192]" -W 24:00 "%s"' % add))
        
file = open('troubleshooting.txt', 'w')
for item in run:
    file.write("%s\n" % item)

file.close()