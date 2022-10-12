

import numpy as np
import matplotlib.pyplot as plt



aa = np.array([
[ 4.63468, 0.34575, 4.71283, 0.34198,13.28045, 0.44282],
[ 3.96494, 0.33617, 4.12694, 0.33577,12.05528, 0.39306],
[ 5.42612, 0.36068, 5.55766, 0.35731,13.13804, 0.38443],
[ 7.13102, 0.39609, 7.36123, 0.39393,13.76039, 0.43585],
[ 6.58512, 0.37169, 6.83565, 0.36759,13.713  , 0.4118 ],
[ 2.98477, 0.31876, 3.07609, 0.31782,10.91666, 0.3508 ],
[ 8.2998 , 0.42939, 8.40895, 0.43062,14.38938, 0.5396 ],
[ 9.6113 , 0.48095, 9.71629, 0.48208,14.81024, 0.65854],
[12.5917 , 0.82733,12.97668, 0.82796,15.23245, 1.05798],
[ 7.61352, 0.40873, 7.92056, 0.41084,14.17641, 0.48862],
])

bb = np.argsort(aa[:,1])
plt.plot(aa[:, 1][bb], aa[:,0][bb], color = 'green', marker='o', linewidth=2, markersize=8, label = 'Greedy')
bb = np.argsort(aa[:,3])
plt.plot(aa[:, 3][bb], aa[:,2][bb], color = 'blue', marker='o', linewidth=2, markersize=8, label = 'Search')
bb = np.argsort(aa[:,5])
plt.plot(aa[:, 5][bb], aa[:,4][bb], color = 'red', marker='o', linewidth=2, markersize=8, label = 'M-Attack')

plt.grid(axis = 'y')
plt.xticks(fontsize = 10)
plt.xlabel('M-Distance', fontsize = 12)
plt.ylabel('Loss', fontsize = 12)

plt.legend(fontsize = 12, loc = 4)
plt.savefig('results/home_exp1_0.3_4.png')