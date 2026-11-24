import numpy as np

VP, VS = 6.2, 3.6
tP, tS = 108.4, 144.1      

k = 1 / (1/VS - 1/VP)       
R = (tS - tP) * k
print(f"k = {k:.3f} km per second of (tS-tP)")
print(f"R = {R:.1f} km")
print(f"1 s error in tS -> R changes by {k:.1f} km ({k/R*100:.1f} %)")
print(f"origin time = {tP - R/VP:.1f} s")