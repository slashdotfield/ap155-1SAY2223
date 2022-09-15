import math
Z = range(1, 101)
A = 58
a_1 = 15.67
a_2 = 17.23
a_3 = 0.75
a_4 = 93.2
def a_5():
    if Z%2 == 0:
        return 12
    else:
        return -12
N = [] #list of binding energy per nucleon
for Z in range(1, 101):
    N.append((a_1*A - a_2*A**(2/3) - a_3*Z**(2)/A**(1/3) - a_4*(A-2*Z)**(2)/A + a_5()/A**(1/2))/A)
print(N)
print(N.index(max(N))+1)
print(N[N.index(max(N))])