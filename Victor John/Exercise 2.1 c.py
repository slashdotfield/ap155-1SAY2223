import math
Z = float(input())
A = range(int(Z), 3*int(Z)+1)
a_1 = 15.67
a_2 = 17.23
a_3 = 0.75
a_4 = 93.2
def a_5():
    if A%2 != 0:
        return 0
    elif A%2 == 0 and Z%2 == 0:
        return 12
    elif A%2 == 0 and Z%2 != 0:
        return -12
N = [] #This is the list of total binding energy per nucleon for each value of Z from A to 3Z
for A in range(int(Z), 3*int(Z)+1):
    N.append((a_1*A - a_2*A**(2/3) - a_3*Z**(2)/A**(1/3) - a_4*(A-2*Z)**(2)/A + a_5()/A**(1/2))/A)
print(N)
print("The largest binding energy per nucleon is", max(N), "and is given out by the most stable nucleus which has the mass number" , N.index(max(N))+28)