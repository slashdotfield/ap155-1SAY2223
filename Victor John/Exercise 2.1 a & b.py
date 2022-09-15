import math
A = float(input())
Z = float(input())
a_1 = float(15.67)
a_2 = float(17.23)
a_3 = float(0.75)
a_4 = float(93.2)
def a_5():
    if A%2 != 0:
        return 0
    elif A%2 == 0 and Z%2 == 0:
        return 12
    elif A%2 == 0 and Z%2 != 0:
        return -12

B = a_1*A - a_2*A**(2/3) - a_3*Z**(2)/A**(1/3) - a_4*(A-2*Z)**(2)/A + a_5()/A**(1/2)
print("The binding energy of an atom with mass number", A ,"and atomic number", Z ,"is", B)
print("Its binding energy per nucleon is", B/A)