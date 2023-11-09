cash=[3635/4736, 47436/91992]
creditCard=[15252/22324, 22295/29533, 1456/2750]

C=0
D=0
T=0
for i in cash:
    for j in creditCard:
        if (i>j):
            C=C+1
        elif (i<j):
            D=D+1
        else:
            T=T+1
            
AUC=0.5+(0.5*(C-D)/len(cash)*len(creditCard))
print("Area Under Curve value: ",AUC)