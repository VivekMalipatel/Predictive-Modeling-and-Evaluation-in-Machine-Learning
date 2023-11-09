from math import sqrt

ErrorCash=[1101/4736,44556/91992]
ErrorCreditCard=[7072/22324, 7238/29533, 1294/2750]

ASE=0

for i in ErrorCash:
    ASE=ASE+(i*i)

for j in ErrorCreditCard:
    ASE=ASE+(j*j)

ASE=ASE/(len(ErrorCash)+len(ErrorCreditCard))
print("Root Average Squared Error: ",sqrt(ASE))