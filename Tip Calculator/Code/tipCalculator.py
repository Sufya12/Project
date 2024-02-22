#Print The Statement
print('Welcome to the tip calacutor')
#Create The  Variable and use input function
bill_amount = int(input("What was the total bill? $ " ))
#Create The  Variable and use input function
divided_person = int(input("How many people to split the bill "))
#Create The  Variable and use input function
tip_for_bill = int(input("What percentage tip would you like to give? 10,12,or 15? "))
#Create The  Variable and apply the Tip Percentage Formula
tip_prcent = (bill_amount/100)*tip_for_bill
#Create The  Variable and Divided the Each Person Bill
each_person_bill =  (bill_amount + tip_prcent )/ (divided_person)
#Print The Statement and Each Person Bill1
print("Each person should pay:  $",each_person_bill)