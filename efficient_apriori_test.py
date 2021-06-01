import pandas as pd
from efficient_apriori import apriori

# N is a number of columns to import from csv file
N = 10

# read data from csv file and put it in dataframe
df = pd.read_csv("groceries.csv", header=None, names=range(N), na_filter=True)

# create a list of lists removing "nan" value
transactions = df.T.apply(lambda x: x.dropna().tolist()).tolist()
# print(transactions)

# converting a list of lists into a list of tuples
list_of_tuples = [tuple(row) for row in transactions]
# print(list_of_tuples)

# applying the algorithm
itemsets, rules = apriori(list_of_tuples, min_support=0.02, min_confidence=0.5)

# Print out every rule with 2 items on the left hand side,
# 1 item on the right hand side, sorted by lift
rules_rhs = filter(lambda rul: len(rul.lhs) == 2 and len(rul.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rul: rul.lift):
    print(rule)  # Prints the rule and its confidence, support, lift, ...
