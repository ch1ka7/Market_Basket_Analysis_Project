import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# N is a number of columns to import from csv file
N = 20

# read data from csv file and put it in dataframe
df = pd.read_csv("groceries.csv", header=None, names=range(N), na_filter=True)

# create a list of lists removing "nan" value
transactions = df.T.apply(lambda x: x.dropna().tolist()).tolist()
# print(transactions)

# transform data into a one-hot encoded dataframe
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
result = pd.DataFrame(te_ary, columns=te.columns_)

# extract frequent itemsets
frequent_itemsets = apriori(result, min_support=0.02, use_colnames=True)
# frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# extract association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.45)
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

# save final association rules into csv
# rules.to_csv("rules.csv")

print("******************************************************")
print("List of Rules")
print("******************************************************")
print(rules.value_counts())

filtered = rules[(rules["antecedent_len"] >= 2) &
                 (rules["confidence"] > 0.5) &
                 (rules["lift"] > 2)]

print("\n")
print("******************************************************")
print("The three items most often purchased together are: ")
print("******************************************************")
print(filtered.value_counts())
