import pandas
import kfold_template

from sklearn import tree
import matplotlib.pyplot as pyplot


dataset = pandas.read_csv("temperature_data.csv")

# print(dataset)

dataset = pandas.get_dummies(dataset)

dataset = dataset.sample(frac=1).reset_index()

print(dataset)

target = dataset['actual'].values
# print(target)

data = dataset.drop('actual', axis = 1)
# data = data.drop('historical', axis = 1)

feature_list = data.columns
data = data.values
# print(data)

machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=2)
return_values = kfold_template.run_kfold(data, target, machine, 4, True, False, False) 
print(return_values)


machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=2)
machine.fit(data, target)
feature_importances_raw = machine.feature_importances_
print(feature_importances_raw)
print(feature_list)
feature_zip = zip(feature_list, feature_importances_raw)
# print(feature_zip)
feature_importances = [(feature, round(importance, 4))  for feature, importance in feature_zip]
feature_importances = sorted(feature_importances, key = lambda x: x[1])
# print(feature_importances)
[ print('{:13}: {}'.format(*feature_importance)) for feature_importance in feature_importances]

x_values = list(range(len(feature_importances_raw)))
y_values = feature_importances_raw
pyplot.bar(x_values, y_values)
pyplot.xticks(x_values, feature_list, rotation="vertical")
pyplot.title("Feature Importance")
pyplot.tight_layout()
pyplot.savefig("feature_importances.png")
pyplot.close()






