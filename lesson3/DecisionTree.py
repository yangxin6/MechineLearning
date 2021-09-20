
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'data.csv', 'r')
reader = csv.reader(allElectronicsData)
headers = next(reader)

# print(headers)

featureList = []
labelList = []
for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

# print(featureList)
# print(labelList)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

# print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

# print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
# entropy
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
# print("clf: " + str(clf))

# Visualize model
with open("data.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# predict a new row

featureDict = {}
for i in vec.get_feature_names():
    tmp = i.split(vec.separator)
    if tmp[0] not in featureDict:
        featureDict[tmp[0]] = tmp[1]
    else:
        featureDict[tmp[0]] += '/' + tmp[1]

inputFeature = {}
print("预测输入: ")
for i in featureDict:
    inputFeature[i] = input(i+featureDict[i])
print("输入结果：")
print(inputFeature)

predictedX = vec.transform(inputFeature).toarray()
print(predictedX)


predictedY = clf.predict(predictedX)
print("预测结果: 类别(是否给贷款)", lb.classes_[predictedY][0])



