import json
import gzip
from pprint import pprint
from sets import Set
from collections import Counter
import matplotlib.pyplot as plt
import re



#used to remove certain words in order to reduce feature space
black_list = ["whole","fat", "reduced", "low"]

def clean(words):
    clean_list=[]
    for word in words:
        clean_word = re.sub(r"[^a-zA-Z\s]","",word)
        clean_list.append(clean_word.lower())
    return clean_list

def graph(dict_to_graph={}):
    #y_pos = np.arange(len(list(d.get_keys()))
    plt.barh(range(len(dict_to_graph)), dict_to_graph.values(),align='center')
    plt.yticks(range(len(dict_to_graph)), dict_to_graph.keys())
    plt.show()


#main starts here
def main():
    features = []
    classes = []
    #used to count occurances of each ingrediant and cuisine type
    feature_cnt = Counter()
    class_cnt = Counter()

    with open('train.json') as f:
        recipes = json.loads(f.read())

        for recipe in recipes:
            classes.append(str(recipe['cuisine']).strip())
            #print recipe['cuisine']
            features += recipe['ingredients']
            # pprint(recipe['ingredients'])

    #pprint(Set(features))
    #pprint(classes)

    for feature in features:
        feature_cnt[feature] += 1

    for cl in classes:
        class_cnt[cl] += 1

    class_set = Set(classes)
    feature_set = Set(features)
    #print feature_set
    print clean(class_set)
    feature_set = clean(feature_set)
    feature_set.sort()
    #pprint(feature_set)
    #print clean(ingredients)
    #print "Number of Features [%d]" % len(feature_set)
    #print "Number of Classes [%d]" % len(class_set)
    #pprint(class_cnt)
    #pprint(feature_cnt)
    graph(class_cnt)

if __name__ == "__main__":
    main()