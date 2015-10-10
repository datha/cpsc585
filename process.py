import json
import gzip
from pprint import pprint
from sets import Set
from collections import Counter
import matplotlib.pyplot as plt
import re



#will used to remove certain words in order to reduce feature space
black_list = ["whole","fat", "reduced", "low","crushed","fine","fresh",
            "ground","less","chopped","nonfat","lowfat","large","grated","sodium","lowsodium","free","lean","no", "solid","cooking","tips"]

def intersect(a,b):
    return list(set(a) & set(b))

def remove_words(words=[], black_list=[]):
    new_list = []
    for word in words:
        overlap = intersect(word.lower().split(" "),black_list)
        if overlap:
            for term in overlap:
                word = word.replace(term,"")
        new_list.append(word)
    return new_list

def clean(words=[]):
    clean_list=[]
    #Remove everything that's not a letter or space
    for word in words:
        clean_word = re.sub(r"[^a-zA-Z]"," ",word)
        clean_list.append(clean_word.lower().strip())
    return clean_list

def graph(dict_to_graph={}):
    plt.barh(range(len(dict_to_graph)), dict_to_graph.values(),align='center')
    plt.yticks(range(len(dict_to_graph)), dict_to_graph.keys())
    plt.show()

def reduce_to_single_word(words):
    new_set = Set()
    for word in words:
        for w in word.lower().split(" "):
            if w:
                new_set.add(w)
                print w
    return new_set

#main starts here
def main():
    #If true uses blacklist to remove words
    reduce_feature = True
    reduce_to_single = True
    features = []
    classes = []
    #used to count occurances of each ingrediant and cuisine type
    feature_cnt = Counter()
    class_cnt = Counter()

    feature_total_cnt = Counter()
    class_total_cnt = Counter()

    with open('train.json') as f:
        recipes = json.loads(f.read())

        for recipe in recipes:
            classes.append(str(recipe['cuisine']).strip())
            features += recipe['ingredients']

    for feature in features:
        feature_total_cnt[feature] += 1

    for cl in classes:
        class_total_cnt[cl] += 1

    #remove duplicate ingrediant entries
    class_set = Set(classes)
    feature_set = Set(features)

    print "Total Number of Unique Classes [%d]" % len(class_set)
    print "Total Number of Unique Features [%d]" % len(feature_set)
    pprint(class_set)
    feature_set = Set(clean(feature_set))
    if reduce_to_single:
        print "Reducing to single"
        feature_set = Set(reduce_to_single_word(feature_set))
    if reduce_feature:
        print "Removing words"
        feature_set = Set(remove_words(feature_set,black_list))

    feature_set = Set(clean(feature_set))

    print "Cleaned Number of Features [%d]" % len(feature_set)

    with open('ingrediants_by_occurance.txt', 'w') as sorted_ingrediants_file:

        for ingrediant , cnt in feature_total_cnt.most_common():
            sorted_ingrediants_file.write(str(ingrediant.encode('utf-8')) + ":" + str(cnt)+"\n")
    with open('ingrediants_reduced.txt', 'w') as feature_set_file:
        for ingrediant in feature_set:
            feature_set_file.write(str(ingrediant.encode('utf-8')) +"\n")
    graph(class_total_cnt)
if __name__ == "__main__":
    main()