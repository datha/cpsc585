import json
import gzip
from pprint import pprint
from sets import Set
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import sys
import pickle
from collections import Counter

from random import shuffle
class Recipe(object):

    def __init__(self, uid, cuisine, ingredients):
        self.uid = uid
        self.cuisine = cuisine
        self.ingredients = ingredients

    def get_int_ingredients(self, feature_dict):
        #array returned will be formatted as follows:
        #[ingredient0, ingredient1 ... ingredientn]
        return [feature_dict[ingredient] for ingredient in self.ingredients]
    def get_int_cuisine(self,cuisine_dict):
        return cuisine_dict[self.cuisine] 

# will used to remove certain words in order to reduce feature space
black_list = ["whole", "fat", "reduced", "low", "crushed", "fine", "fresh",
              "ground", "less", "chopped", "nonfat", "lowfat", "large", "grated", "sodium", "lowsodium", "free", "lean", "no", "solid", "cooking", "tips"]


def intersect(a, b):
    return list(set(a) & set(b))


def remove_words(words=[], black_list=[]):
    new_list = []
    for word in words:
        overlap = intersect(word.lower().split(" "), black_list)
        if overlap:
            for term in overlap:
                word = word.replace(term, "")
        if word:
            new_list.append(word)
    return new_list


def clean(words=[]):
    clean_list = []
    # Remove everything that's not a letter or space
    for word in words:
        clean_word = re.sub(r"[^a-zA-Z]", " ", word)
        clean_list.append(clean_word.lower().strip())
    return clean_list


def graph(dict_to_graph={}):
    plt.barh(range(len(dict_to_graph)), dict_to_graph.values(), align='center')
    plt.yticks(range(len(dict_to_graph)), dict_to_graph.keys())
    plt.show()


def reduce_to_single_word(words=[]):
    new_set = Set()
    for word in words:
        for w in word.lower().split(" "):
            if w:
                new_set.add(w)
    return new_set

# main starts here


def main():

    recipes = []
    # If true uses blacklist to remove words
    REMOVE_WORDS = False
    REDUCE_TO_SINGLE = True
    CLEAN = True
    GET_UNIFORM = True
    features = []
    classes = []
    # used to count occurances of each ingrediant and cuisine type
    feature_cnt = Counter()
    class_cnt = Counter()

    feature_map = {}
    class_map = {}

    class_total_cnt = Counter()

    with open('train.json') as f:
        recipes_json = json.loads(f.read())

        for recipe in recipes_json:
            uid = int(recipe['id'])
            cuisine = str(recipe['cuisine']).strip()
            classes.append(cuisine)
            features += recipe['ingredients']
            recipes.append(Recipe(uid, cuisine, recipe['ingredients']))

    print "Total number of samples: %d" % len(recipes)
    for cl in classes:
        class_total_cnt[cl] += 1

    shuffle(recipes)
    label = 0
    for c in Set(classes):
        class_map[c] = label
        label += 1
    print class_map

    #clean recipes
    if CLEAN:
        features = clean(features)
        for recipe in recipes:
            recipe.ingredients= clean(recipe.ingredients)

    if REDUCE_TO_SINGLE:
        print "Reducing to single"
        features = Set(reduce_to_single_word( ))
        for recipe in recipes:
            recipe.ingredients = Set(reduce_to_single_word(recipe.ingredients))
    if REMOVE_WORDS:
        print "Removing words"
        features = Set(remove_words(features, black_list))

    if GET_UNIFORM:
        #get least number fo samples
        c, least = class_total_cnt.most_common()[-1]
        number_per_class = least*3
        print number_per_class
        bins = defaultdict(list)
        for recipe in recipes:
            bins[class_map[recipe.cuisine]].append(recipe)
        
        recipes = []
        for key in bins.keys():
            recipes = recipes + bins[key][:number_per_class]

    #print recipes[0]
    shuffle(recipes)
    print "Numober of samples %d" % len(recipes)

    feature_map = []
    y_train =[]
    labels =[]
    for recipe in recipes:
        ing_d = dict()
        labels.append(recipe.uid)
        y_train.append(class_map[recipe.cuisine])
        for ingredient in recipe.ingredients:
            ing_d[ingredient] = ingredient
        feature_map.append(ing_d)

    x_vec = DictVectorizer(dtype=np.int)

    X_train = x_vec.fit_transform(feature_map).toarray()

    y_train = np.asarray(y_train)
    labels = np.asarray(labels)
    label_y = np.append([labels],[y_train],axis=0)
    #I think this still keesp the data lined up..need to double check
    all_data = np.append(label_y,X_train.transpose(),axis=0)
    #print all_data
    np.save('nn_input', all_data)
    #print y_train
if __name__ == "__main__":
    main()
