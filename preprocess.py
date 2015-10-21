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
import copy
from random import shuffle


class Recipe(object):

    def __init__(self, uid=None, cuisine=None, ingredients=[]):
        self.uid = uid
        self.cuisine = cuisine
        self.ingredients = ingredients

    def get_int_ingredients(self, feature_dict):
        # array returned will be formatted as follows:
        #[ingredient0, ingredient1 ... ingredientn]
        return [feature_dict[ingredient] for ingredient in self.ingredients]

    def get_int_cuisine(self, cuisine_dict):
        return cuisine_dict[self.cuisine]

# will used to remove certain words in order to reduce feature space
black_list = ["whole", "fat", "reduced", "low", "crushed", "fine", "fresh",
              "ground", "less", "chopped", "nonfat", "lowfat", "large", "grated", "sodium", "lowsodium", "free", "lean", "no", "solid", "cooking", "tips", "kraft", "fresh", "frozen", "chopped", "oz", "boneless", "skinless", "tastethai", "barilla", "bertolli", "bestfoods", "campbells", "lowfat", "crisco pure", "crystal farms" "reduced fat", "delallo", "domino", "heinz", "herdez", "hiddenvalleyoriginal", "johnsonville", "and", "fat free", "reducedsodium", "lowsodium", "lowersodium"]


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
    recipes_test = []
    # If true uses blacklist to remove words
    REMOVE_WORDS = True
    REDUCE_TO_SINGLE = True
    CLEAN = True
    GET_UNIFORM = False
    #artifically inflate data size by using duplicate data
    GET_DUPLICATE = False
    # Multiplied the the class count of the class with the least number of
    # occurances.
    NUMBER_OF_SAMPLES_PER_CLASS = 1000

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

    with open('test.json') as f:
        recipes_json = json.loads(f.read())

        for recipe in recipes_json:
            uid = int(recipe['id'])
            features += recipe['ingredients']
            recipes_test.append(
                Recipe(uid=uid, ingredients=recipe['ingredients']))
    print "Total number of samples: %d" % len(recipes)
    print "Total number of test samples %d" % len(recipes_test)
    # the recipes are shuffled each time. This way we can get different
    # training subsets from the larager data set
    shuffle(recipes)

    label = 0
    for c in Set(classes):
        class_map[c] = label
        label += 1
    print class_map

    # clean recipes
    if CLEAN:
        features = clean(features)
        for recipe in recipes:
            recipe.ingredients = clean(recipe.ingredients)
        for recipe in recipes_test:
            recipe.ingredients = clean(recipe.ingredients)

    if REDUCE_TO_SINGLE:
        print "Reducing to single"
        features = Set(reduce_to_single_word(features))
        for recipe in recipes:
            recipe.ingredients = Set(reduce_to_single_word(recipe.ingredients))
        for recipe in recipes_test:
            recipe.ingredients = Set(reduce_to_single_word(recipe.ingredients))

    if REMOVE_WORDS:
        print "Removing words"
        features = Set(remove_words(features, black_list))
        for recipe in recipes:
            recipe.ingredients = Set(
                remove_words(recipe.ingredients, black_list))
        for recipe in recipes_test:
            recipe.ingredients = Set(
                remove_words(recipe.ingredients, black_list))

    if GET_UNIFORM:
        bins = defaultdict(list)
        for recipe in recipes:
            bins[class_map[recipe.cuisine]].append(recipe)

        recipes = []
        for key in bins.keys():
            class_count = len(bins[key])
            while GET_DUPLICATE and (class_count < NUMBER_OF_SAMPLES_PER_CLASS):
                print "Adding [%d] more recipes to [%s]" % (class_count, key)
                bins[key] += copy.copy(bins[key])
                class_count = len(bins[key])
            # print len(bins[key])
            # for i in range(len(bins[key]) % NUMBER_OF_SAMPLES_PER_CLASS):
            recipes = recipes + bins[key][:NUMBER_OF_SAMPLES_PER_CLASS]

    # print recipes[0]
    shuffle(recipes)
    print "Numober of train samples %d" % len(recipes)
    print "Numober of test samples %d" % len(recipes_test)
    feature_map = []
    y_train = []
    labels = []

    for recipe in recipes:
        ing_d = dict()
        labels.append(recipe.uid)
        y_train.append(class_map[recipe.cuisine])
        for ingredient in recipe.ingredients:
            ing_d[ingredient] = ingredient
        feature_map.append(ing_d)

    labels_test = []
    feature_map_test = []
    for recipe in recipes_test:
        ing_d = dict()
        labels_test.append(recipe.uid)
        for ingredient in recipe.ingredients:
            ing_d[ingredient] = ingredient
        feature_map_test.append(ing_d)

    vec = DictVectorizer(dtype=np.int)

    X_train = vec.fit_transform(feature_map).toarray()
    X_test = vec.transform(feature_map_test).toarray()

    print "num train features : %d" % len(X_train.transpose())
    print "num test features : %d" % len(X_test.transpose())
    y_train = np.asarray(y_train)
    labels = np.asarray(labels)
    labels_test = np.asarray(labels_test)

    label_y = np.append([labels], [y_train], axis=0)

    all_data = np.append(label_y, X_train.transpose(), axis=0)
    all_data_test = np.append([labels_test], X_test.transpose(), axis=0)
    # print all_data
    train_file_name = 'nn_train_%d' % len(recipes)
    test_file_name = 'nn_test' 
    np.save(train_file_name, all_data)
    np.save(test_file_name, all_data_test)
    print "Saving %s" % train_file_name
    print "Saving %s" % test_file_name
    # print y_train
if __name__ == "__main__":
    main()
