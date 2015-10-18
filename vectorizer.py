import json
import gzip
from pprint import pprint
from sets import Set
from collections import Counter
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
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
    REDUCE_TO_SINGLE = False
    CLEAN = False
    features = []
    classes = []
    # used to count occurances of each ingrediant and cuisine type
    feature_cnt = Counter()
    class_cnt = Counter()

    feature_map = {}
    class_map = {}

    feature_total_cnt = Counter()
    class_total_cnt = Counter()

    with open('train.json') as f:
        recipes_json = json.loads(f.read())

        for recipe in recipes_json:
            uid = int(recipe['id'])
            cuisine = str(recipe['cuisine']).strip()
            classes.append(cuisine)
            features += recipe['ingredients']
            recipes.append(Recipe(uid, cuisine, recipe['ingredients']))

    for feature in features:
        feature_total_cnt[feature] += 1

    for cl in classes:
        class_total_cnt[cl] += 1

    print "Total Number of  Classes [%d]" % len(classes)
    print "Total Number of  Features [%d]" % len(features)
    print "Total Number of Recipes [%d]" % len(recipes)
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

    features = Set(clean(features))
    for recipe in recipes:
        recipe.ingredients= clean(recipe.ingredients)
    # create mappings here so we can convert recipes to int arrays
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(features)
    print X
    vectorizer.transform(recipes[0].ingredients).toarray()

    #print vectorizer.get_feature_names() 
    #print vectorizer.transform(recipes[0].ingredients).toarray()
if __name__ == "__main__":
    main()
