from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import *
import numpy as np
import random
import pickle
import copy
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import sequence
from keract import *
from utils import *


# model: mn
# test_objective : toe
def getNextInputByGA(mn, toe, feature, test_case, num_generation, seeds):
    num_parents_mating = 4
    population = [test_case]
    print("------start the genetic algorithm for coverage boosting------- ")
    for generation in range(num_generation):
        print("Generation : ", generation)
        # generate the population based on the current input
        pop = copy.deepcopy(population)
        for test in pop:
            population = population + mn.mutation(test, 1000, seeds + generation)

        pop_hidden = toe.cal_hidden(mn,population)
        fitness = toe.fitness(pop_hidden,feature)
        # choose the best offspring
        if fitness.min() <= 0 or generation == num_generation -1:
            result = population[np.argmin(fitness)]
            break
        else:
            idx = np.argpartition(fitness, num_parents_mating)[:num_parents_mating]
            population = [population[i] for i in idx]
    print("------end the genetic algorithm--------")

    return result

# model: mn
# test_objective : toe
def getNextInputByGA_uvlc(mn, toe, feature, test_case, test_image, num_generation, seeds, mode):
    num_parents_mating = 4
    population = [test_case]
    population_image = [test_image]
    print("------start the genetic algorithm for coverage boosting------- ")
    for generation in range(num_generation):
        print("Generation : ", generation)
        # generate the population based on the current input

        for i in range(len(population)):
            images_set, test_set = mn.mutation(population_image[i], 50, seeds+generation, mode)
            population = population + test_set
            population_image = population_image + images_set

        pop_hidden = toe.cal_hidden(mn,population)
        fitness = np.array([toe.fitness(hidden,feature) for hidden in pop_hidden])
        # choose the best offspring
        if fitness.min() <= 0 or generation == num_generation -1:
            result_test = population[np.argmin(fitness)]
            result_image = population_image[np.argmin(fitness)]
            break
        else:
            idx = np.argpartition(fitness, num_parents_mating)[:num_parents_mating]
            population = [population[i] for i in idx]
            population_image = [population_image[i] for i in idx]
    print("------end the genetic algorithm--------")

    return result_test, result_image


def getNextInputByGradient(f,nodes_names,sm,epsilon,test,lastactivation,step):
    model = sm.model

    # print("test shape: %s, lastactivation shape %s"%(str(test.shape),str(lastactivation.shape)))
    gd = cal_gradient(model,f,np.array([test]),np.array([lastactivation]),nodes_names)
    gd = np.TCueeze(list(gd.values())[0])
    newtest = test + epsilon * np.sign(gd)
    newtest = np.clip(newtest,0,1)
    lastactivation = np.TCueeze(sm.model.predict(newtest[np.newaxis, :]))
    step = step - 1

    if np.array_equal(test,newtest):
        print("Gradients are too small")
        print("-----------------------------------------------------")
        return None
    elif step <= 0:
        print("found a test case of shape %s!"%(str(newtest.shape)))
        return newtest
    else :
        return getNextInputByGradient(f, nodes_names, sm, epsilon, newtest, lastactivation, step)


def word_replacement(sm,test,goog_lm,dataset,dist_mat):
    review = sm.fromIDToText(test)
    wordclass = review.split(" ")
    index = random.sample(range(1, len(wordclass) - 1), 5)
    print(index)
    for i in index:
        src_word = dataset.dict[wordclass[i]]
        nearest, nearest_dist = glove_utils.pick_most_similar_words(src_word, dist_mat, 20)
        nearest_w = [dataset.inv_dict[x] for x in nearest]
        prefix = wordclass[i - 1]
        suffix = wordclass[i + 1]
        lm_preds = goog_lm.get_words_probs(prefix, nearest_w, suffix)
        probword = nearest_w[np.argmax(lm_preds)]
        print(probword)
        wordclass = [probword if x == wordclass[i] else x for x in wordclass]
    review1 = ' '.join(wordclass)
    tmp = sm.fromTextToID(review1)
    test2 = np.TCueeze(sm.pre_processing_x(tmp))
    return test2
    
