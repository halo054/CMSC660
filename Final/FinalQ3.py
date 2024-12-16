import numpy as np
import scipy
import random

vol_cube = 1


def calc_ball_vol(dimension):
    temp = np.pi **(dimension/2)
    denominator = (dimension/2) * scipy.special.gamma(dimension/2)
    return temp / denominator



def way_one(dimension): #dot in cube
    in_both = 0
    N_ball = 1000000
    for i in range(N_ball):
        x = np.random.rand(dimension)
        ones = np.ones(dimension)
        x = x - ones/2
        
        if np.inner(x,x) <= 1:
            in_both+=1
    return in_both/N_ball


intersection_way_one_5 = way_one(5)
intersection_way_one_10 = way_one(10)
intersection_way_one_15 = way_one(15)
intersection_way_one_20 = way_one(20)
print("Way one:")
print("Dimension =", 5, ", Volumn of intersection =",intersection_way_one_5)
print("Dimension =", 10, ", Volumn of intersection =",intersection_way_one_10)
print("Dimension =", 15, ", Volumn of intersection =",intersection_way_one_15)
print("Dimension =", 20, ", Volumn of intersection =",intersection_way_one_20)

'''
def generate_x_for_ball(dimension):
    summation = 1
    entry_list = []
    for i in range(dimension):
        x = random.uniform(0, summation)
        entry_list.append(x**(1/2))
        summation = summation - x
    return np.array(entry_list)
    '''
def generate_x_for_ball(dimension):
    entry = []
    for i in range(dimension):
        entry.append(np.random.normal())
    entry = np.array(entry)
    return (random.uniform(0,1)**(1/dimension)/np.inner(entry,entry)**(1/2)) *entry
    
def way_two(dimension): #dot in cube
    vol_ball = calc_ball_vol(dimension)
    #print("vol_ball",vol_ball)
    in_both = 0
    N_ball = 1000000
    for i in range(N_ball):
        x = generate_x_for_ball(dimension)
        #ones = np.ones(dimension)
        #x = x - ones/2
        #x = x * 2
        if np.max(abs(x)) <= 1/2:
            in_both+=1
        
    return (in_both/N_ball)*vol_ball

intersection_way_two_5 = way_two(5)
intersection_way_two_10 = way_two(10)
intersection_way_two_15 = way_two(15)
intersection_way_two_20 = way_two(20)
print("Way two:")
print("Dimension =", 5, ", Volumn of intersection =",intersection_way_two_5)
print("Dimension =", 10, ", Volumn of intersection =",intersection_way_two_10)
print("Dimension =", 15, ", Volumn of intersection =",intersection_way_two_15)
print("Dimension =", 20, ", Volumn of intersection =",intersection_way_two_20)