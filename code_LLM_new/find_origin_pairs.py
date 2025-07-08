import random
import pickle
import Levenshtein


input = pickle.load(open("test_input.p", "rb"))

with open("test_pairs_expr.txt", encoding="utf-8") as f:
    file = f.readlines()

pairlist=[]
for line in file:
    s1, s2, label = line.strip().split('\t')
    key_1 = [key for key, val in input.items() if s1 in val][0]
    key_2 = [key for key, val in input.items() if s2 in val][0]

    pairlist.append([key_1,key_2,label])
    
    

pickle.dump(pairlist, open("pairlist_expr.p", "wb"))  # save it into a file named save.p




