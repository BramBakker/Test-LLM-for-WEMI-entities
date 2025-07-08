import pickle



input = pickle.load(open("test_input.p", "rb"))

pair_list = pickle.load(open("pairlist_real.p","rb"))

with open("output.txt", "w", encoding="utf-8") as f:
    for key1, key2, label in pair_list:
        val1 = input.get(key1, "")
        val2 = input.get(key2, "")
        f.write(f"{val1}\t{val2}\t{label}\n")