with open("test_pairs_work.txt", "r", encoding="utf-8") as f:
    file = f.readlines()


pos=0
neg=0


for line in file:
    s1, s2, label = line.strip().split('\t')
    if label=='0':
        pos+=1
    if label=='1':
        neg+=1
print(pos)
print(neg)