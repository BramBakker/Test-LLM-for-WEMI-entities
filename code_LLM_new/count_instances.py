with open("test_pairs_work.txt", "r", encoding="utf-8") as f:
    file = f.readlines()


zeroes=0
ones=0
for line in file:
    s1, s2, label = line.strip().split('\t')
    if label=='0':
        zeroes+=1
    if label=='1':
        ones+=1
print(zeroes)
print(ones)