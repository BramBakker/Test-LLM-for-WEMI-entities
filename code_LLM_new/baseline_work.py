from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pickle
import Levenshtein
import re
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

pattern = r'COL (.*?) VAL (.*?)(?= COL |$)'
preds = []
labels = []
total_correct = 0
with open("test_pairs_work.txt", "r", encoding="utf-8") as f:
    file = f.readlines()
def is_match(val1, val2, threshold=3):
    return Levenshtein.distance(val1.lower(), val2.lower()) <= threshold
pos = 0
neg = 0

imp_labels= ['ondertitel','titel','plaats','taal','uitgever','2e auteur','hoofdauteur']
imp_dict= {'ondertitel':0,'titel':0,'plaats':0,'taal':0,'uitgever':0,'2e auteur':0,'hoofdauteur':0}

combo_stats = defaultdict(list)
for line in file:
    s1, s2, label = line.strip().split('\t')
    match_score = 0.0
    pred=0
    # Find all matches
    m1 = re.findall(pattern, s1)
    m2 = re.findall(pattern, s2)

    # Convert to dictionary or list of tuples if needed
    mdict1 = dict(m1)
    mdict2 = dict(m2)
    shared_keys = set(mdict1) & set(mdict2)
    match_fields = []

    for key in shared_keys:
        if is_match(mdict1[key], mdict2[key]):
            if key=='hoofdauteur':
                if 'vertaling' in s1 or 'vertaling' in s2:
                    match_score+=5
                
            if key in imp_labels:
                if label=='0':
                    imp_dict[key]+=1
                if key=='taal':
                    match_score += 3

                match_score += 1
        #if not is_match(mdict1[key], mdict2[key]):
            #if 'jaar' in key:
                #match_score-=1

    #if herz.
    if match_score >= 6:
        pred=0
    else:
        pred=1
    if pred== int(label):
        total_correct += 1
    else:
        print(mdict1, 'is supposed to be:',label)
        print(mdict2)
    preds.append(pred)
    labels.append(int(label))
print(imp_dict)
print(preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    labels, preds, pos_label=0, average='binary'
)

print(f"Precision (for '0'): {precision:.4f}")
print(f"Recall (for '0'): {recall:.4f}")
print(f"F1-score (for '0'): {f1:.4f}")
print(f"Total correct: {total_correct} out of {len(file)} lines")
print(f"Accuracy: {total_correct / len(file) * 100:.2f}%")