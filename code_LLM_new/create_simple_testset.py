import random
import pickle
import Levenshtein

def get_title(text):
    # Implement this function to extract the title from your text
    # For example, if your text is "COL titel VAL The Title ...", extract "The Title"
    # Adjust this function based on your actual data format
    for part in text.split("COL "):
        if part.startswith("titel VAL "):
            return part.split("VAL ")[1].split(",")[0].strip()
    return ""
input = pickle.load(open("test_input.p", "rb"))
targets = pickle.load(open("test_targets.p", "rb"))
x=[]
y=[]

print(len(input))
print(len(targets))

for trip in targets:

    if trip[0] in input.keys():
        cor_input=input[trip[0]]
        x.append(cor_input)
        y.append(trip[1])

# Build index_by_work and index_by_expr
index_by_work = {}
index_by_expr = {}
for i, (work_id, expr_id) in enumerate(y):
    index_by_work.setdefault(work_id, []).append(i)
    index_by_expr.setdefault(expr_id, []).append(i)

already_did_it=[]
amount=0
pos_amount=0
neg=False
with open("neg_work_simple.txt", "w", encoding="utf-8") as f_work:

    for i in range(40000,len(x)):
        anchor_text = x[i]
        anchor_title = get_title(anchor_text)
        anchor_work, anchor_expr = y[i]
        same_work = index_by_work[anchor_work]
        # Positive: same work, same expr, not self
        pos_candidates = [j for j in same_work if y[j][1] == anchor_expr and j != i]
        sim_candidates=[]
        pos_idx=False
        if pos_candidates:
            for p in pos_candidates:
                c_text=x[p]
                c_title= get_title(c_text)
                if c_title not in already_did_it:
                    sim_candidates.append(p)
            if sim_candidates:
                pos_idx = random.choice(sim_candidates)

        neg_work_candidates = []
        for j in range(len(y)):
            if y[j][0] != anchor_work:
                candidate_title = get_title(x[j])
                if candidate_title not in already_did_it:
                    if candidate_title and anchor_title:
                        sim = Levenshtein.ratio(anchor_title, candidate_title)
                        if sim < 0.7:  # Set your similarity threshold here
                            neg_work_candidates.append(j)

            if len(neg_work_candidates)==5:
                continue
        if pos_idx:
            neg_idx = random.choice(neg_work_candidates)
            if pos_amount<50:
                f_work.write(f"{anchor_text}\t{x[pos_idx]}\t0\n")
                already_did_it.append(c_title)
                pos_amount+=1
            else:
                f_work.write(f"{anchor_text}\t{x[neg_idx]}\t1\n")
                already_did_it.append(candidate_title)
            amount+=1

                
        if amount==500:
            break
