from collections import defaultdict
import pickle
from collections import Counter
# Work group ID store (global)
work_groups = {}
# Expression group ID store scoped to work
expr_groups_within_work = defaultdict(dict)

triples = []
test_triples = []
def get_group_id(section_str, group_dict, prefix):
    key = section_str
    if key not in group_dict:
        group_dict[key] = f"{prefix}_{len(group_dict)+1}"
    return group_dict[key]

def get_expr_group_within_work(work_id, expr_data):
    key = tuple(sorted(expr_data))
    group_dict = expr_groups_within_work[work_id]
    if key not in group_dict:
        group_dict[key] = f"E_{len(group_dict)+1}"
    return group_dict[key]
aut_counts={}

aut_list=['068435061#couperus',"072614730#bruna","068611447#hulst","068243391#multatuli",
          "068808658#tolstoj","068690436#kieviet", "068396198#andersen", "068291965#simenon", 
          "069804354#beckman","068405944#balzac","074333682#slee","069815100#teng","068412606#terlouw",
          "068800355#thijssing-boer","068256671#reve","07492866X#murakami","075153416#grunberg"]


bottom_aut_list=[]

top_aut_list=['known', '2614730#bruna', '8611447#hulst', '8396198#andersen', '8355289#vries', '8650256#saris', '8609124#grimm', '8730748#steel', '8291965#simenon', '8435061#couperus', '8690436#kieviet', '8655193#king', '1459273#amant', '8426046#verne', '8758162#christie', '0328531#van genechten', '4830653#slegers', '8747772#velthuijs', '8761155#cramer', '8729332#konsalik', '8911408#lindgren', '9691444#busser', '4333682#slee', '9143005#schmid', '8747055#dahl',
              '8870752#vandersteen', '9060843#gerdes', '8808658#tolstoj', '8569807#bomans', '8756313#hill','8380194#dostoevskij', '8230656#haasse', '8905106#abkoude', '8207581#vestdijk', "068243391#multatuli",
              '9046581#zeeuw', '8685440#andriessen', '8812507#kromhout', '8865953#hagers', '8586620#rutgers van der loeff', '8877048#haar', '8474202#disney', '8342853#dickens', '8819293#norel', '8417489#fabricius', '0060312#louwerse', '0340080#tellegen', '8589816#cookson', '0844383#mansell', '7744346#toonder', '9499160#boeke', '8800355#thijssing-boer', '1598545#grashoff', '8791097#graaf', '9636664#alcott', '9625069#marxveldt', '8911424#carle','854796X#mulisch', 
              '895462X#hildebrand', '8873131#peters', "068352050#dumas", '862820X#burgers-drost', '7768349#oud', '9339279#hartog', '9459452#malot', '1039910#roggeveen', '8845057#rendell', '9366829#potter', '8428790#may', '847749X#hermans', '9669961#diekmann', '0592675#charteris', '9461166#coben', '9088330#ludlum', '2928948#loon', '4146642#montefiore', '3097829#de bel', '6369822#cousins', '8922000#verroen', '3621196#hollander', '8412606#terlouw', '8256671#reve']

with open("romans_kinderboeken_CAP_Manifestations.txt", "r", encoding="utf-8") as f:
    next(f) 

    for line in f:
        if not line:
            continue

        parts = line.split("\t")
        if len(parts)>1:
            identifier = parts[0]
            data = parts[1]
            tokens = data.split("\\\\")
            work_data = []
            expr_data = []
            if tokens[0] in aut_counts:
                aut_counts[tokens[0]] += 1
            else:
                aut_counts[tokens[0]] = 1
                
            for token in tokens:
                if token.startswith(r"\W1"):
                    aut_token=token
                    work_data.append(token)
                if token.startswith("W3"):
                    title_token=token
                if token.startswith("W"):
                    work_data.append(token)
                elif token.startswith("E"):
                    expr_data.append(token)
            work_string=" ".join(work_data)
            # Assign work group
            work_id = get_group_id(work_string, work_groups, "W")


            # Assign expression group scoped *within* this work
            expr_id = get_expr_group_within_work(work_id, expr_data)
            if any(substring in tokens[0] for substring in top_aut_list):
                test_triples.append((identifier, [work_id, expr_id]))
            else:
                triples.append((identifier, [work_id, expr_id]))


topaut_list=[]
t = Counter(aut_counts)             
for k, v in t.most_common(500):
    topaut_list.append(k[6:])
print(topaut_list)
print(len(test_triples))
print(len(triples))
pickle.dump(triples, open("train_targets.p", "wb"))  # save it into a file named save.p
pickle.dump(test_triples, open("test_targets.p", "wb"))  # save it into a file named save.p
