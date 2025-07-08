
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sklearn.metrics import precision_recall_fscore_support

import os

# Set your API key
model_name = "BramVanroy/fietje-2-instruct"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# Create pipeline
def make_prompt(record_a: str, record_b: str) -> str:
    return f"""
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    Dit betekent: Hebben de twee boeken een gelijke intellectuele inhoud, dezelfde karakters en hetzelfde plot?
    belangrijke feiten:
    1: Vertalingen van hetzelfde boek behoren ook tot hetzelfde werk, originele auteursnamen kunnen dan in het veld "hoofdauteur" of "2e auteur" staan.
    2: Stripboeken en theaterstukken van hetzelfde verhaal behoren veelal tot hetzelfde werk.
    3: Vertalingen van verschillende werken kunnen soms op elkaar lijken.
    4: Boeken met verschillende titels in dezelfde serie, dus met dezelfde karakters en setting behoren tot verschillende werken.
    5: Kleine verschillen in interpunctie en spellingsvariaties kunnen voorkomen in titels van hetzelfde werk.
    6: Woorden met andere semantische lading in de titel zijn een aanwijzing voor een ander werk.
    
    voorbeeld 1:
    
    record A: COL titel VAL Het grote voorleesboek COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Hulst, Willem Gerrit van de Hulst 1917-2006 COL editie VAL 11e
    record B: COL titel VAL Het grote voorleesboek COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Hulst, Willem Gerrit van de Hulst 1917-2006 COL editie VAL 5e dr

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee met korte redenering: 
    reactie: Ja (exact dezelfde titel en auteur)

    nu jij:

Record A: {record_a}

Record B: {record_b}

Behoren deze records tot hetzelfde werk? Antwoord alleen Ja of Nee met korte redenering:

"""
batch_size = 8  # You can increase this if you have enough GPU memory

prompts = []
labels = []
with open("short_pairs_work.txt", "r", encoding="utf-8") as f:
    for line in f:
        s1, s2, label = line.strip().split('\t')
        prompts.append(make_prompt(s1, s2))
        labels.append(label)

correct_match = 0
correct_labels=[]
pred=[]
for i in range(0, 25, batch_size):
    batch_prompts = prompts[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    outputs = pipe(batch_prompts, max_new_tokens=100, temperature=0.001)
    for output, label in zip(outputs, batch_labels):
        text = output[0]['generated_text']
        answer = text.split("Antwoord alleen Ja of Nee met korte redenering:",1)[1]
        print(answer[:10])
        if 'Ja' in answer[:10]:
            pr='0'
            if label == '0':
                correct_match += 1
                print('got it good')
            else:
                print('missed a not one')
        else:
            pr='1'
            if label == '1':
                correct_match += 1
                print('got it, not one')
            else:
                print('missed one')
        pred.append(pr)
        correct_labels.append(label)

pred = [int(p) for p in pred]
correct_labels = [int(l) for l in correct_labels]
print(pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    correct_labels, pred, pos_label=0, average='binary'
)

print(f"Precision (for '0'): {precision:.4f}")
print(f"Recall (for '0'): {recall:.4f}")
print(f"F1-score (for '0'): {f1:.4f}")
print(correct_match / len(prompts))