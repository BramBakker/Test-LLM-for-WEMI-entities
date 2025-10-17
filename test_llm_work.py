
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sklearn.metrics import precision_recall_fscore_support

import os

# Set your API key
hf_token = "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"
model_id = "meta-llama/Llama-3.3-70B-Instruct"

# Load model & tokenizer

device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto"
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


# Create pipeline
def make_prompt(record_a: str, record_b: str) -> str:
    return f"""
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    De term "werk" omvat de intellectuele inhoud, zoals plot en karakters.
    Parafrases, herschrijvingen, parodieën, aanpassingen en samenvattingen worden beschouwd als amdere werken. 
    Nieuwe edities en vertalingen behoren wel tot hetzelfde werk.

    voorbeeld 1:

    record A: COL taal VAL ned COL jaar VAL 1974 COL titel VAL Een @handvol rogge COL extra annotatie VAL [geaut. vert. uit het Engels] COL hoofdauteur VAL Christie, Agatha Christie 1890-1976 COL editie VAL [Herdr.] COL plaats VAL Leiden COL uitgever VAL Sijthoff COL omschrijving VAL Balans COL vertaling van VAL A pocket full of rye. - 1953
    record B: COL ISBN VAL 9021800500 COL taal VAL ned COL jaar VAL 1970 COL titel VAL Een @handvol rogge COL extra annotatie VAL [geaut. vert. uit het Engels] COL hoofdauteur VAL Christie, Agatha Christie 1890-1976 COL editie VAL 2e dr COL plaats VAL Leiden COL uitgever VAL Sijthoff COL vertaling van VAL A pocket full of rye. - 1953

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja

    voorbeeld 2:
    
    record A: COL taal VAL ned COL jaar VAL 1963 COL titel VAL De @Saint aan het stuur COL hoofdauteur VAL Charteris, Leslie Charteris 1907-1993 COL 2e auteur VAL Beks, Martinus Arnoldus Maria Beks 1929-2001 COL plaats VAL Utrecht COL uitgever VAL Bruna COL vertaling van VAL le saint au volant. - Paris : Fayard, 1961
    record B: COL ISBN VAL 9022902196 COL taal VAL ned COL jaar VAL 1980 COL titel VAL De @Saint en de tijger COL hoofdauteur VAL Charteris, Leslie Charteris 1907-1993 COL 2e auteur VAL Havank, Havank COL plaats VAL Utrecht COL uitgever VAL A.W. Bruna & Zoon COL vertaling van VAL Meet the tiger. - ©1935

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Nee

    nu jij:
    
    Record A: {record_a}
    Record B: {record_b}

    Behoren deze records tot hetzelfde werk? Antwoord alleen Ja of Nee:

"""
def make_prompt_short(record_a: str, record_b: str) -> str:
    return f"""
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    De term "werk" omvat de intellectuele inhoud, zoals plot en karakters.
    Parafrases, herschrijvingen, parodieën, aanpassingen en samenvattingen worden beschouwd als amdere werken. 
    Nieuwe edities en vertalingen behoren wel tot hetzelfde werk.
    
    Record A: {record_a}
    Record B: {record_b}
    
    Behoren deze records tot hetzelfde werk? Antwoord alleen Ja of Nee:
"""

def make_prompt_super_short(record_a: str, record_b: str) -> str:
    return f"""
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    
    Record A: {record_a}
    Record B: {record_b}
    
    Behoren deze records tot hetzelfde werk? Antwoord alleen Ja of Nee:
"""
batch_size = 2  # You can increase this if you have enough GPU memory

prompts = []
labels = []
with open("/scratch/bbakker/test_pairs_work.txt", "r", encoding="utf-8") as f:
    for line in f:
        s1, s2, label = line.strip().split('\t')
        prompts.append(make_prompt(s1, s2))
        labels.append(label)

correct_labels=[]
pred=[]
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    outputs = pipe(batch_prompts, max_new_tokens=100, temperature=0.001)
    for output, label in zip(outputs, batch_labels):
        text = output[0]['generated_text']
        answer = text.split("Antwoord alleen Ja of Nee:",1)[1]
        if 'Ja' in answer[:20]:
            pr='0'
        else:
            pr='1'
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