
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os

hf_token = ""
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



def make_prompt(record_a: str, record_b: str) -> str:
    return f"""
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde expressie behoren.
    De term "expressie" omvat de specifieke inhoud in de vorm van woorden, zinnen en paragrafen.
    Dit betreft alleen aspecten die integraal onderdeel zijn van de (tekstuele) artistieke realisatie.
    Vertalingen, herziene edities of bewerkingen van hetzelfde werk worden als andere expressies beschouwd.
    
    Voorbeeld 1:

    record A: COL taal VAL ned COL jaar VAL 1961 COL titel VAL @Van de boze koster! COL hoofdauteur vermelding VAL door W.G. van de Hulst COL 2e auteur vermelding VAL met tekeningen van W.G. van de Hulst Jr. COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Hulst, Willem Gerrit van de Hulst 1917-2006 COL editie VAL 15e dr., 142e-153e duiz COL plaats VAL Nijkerk COL uitgever VAL G.F. Callenbach
    record B: COL ISBN VAL 9026642520 COL taal VAL ned COL jaar VAL 1978 COL titel VAL @Van de boze koster COL hoofdauteur vermelding VAL W.G. van de Hulst COL 2e auteur vermelding VAL met zwarte en gekleurde tekn. van W.G. van de Hulst Jr COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Hulst, Willem Gerrit van de Hulst 1917-2006 COL editie VAL 20e dr COL plaats VAL Nijkerk COL uitgever VAL Callenbach COL taal VAL ned COL jaar VAL 1900 COL titel VAL @Tekstuitgaaf van de varkenshoeder COL ondertitel VAL een sprookje met zang en piano-begeleiding COL plaats VAL Schiedam COL uitgever VAL Roelants
    
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Ja

    nu jij:

    Record A: {record_a}
    Record B: {record_b}

Behoren deze records tot dezelfde expressie? Antwoord alleen Ja of Nee:

"""
def make_prompt_short(record_a: str, record_b: str) -> str:
    return f"""
    Je bent een data assistent.
    Bepaal of twee boekenrecords tot dezelfde expressie behoren.
    De term "expressie" omvat de specifieke inhoud in de vorm van woorden, zinnen en paragrafen.
    Dit betreft alleen aspecten die integraal onderdeel zijn van de (tekstuele) artistieke realisatie.
    Vertalingen, herziene edities of bewerkingen van hetzelfde werk worden als andere expressies beschouwd.
    
    Record A: {record_a}
    Record B: {record_b}
    
    Behoren deze records tot hetzelfde expressie? Antwoord alleen Ja of Nee:
"""



batch_size = 1  # You can increase this if you have enough GPU memory

prompts = []
labels = []
with open("/scratch/bbakker/short_pairs_expr.txt", "r", encoding="utf-8") as f:
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

pred = np.array([int(p) for p in pred])
correct_labels = np.array([int(l) for l in correct_labels])
print(pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    correct_labels, pred, pos_label=0, average='binary'
)

print(f"Precision (for '0'): {precision:.3f}")
print(f"Recall (for '0'): {recall:.3f}")
print(f"F1-score (for '0'): {f1:.3f}")

correct = pred == correct_labels
accuracy = correct.sum() / correct.size
print(f"Accuracy: {accuracy:.3f}")
