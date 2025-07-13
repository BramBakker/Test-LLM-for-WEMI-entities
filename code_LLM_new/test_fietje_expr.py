
from sklearn.metrics import precision_recall_fscore_support
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

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
    Bepaal of twee boekenrecords tot hetzelfde expressie behoren.
    De term "expressie" omvat de specifieke inhoud in de vorm van woorden, zinnen en paragrafen.
    Dit betreft alleen aspecten die integraal onderdeel zijn van de (tekstuele) artistieke realisatie.
    Vertalingen, herziene edities of bewerkingen van hetzelfde werk worden als andere expressies beschouwd.

    Hier volgen een aantal voorbeelden.
    
    Voorbeeld 1:

    record A: COL taal VAL ned COL jaar VAL 1961 COL titel VAL @Van de boze koster! COL hoofdauteur vermelding VAL door W.G. van de Hulst COL 2e auteur vermelding VAL met tekeningen van W.G. van de Hulst Jr. COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Hulst, Willem Gerrit van de Hulst 1917-2006 COL editie VAL 15e dr., 142e-153e duiz COL plaats VAL Nijkerk COL uitgever VAL G.F. Callenbach
    record B: COL ISBN VAL 9026642520 COL taal VAL ned COL jaar VAL 1978 COL titel VAL @Van de boze koster COL hoofdauteur vermelding VAL W.G. van de Hulst COL 2e auteur vermelding VAL met zwarte en gekleurde tekn. van W.G. van de Hulst Jr COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Hulst, Willem Gerrit van de Hulst 1917-2006 COL editie VAL 20e dr COL plaats VAL Nijkerk COL uitgever VAL Callenbach COL taal VAL ned COL jaar VAL 1900 COL titel VAL @Tekstuitgaaf van de varkenshoeder COL ondertitel VAL een sprookje met zang en piano-begeleiding COL plaats VAL Schiedam COL uitgever VAL Roelants
    
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Ja
    
    Voorbeeld 2:

    record A: COL taal VAL fra COL jaar VAL 1920 COL titel VAL @Sans famille COL hoofdauteur vermelding VAL par Hector Malot COL hoofdauteur VAL Malot, Hector Henri Malot 1830-1907 COL plaats VAL Paris COL uitgever VAL Fayard
    record B: COL taal VAL fra COL jaar VAL 1954 COL titel VAL @Sans famille COL hoofdauteur vermelding VAL Hector Malot COL 2e auteur vermelding VAL Marianne Clouzot [ill.] COL hoofdauteur VAL Malot, Hector Henri Malot 1830-1907 COL 2e auteur VAL Clouzot COL plaats VAL Paris COL uitgever VAL Hachette

    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Ja

    Voorbeeld 3:

    record A: COL ISBN VAL 3855397937 COL taal VAL ned COL jaar VAL 1987 COL titel VAL De @varkenshoeder COL ondertitel VAL een sprookje van Hans Christian Andersen COL hoofdauteur vermelding VAL vert. [uit het Duits naar de oorspr. Deense uitg.] door Ineke Ris COL 2e auteur vermelding VAL met ill. van Dorothée Duntze COL hoofdauteur VAL Andersen, Hans Christian Andersen 1805-1875 COL 2e auteur VAL Ris, Ineke Ris COL 2e auteur VAL Duntze, Dorothée Duntze 1960- COL plaats VAL Den Haag COL uitgever VAL De Vier Windstreken COL vertaling van VAL Der Schweinehirt. - Mönchaltorf : Nord-Süd Verlag, cop. 1987 COL auteur/primair VAL Andersen
    record B:COL ISBN VAL 3851951212 COL taal VAL dui COL jaar VAL 1982 COL titel VAL Der @Schweinehirt COL hoofdauteur vermelding VAL H.C. Andersen COL 2e auteur vermelding VAL [ill.] Lisbeth Zwerger COL hoofdauteur VAL Andersen, Hans Christian Andersen 1805-1875 COL 2e auteur VAL Zwerger, Lisbeth Zwerger 1954- COL plaats VAL Salzburg COL uitgever VAL Verlag Neugebauer Press
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Nee

    Voorbeeld 4:

    record A: COL taal VAL ned COL jaar VAL 2014 COL titel VAL @Omdat ik zoveel van je hou COL hoofdauteur vermelding VAL Guido van Genechten COL hoofdauteur VAL Van Genechten, Guido Van Genechten 1957- COL editie VAL Zesde herziene druk COL plaats VAL [Amsterdam] COL uitgever VAL Clavis
    record B: COL taal VAL ned COL jaar VAL 2016 COL titel VAL @Omdat ik zoveel van je hou COL hoofdauteur vermelding VAL Guido van Genechten COL hoofdauteur VAL van Genechten, Guido Van Genechten 1957- COL editie VAL Eerste druk editie klein formaat COL plaats VAL [Amsterdam] COL uitgever VAL Clavis
    
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Nee

    nu jij:

Record A: {record_a}

Record B: {record_b}

Behoren deze records tot hetzelfde werk? Antwoord alleen Ja of Nee met korte redenering:

"""
batch_size = 1  # You can increase this if you have enough GPU memory

prompts = []
labels = []
with open("short_pairs_expr.txt", "r", encoding="utf-8") as f:
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