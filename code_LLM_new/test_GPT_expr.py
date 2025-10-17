import openai
import os
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


long_prompt = """
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

"""
short_prompt = """
    Je bent een data assistent.
    Bepaal of twee boekenrecords tot dezelfde expressie behoren.
    De term "expressie" omvat de specifieke inhoud in de vorm van woorden, zinnen en paragrafen.
    Dit betreft alleen aspecten die integraal onderdeel zijn van de (tekstuele) artistieke realisatie.
    Vertalingen, herziene edities of bewerkingen van hetzelfde werk worden als andere expressies beschouwd.
"""

with open("test_pairs_expr.txt", "r", encoding="utf-8") as f:
    file = f.readlines()

base_messages=[
    {"role": "user", "content": long_prompt}
]
pred=[]
correct_labels=[]

for line in file:
    s1, s2, label = line.strip().split('\t')

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=base_messages+[
            {"role": "user", "content": f"Record A: {s1}"},
            {"role": "user", "content": f"Record B: {s2}"},
            {"role": "user", "content": "Behoren deze records tot dezelfde expressie? Antwoord alleen Ja of Nee:"}
        ],
        temperature=0
    )
    #print(response.choices[0].message.content)
    if 'Ja' in response.choices[0].message.content:
        pr='0'
    else:
        pr='1'
    if pr!=label:
        print(pr)
        print(label)
        print(s1)
        print(s2)
        print(label)
    pred.append(pr)
    if pr!=label:
        print(s1)
        print(s2)
        print(label)
    correct_labels.append(label)


pred = np.array([int(p) for p in pred])
correct_labels = np.array([int(l) for l in correct_labels])

print(pred)
print(correct_labels)

precision, recall, f1, _ = precision_recall_fscore_support(
    correct_labels, pred, pos_label=0, average='binary'
)
print(f"Precision (for '0'): {precision:.3f}")
print(f"Recall (for '0'): {recall:.3f}")
print(f"F1-score (for '0'): {f1:.3f}")

# Now this works
correct = pred == correct_labels
accuracy = correct.sum() / correct.size
print(f"Accuracy: {accuracy:.3f}")

