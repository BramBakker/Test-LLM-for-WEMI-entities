import openai
import os
from sklearn.metrics import precision_recall_fscore_support

# Set your API key
client = openai.OpenAI(api_key = "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")


long_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde expressie behoren.
    De term "expressie" omvat de specifieke inhoud in de vorm van woorden, zinnen en paragrafen.
    De expressiegrenzen sluiten aspecten uit die geen integraal onderdeel zijn van deze artistieke realisatie van het werk.
    Vertalingen, herziene edities of bewerkingen van hetzelfde werk worden als andere expressies beschouwd.

    Hier volgen wat voorbeelden.
    
    Voorbeeld 1:

    record A: COL taal VAL ned COL jaar VAL 1956 COL titel VAL @Van de boze koster! COL 2e auteur annotatie VAL met tek. van Tjeerd Bottema COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Bottema, Tjeerd Bottema 1884-1978 COL editie VAL 13e dr COL plaats VAL Nijkerk COL uitgever VAL G.F. Callenbach
    record B: COL taal VAL ned COL jaar VAL 1966 COL titel VAL @Van de boze koster COL hoofdauteur VAL Hulst COL editie VAL 17e dr COL plaats VAL Nijkerk COL uitgever VAL G.F. Callenbach COL pagina’s VAL 48 p
    
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Ja
    
    Voorbeeld 2:

    record A: COL taal VAL fra COL jaar VAL 1920 COL titel VAL @Sans famille COL hoofdauteur VAL Malot, Hector Henri Malot 1830-1907 COL plaats VAL Paris COL uitgever VAL Fayard
    record B: COL taal VAL fra COL jaar VAL 189X COL titel VAL @Sans Familie COL 2e auteur annotatie VAL ill. par É. Bayard COL hoofdauteur VAL Malot, Hector Henri Malot 1830-1907 COL 2e auteur VAL Bayard, Émile Antoine Bayard 1837-1891 COL plaats VAL Paris COL uitgever VAL Librairie Hachette

    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Ja

    Voorbeeld 3:

    record A: COL ISBN VAL 3855397937 COL taal VAL ned COL jaar VAL 1987 COL titel VAL De @varkenshoeder COL ondertitel VAL een sprookje van Hans Christian Andersen COL hoofdauteur VAL Andersen, Hans Christian Andersen 1805-1875 COL 2e auteur VAL Ris, Ineke Ris COL 2e auteur VAL Duntze, Dorothée Duntze 1960- COL plaats VAL Den Haag COL uitgever VAL De Vier Windstreken COL vertaling van VAL Der Schweinehirt. - Mönchaltorf : Nord-Süd Verlag, cop. 1987 COL auteur/primair VAL Andersen 
    record B: COL ISBN VAL 3851951212 COL taal VAL dui COL jaar VAL 1982 COL titel VAL Der @Schweinehirt COL hoofdauteur VAL Andersen, Hans Christian Andersen 1805-1875 COL 2e auteur VAL Zwerger, Lisbeth Zwerger 1954- COL plaats VAL Salzburg COL uitgever VAL Verlag Neugebauer Press 
    
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Nee

    Voorbeeld 4:

    record A: CCOL taal VAL ned COL jaar VAL 2014 COL titel VAL @Omdat ik zoveel van je hou COL hoofdauteur VAL Van Genechten, Guido Van Genechten 1957- COL editie VAL Zesde herziene druk COL plaats VAL [Amsterdam] COL uitgever VAL Clavis 	
    record B: COL taal VAL ned COL jaar VAL 2016 COL titel VAL @Omdat ik zoveel van je hou COL hoofdauteur VAL van Genechten, Guido Van Genechten 1957- COL editie VAL Eerste druk editie klein formaat COL plaats VAL [Amsterdam] COL uitgever VAL Clavis
    
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee: 
    reactie: Nee
    
    nu jij:

"""
short_prompt = """
    Je bent een data assistent.
    Bepaal of twee boekenrecords tot dezelfde expressie behoren.
    De term "expressie" omvat de specifieke inhoud in de vorm van woorden, zinnen en paragrafen.
    De expressiegrenzen sluiten aspecten uit die geen integraal onderdeel zijn van deze artistieke realisatie van het werk.
    Vertalingen, herziene edities of bewerkingen van hetzelfde werk worden als andere expressies beschouwd.
"""

super_short_prompt = """
    Je bent een data assistent.
    Bepaal of twee boekenrecords tot dezelfde expressie behoren.
"""


with open("test_pairs_expr.txt", "r", encoding="utf-8") as f:
    file = f.readlines()

base_messages=[
    {"role": "user", "content": short_prompt}
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
    pred.append(pr)
    correct_labels.append(label)
    

pred = [int(p) for p in pred]
correct_labels = [int(l) for l in correct_labels]
print(pred)
print(correct_labels)
precision, recall, f1, _ = precision_recall_fscore_support(
    correct_labels, pred, pos_label=0, average='binary'
)
print(f"Precision (for '0'): {precision:.4f}")
print(f"Recall (for '0'): {recall:.4f}")
print(f"F1-score (for '0'): {f1:.4f}")
correct = (pred == correct_labels)
accuracy = correct.sum() / correct.size