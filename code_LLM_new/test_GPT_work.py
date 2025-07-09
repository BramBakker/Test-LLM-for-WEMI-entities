import openai
import os
from sklearn.metrics import precision_recall_fscore_support


client = openai.OpenAI(api_key = "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")

long_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    Dit betekent: Hebben de twee boeken een gelijke intellectuele inhoud, dezelfde karakters en hetzelfde plot?
    
    Hier volgen wat voorbeelden.

    voorbeeld 1:
    
    record A: COL titel VAL Het grote voorleesboek COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Hulst, Willem Gerrit van de Hulst 1917-2006 COL editie VAL 11e
    record B: COL titel VAL Het grote voorleesboek COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Hulst, Willem Gerrit van de Hulst 1917-2006 COL editie VAL 5e dr

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja

    voorbeeld 2:

    record A: COL taal VAL ned COL titel VAL De @atoomtrillingen COL hoofdauteur VAL Toonder, Marten Toonder 1912-2005
    record B: COL taal VAL ned COL titel VAL De @toornviolen COL hoofdauteur VAL Toonder, Marten Toonder 1912-2005 COL Eerder verschenen in VAL Een heer moet alles alleen doen. - Amsterdam : De Bezige Bij, 1969. - (Literaire reuzenpocket ; 310) COL auteur/primair VAL Toonder
    
    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Nee

    voorbeeld 3:
    record A: COL taal VAL ned COL titel VAL @Eustachius COL hoofdauteur VAL Schmid, Johann Christoph Friedrich von Schmid 1768-1854
    record B: COL taal VAL fra COL titel VAL @Eustache COL ondertitel VAL épisode des premiers temps du christianisme COL hoofdauteur VAL Schmid, Johann Christoph Friedrich von Schmid 1768-1854 COL 2e auteur VAL Friedel

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja

    voorbeeld 4:
    record A: COL taal VAL ned COL titel VAL De @Saint aan het stuur COL hoofdauteur VAL Charteris, Leslie Charteris 1907-1993 COL 2e auteur VAL Beks, Martinus Arnoldus Maria Beks 1929-2001 COL vertaling van VAL le saint au volant. - Paris : Fayard, 1961
    record B: COL taal VAL ned COL titel VAL De @Saint en de tijger COL hoofdauteur VAL Charteris, Leslie Charteris 1907-1993 COL 2e auteur VAL Havank, Havank COL vertaling van VAL Meet the tiger. - ©1935

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Nee

    voorbeeld 5:
    record A: COL taal VAL ned COL titel VAL @Jungleboek COL hoofdauteur VAL Disney, Walter Elias Disney 1901-1966 COL 2e auteur VAL Kipling, Joseph Rudyard Kipling 1865-1936 COL 2e auteur VAL Marel, Joop v.d. Marel COL 2e auteur VAL Ekel, Pieter Ekel 1921-2012
    record B: COL taal VAL ned COL titel VAL Het @jungle-boek COL hoofdauteur VAL Disney, Walter Elias Disney 1901-1966 COL 2e auteur VAL Kipling, Joseph Rudyard Kipling 1865-1936Havank COL vertaling van VAL Meet the tiger. - ©1935

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja

    voorbeeld 6:
    record A: COL taal VAL ned COL titel VAL @Gelukkige Hans COL hoofdauteur VAL Grimm, Jacob Ludwig Carl Grimm 1785-1863
    record B: COL taal VAL ned COL titel VAL @Gelukkig Hansje COL ondertitel VAL eene vertelling voor kinderen COL hoofdauteur VAL Grimm, Jacob Ludwig Carl Grimm 1785-1863 COL 2e auteur VAL Schrijver der Historie van het huis van Adriaan, De Schrijver der Historie van het huis van Adriaan
    
    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja 

    nu jij:
"""
short_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    Dit betekent: Hebben de twee boeken een gelijke intellectuele inhoud, dezelfde karakters en hetzelfde plot?
"""
correct_match=0
with open("values_only_work_short.txt", "r", encoding="utf-8") as f:
    file = f.readlines()

base_messages=[
    {"role": "user", "content": short_prompt}
]
pred=[]
correct_labels=[]


for line in file:
    s1, s2, label = line.strip().split('\t')

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=base_messages+[
            {"role": "user", "content": f"Record A: {s1}"},
            {"role": "user", "content": f"Record B: {s2}"},
            {"role": "user", "content": "Behoren deze records tot dezelfde werk? Antwoord Ja of Nee:"}
        ],
        temperature=0
    )
    print(response.choices[0].message.content[:15])
    if 'Ja' in response.choices[0].message.content[:15]:
        pr='0'
        if label == '0':
            correct_match += 1
        else:
            print('missed a not one', s1)
            print(s2)
        pred.append(pr)

    else:
        pr='1'
        if label == '1':
            correct_match += 1
        else:
            print('missed one:', s1)
            print(s2)
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
print(correct_match / len(correct_labels))