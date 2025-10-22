import openai
import os
from sklearn.metrics import precision_recall_fscore_support




client = openai.OpenAI(api_key = "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")

long_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    De term "werk" omvat de intellectuele inhoud.

    Hier volgen een aantal voorbeelden.

    voorbeeld 1:

    record A: COL taal VAL ned COL jaar VAL 1974 COL titel VAL Een @handvol rogge COL hoofdauteur vermelding VAL Agatha Christie COL 2e auteur vermelding VAL [geaut. vert. uit het Engels] COL hoofdauteur VAL Christie, Agatha Christie 1890-1976 COL editie VAL [Herdr.] COL plaats VAL Leiden COL uitgever VAL Sijthoff COL omschrijving VAL Balans COL vertaling van VAL A pocket full of rye. - 1953
    record B: COL ISBN VAL 9021800500 COL taal VAL ned COL jaar VAL 1970 COL titel VAL Een @handvol rogge COL hoofdauteur vermelding VAL Agatha Christie COL 2e auteur vermelding VAL [geaut. vert. uit het Engels] COL hoofdauteur VAL Christie, Agatha Christie 1890-1976 COL editie VAL 2e dr COL plaats VAL Leiden COL uitgever VAL Sijthoff COL vertaling van VAL A pocket full of rye. - 1953
    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja

    voorbeeld 2:

    record A: COL taal VAL ned COL jaar VAL 1828 COL titel VAL @Eustachius, of De zegepraal van het christendom COL ondertitel VAL eene geschiedenis der vroegere christelijke eeuw COL hoofdauteur vermelding VAL door H.C. Schmid COL 2e auteur vermelding VAL [vert. uit het Duits] COL hoofdauteur VAL Schmid, Johann Christoph Friedrich von Schmid 1768-1854 COL plaats VAL Amsterdam COL uitgever VAL Ten Brink en De Vries COL omschrijving VAL Saakes 9 (1829), p. 16 COL vertaling van VAL Eustachius, eine Geschichte der christlichen Vorzeit. - 1828
    record B: COL taal VAL fra COL jaar VAL 1843 COL titel VAL @Eustache COL ondertitel VAL épisode des premiers temps du christianisme COL hoofdauteur vermelding VAL Christoph von Schmid COL 2e auteur vermelding VAL Louis Friedel [transl.] COL hoofdauteur VAL Schmid, Johann Christoph Friedrich von Schmid 1768-1854 COL 2e auteur VAL Friedel COL plaats VAL Tours COL uitgever VAL Mame

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja

    voorbeeld 3:

    record A: COL ISBN VAL 9022905349 COL taal VAL ned COL jaar VAL 1976 COL titel VAL De @Saint aan het stuur COL hoofdauteur vermelding VAL Leslie Charteris COL 2e auteur vermelding VAL vertaling [uit het Frans]: Maarten Beks COL hoofdauteur VAL Charteris, Leslie Charteris 1907-1993 COL 2e auteur VAL Beks, Martinus Arnoldus Maria Beks 1929-2001 COL plaats VAL Utrecht COL uitgever VAL A.W. Bruna & Zoon COL vertaling van VAL Le Saint au volant. - Paris : Fayard, 1961 uitgever VAL Sijthoff
    record B:COL taal VAL ned COL jaar VAL 1959 COL titel VAL De @Saint en de tijger COL hoofdauteur vermelding VAL Leslie Charteris COL 2e auteur vermelding VAL vertaling [uit het Engels]: Havank COL hoofdauteur VAL Charteris, Leslie Charteris 1907-1993 COL 2e auteur VAL Havank, Havank COL plaats VAL Utrecht COL uitgever VAL A.W. Bruna & Zoon COL vertaling van VAL Meet the tiger. - 1935

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Nee

    voorbeeld 4:
    record A: COL taal VAL ned COL jaar VAL 2013 COL titel VAL De @vreugde van het leven COL hoofdauteur vermelding VAL Catherine Cookson COL 2e auteur vermelding VAL vertaling [uit het Engels]: Annet Mons COL hoofdauteur VAL Cookson, Catherine Anne Cookson 1906-1998 COL 2e auteur VAL Mons, Annet Mons COL plaats VAL Amsterdam COL uitgever VAL Boekerij COL vertaling van VAL My beloved son. - London : Bantam Press, 1991
    record B: COL taal VAL ned COL jaar VAL 2013 COL titel VAL De @drempel van het leven COL hoofdauteur vermelding VAL Catherine Cookson COL 2e auteur vermelding VAL vertaling [uit het Engels]: Annet Mons COL hoofdauteur VAL Cookson, Catherine Anne Cookson 1906-1998 COL 2e auteur VAL Mons, Annet Mons COL plaats VAL Amsterdam COL uitgever VAL Boekerij COL vertaling van VAL The rag nymph. - New York : Bantam, ©1991
    
    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Nee

    nu jij:
"""

short_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    De term "werk" omvat de intellectuele inhoud.

"""
correct_match=0
with open("test_pairs_work.txt", "r", encoding="utf-8") as f:
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
            {"role": "user", "content": "Behoren deze records tot hetzelfde werk? Antwoord Ja of Nee:"}
        ],
        temperature=0
    )
    if 'Ja' in response.choices[0].message.content[:15]:
        pr='0'
    else:
        pr='1'
    if pr!=label:
        print(s1)
        print(s2)
        print(label)
    pred.append(pr)
    correct_labels.append(label)
    

pred = [int(p) for p in pred]
correct_labels = [int(l) for l in correct_labels]
print(pred)
print(correct_labels)
precision, recall, f1, _ = precision_recall_fscore_support(
    correct_labels, pred, pos_label=0, average='binary'
)
print(f"Precision (for '0'): {precision:.3f}")
print(f"Recall (for '0'): {recall:.3f}")
print(f"F1-score (for '0'): {f1:.3f}")
print(f"Accuracy: {accuracy:.3f}")


correct = pred == correct_labels
accuracy = correct.sum() / correct.size
print(f"Accuracy: {accuracy:.3f}")
