import openai
import os
from sklearn.metrics import precision_recall_fscore_support


client = openai.OpenAI(api_key = "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")

long_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    De term "werk" omvat de intellectuele inhoud, zoals plot en karakters.
    Parafrases, herschrijvingen, parodieën, aanpassingen en samenvattingen worden beschouwd als amdere werken. 
    Nieuwe edities en vertalingen behoren wel tot hetzelfde werk.

    Hier volgen wat voorbeelden.

    voorbeeld 1:

    record A: COL taal VAL ned COL jaar VAL 1974 COL titel VAL Een @handvol rogge COL extra annotatie VAL [geaut. vert. uit het Engels] COL hoofdauteur VAL Christie, Agatha Christie 1890-1976 COL editie VAL [Herdr.] COL plaats VAL Leiden COL uitgever VAL Sijthoff COL omschrijving VAL Balans COL vertaling van VAL A pocket full of rye. - 1953
    record B: COL ISBN VAL 9021800500 COL taal VAL ned COL jaar VAL 1970 COL titel VAL Een @handvol rogge COL extra annotatie VAL [geaut. vert. uit het Engels] COL hoofdauteur VAL Christie, Agatha Christie 1890-1976 COL editie VAL 2e dr COL plaats VAL Leiden COL uitgever VAL Sijthoff COL vertaling van VAL A pocket full of rye. - 1953

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja

    voorbeeld 2:

    record A: COL taal VAL ned COL jaar VAL 1920 COL titel VAL @Eustachius COL hoofdauteur VAL Schmid, Johann Christoph Friedrich von Schmid 1768-1854 COL editie VAL 2e dr COL plaats VAL Sneek COL uitgever VAL Boeijenga COL omschrijving VAL (In vereenvoudigde spelling) COL omschrijving VAL Ondertitel op omslag: De veldheer der Romeinen : een verhaal uit de eerste Christentijden
    record B: COL taal VAL fra COL jaar VAL 1843 COL titel VAL @Eustache COL ondertitel VAL épisode des premiers temps du christianisme COL hoofdauteur VAL Schmid, Johann Christoph Friedrich von Schmid 1768-1854 COL 2e auteur VAL Friedel COL plaats VAL Tours COL uitgever VAL Mame

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Ja

    voorbeeld 3:

    record A: COL taal VAL ned COL jaar VAL 1963 COL titel VAL De @Saint aan het stuur COL hoofdauteur VAL Charteris, Leslie Charteris 1907-1993 COL 2e auteur VAL Beks, Martinus Arnoldus Maria Beks 1929-2001 COL plaats VAL Utrecht COL uitgever VAL Bruna COL vertaling van VAL le saint au volant. - Paris : Fayard, 1961
    record B: COL ISBN VAL 9022902196 COL taal VAL ned COL jaar VAL 1980 COL titel VAL De @Saint en de tijger COL hoofdauteur VAL Charteris, Leslie Charteris 1907-1993 COL 2e auteur VAL Havank, Havank COL plaats VAL Utrecht COL uitgever VAL A.W. Bruna & Zoon COL vertaling van VAL Meet the tiger. - ©1935

    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Nee

    voorbeeld 4:
    record A: COL taal VAL ned COL jaar VAL 2013 COL titel VAL De @vreugde van het leven COL extra annotatie VAL vertaling [uit het Engels]: Annet Mons COL hoofdauteur VAL Cookson, Catherine Anne Cookson 1906-1998 COL 2e auteur VAL Mons, Annet Mons COL plaats VAL Amsterdam COL uitgever VAL Boekerij COL vertaling van VAL My beloved son. - London : Bantam Press, 1991
    record B: COL taal VAL ned COL jaar VAL 2013 COL titel VAL De @drempel van het leven COL extra annotatie VAL vertaling [uit het Engels]: Annet Mons COL hoofdauteur VAL Cookson, Catherine Anne Cookson 1906-1998 COL 2e auteur VAL Mons, Annet Mons COL plaats VAL Amsterdam COL uitgever VAL Boekerij COL vertaling van VAL The rag nymph. - New York : Bantam, ©1991
    
    Behoren deze boeken tot hetzelfde werk? Antwoord Ja of Nee: 
    reactie: Nee

    nu jij:
"""

short_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot hetzelfde werk behoren.
    De term "werk" omvat de intellectuele inhoud, zoals plot en karakters.
    Parafrases, herschrijvingen, parodieën, aanpassingen en samenvattingen worden beschouwd als amdere werken. 
    Nieuwe edities en vertalingen behoren wel tot hetzelfde werk.
"""

super_short_prompt = """
    Je bent een data assistent. 
"""
correct_match=0
with open("short_pairs_work.txt", "r", encoding="utf-8") as f:
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
            {"role": "user", "content": "Behoren deze records tot hetzelfde werk? Antwoord Ja of Nee:"}
        ],
        temperature=0
    )
    if 'Ja' in response.choices[0].message.content[:15]:
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