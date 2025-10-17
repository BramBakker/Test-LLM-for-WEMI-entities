import openai
from sklearn.metrics import precision_recall_fscore_support

client = openai.OpenAI(api_key = "")

long_prompt = """
Je bent een data assistent die op basis van metadata boeken dezelfde WEMI expressies zijn, in tegenstelling tot vertalingen of herziene edities.
Sommige boeken hebben net andere metadata (titel uitgever jaar editie) maar exact dezelfde tekstuele inhoud, die hebben dus dezelfde expressie.
Let goed op het veld "editie, hier kan bijvoorbeeld staan "herz. dr". Dit is dan altijd een andere expressie.
Verschillende edities kunnen tot dezelfde expressie behoren.
Auteursnamen en titels kunnen vaak verschillen en net anders geschreven staan, ook staat de auteur soms in andere velden, bijvoorbeeld auteur en hoofdauteur
gedeelde ISBN kan een aanwijzing zijn voor 'Ja', vertalingen en herziene drukken en extra auteurs zijn belangrijkste aanwijzing voor 'Nee'

    Hier volgen wat vorobeelden.
    
    Voorbeeld 1:
    record A: COL ISBN VAL 9026642520 COL bind VAL geb. COL taal VAL ned COL jaar VAL 1978 COL titel VAL @Van de boze koster COL auteur VAL Hulst COL editie VAL 20e dr COL plaats VAL Nijkerk COL uitgever VAL Callenbach COL pagina’s VAL 47 blz 	
    record B: COL taal VAL ned COL jaar VAL 1966 COL titel VAL @Van de boze koster COL hoofdauteur VAL Hulst COL editie VAL 17e dr COL plaats VAL Nijkerk COL uitgever VAL G.F. Callenbach COL pagina’s VAL 48 p
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee met een korte redenering: 
    reactie: Ja (ze delen dezelfde titel en taal, geen vertaling dus)

    Voorbeeld 2:
    record A: COL ISBN VAL 9026906544 COL taal VAL ned COL jaar VAL 1987 COL titel VAL Het @slot op de Hoef COL hoofdauteur VAL Kieviet, Cornelis Johannes Kieviet 1858-1931 COL 2e auteur VAL Poortvliet, Marinus Harm Poortvliet 1932-1995 COL 2e auteur VAL Minderhoud, Joost Minderhout COL editie VAL 15e, herz. dr COL plaats VAL Houten COL uitgever VAL Van Holkema & Warendorf COL omschrijving VAL Omslagillustratie van Joost Minderhoud COL Oorspr. titel VAL Het slot Op den Hoef : een verhaal uit den tijd van Ada van Holland. - Amersfoort : Valkhoff & Van den Dries, 1897
    record B: COL taal VAL ned COL jaar VAL 1937 COL titel VAL Het @slot op den Hoef COL ondertitel VAL een verhaal uit de tijd van Ada van Holland COL hoofdauteur VAL Kieviet, Cornelis Johannes Kieviet 1858-1931 COL 2e auteur VAL Reinderhoff, Marie-Louise Reinderhoff 1903-1991 COL editie VAL 7e dr COL plaats VAL Amsterdam COL uitgever VAL Van Holkema & Warendorf
    reactie: Nee (ze delen dezelfde titel en taal, maar de record A is een herziene druk, dat blijkt uit "herz. druk")
    
    Voorbeeld 3:
    record A: COL ISBN VAL 3855397937 COL taal VAL ned COL jaar VAL 1987 COL titel VAL De @varkenshoeder COL ondertitel VAL een sprookje van Hans Christian Andersen COL hoofdauteur VAL Andersen, Hans Christian Andersen 1805-1875 COL 2e auteur VAL Ris, Ineke Ris COL 2e auteur VAL Duntze, Dorothée Duntze 1960- COL plaats VAL Den Haag COL uitgever VAL De Vier Windstreken COL vertaling van VAL Der Schweinehirt. - Mönchaltorf : Nord-Süd Verlag, cop. 1987 COL auteur/primair VAL Andersen 
    record B: COL ISBN VAL 3851951212 COL taal VAL dui COL jaar VAL 1982 COL titel VAL Der @Schweinehirt COL hoofdauteur VAL Andersen, Hans Christian Andersen 1805-1875 COL 2e auteur VAL Zwerger, Lisbeth Zwerger 1954- COL plaats VAL Salzburg COL uitgever VAL Verlag Neugebauer Press 
    reactie: Nee (dit is een vertaling, dus altijd een andere expressie)

    Voorbeeld 4:
    record A: COL taal VAL dui COL jaar VAL 1928 COL titel VAL Die @Komödianten COL ondertitel VAL Roman COL hoofdauteur VAL Couperus, Louis Marie Anne Couperus 1863-1923 COL 2e auteur VAL Otten, Else Otten 1873-1931 COL uitgever VAL Zenith-Verlag COL vertaling van VAL De komedianten. - Rotterdam : Nijgh & Van Ditmar, 1917. - Oorspr. verschenen in het tijdschrift Groot Nederland. - 1917	
    record B: COL taal VAL ned COL jaar VAL 1968 COL titel VAL De @komedianten COL hoofdauteur VAL Couperus, Louis Marie Anne Couperus 1863-1923 COL uitgever VAL Athenaeum-Polak en Van Gennep	1
    reactie: Nee (dit is een vertaling, dus altijd een andere expressie)

    Voorbeeld 5:
    record A: COL taal VAL ned COL jaar VAL 1964 COL titel VAL @Morgen... is laat COL hoofdauteur VAL Saris, Helena Barbara Wilhelmina Saris 1915-1999 COL 2e auteur VAL Borrebach, Henri Carl Johan Elisa Borrebach 1903-1991 COL uitgever VAL West-Friesland
    record B: COL taal VAL ned COL jaar VAL 1996 COL titel VAL @Morgen is laat COL hoofdauteur VAL Saris, Helena Barbara Wilhelmina Saris 1915-1999 COL uitgever VAL Iris Oirschot
    reactie: Nee (Hier is een nieuwe auteur die illustraties heeft gedaan, een nieuwe expressie dus)

    Voorbeeld 6:
    record A: COL taal VAL ned COL titel VAL @Veel geluk! COL ondertitel VAL nieuwjaars- en verjaardagsversjes voor kinderen COL hoofdauteur VAL Louwerse, Pieter Louwerse 1840-1908 COL editie VAL Vijfde, verbeterde druk COL uitgever VAL S. & W.N. van Nooten
    record B: COL taal VAL ned COL titel VAL @Veel geluk COL ondertitel VAL nieuwjaars- en verjaardagsversjes voor kinderen COL hoofdauteur VAL Louwerse, Pieter Louwerse 1840-1908 COL editie VAL Zesde druk COL uitgever VAL S. & W.N. van Nooten
    reactie: Ja (De zesde druk komt na de vijfde verbeterde druk, en is dus dezelfde expressie ondanks de verschillende titel)

    nu jij:

"""
short_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot dezelfde WEMI expressie behoren.
    Bedenk goed wat een WEMI expressie is. 
    Het betekent kort gezegd: Hebben de twee boeken een exact dezelfde tekstuele inhoud?
    Nieuwe vertalingen, herziene edities, edities met een nieuw voorwoord behoren tot verschillende expressies.
    Reprints, nieuwe drukken en edities van hetzelfde boek ZONDER aanpassing behoren tot dezelfde expressie.

"""
orig_prompt= """
Je bent een data assistent die op basis van metadata boeken dezelfde WEMI expressies zijn, in tegenstelling tot vertalingen of herziene edities.
Sommige boeken zijn van andere edities en hebben dus andere metadata (titel uitgever jaar editie) maar dezelfde inhoud. 
Verschillende edities hebben dus vaak dezelfde expressie.
gedeelde ISBN kan een aanwijzing zijn voor JA, vertalingen en herziene drukken en extra auteurs zijn belangrijkste aanwijzing voor NEE
De potentiele paren komen van hetzelfde werk, dus er is een zeer grote kans op JA, zelfs als de uitgever anders is, of de auteur lijkt net anders geschreven te zijn.
Auteursnamen en titels kunnen vaak verschillen en net anders geschreven staan, ook staat de auteur soms in andere velden, bijvoorbeeld auteur en hoofdauteur

De data is georganiseerd in "COL <veld> VAL <waarde>".
Voorbeeld:

record A: COL ISBN VAL 9026642520 COL bind VAL geb. COL taal VAL ned COL jaar VAL 1978 COL titel VAL @Van de boze koster COL auteur VAL Hulst COL editie VAL 20e dr COL plaats VAL Nijkerk COL uitgever VAL Callenbach COL pagina’s VAL 47 blz 	

record B: COL taal VAL ned COL jaar VAL 1966 COL titel VAL @Van de boze koster COL hoofdauteur VAL Hulst COL editie VAL 17e dr COL plaats VAL Nijkerk COL uitgever VAL G.F. Callenbach COL pagina’s VAL 48 p

Hebben deze records dezelfde expressie? Antwoord JA of NEE met korte redenering: 
reactie: JA 
ze delen dezelfde titel en taal, geen vertaling dus

(trouwens, het is erg vaak JA, ook al lijkt de data niet altijd precies op elkaar)
"""
correct_match=0
with open("short_pairs_expr.txt", "r", encoding="utf-8") as f:
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
            {"role": "user", "content": "Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee met een zeer korte redenering:"}
        ],
        temperature=0
    )
    print(response.choices[0].message.content)
    if 'Ja' in response.choices[0].message.content:
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