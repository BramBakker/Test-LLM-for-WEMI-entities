import openai
import os
from sklearn.metrics import precision_recall_fscore_support

# Set your API key






#morgen:
#Check in detail de expression pairs en bedenk daar nieuwe regels voor, te weinig tijd om alles aan te  passen
#Krijg zo snel mogelijk resultaat, kijk of je Llama thuis kan runnen en Fietje resultaten met dezelfde work dingen
#Kijk het verschil voor Work met extra annotatie informatie
#The KB clearly means ofr expressions that we only consider textual content
#We don't care about minor paratextual variation

#er zit een logica in
#Schrijven!!!

#COL ISBN VAL 902110430X COL taal VAL ned COL jaar VAL 1973 COL titel VAL @Harold de vriend van Bonifatius COL extra annotatie VAL geïllustreerd door P.A. Donkersloot COL hoofdauteur VAL Zeeuw, Pieter de Zeeuw J.Gzn. 1890-1968 COL 2e auteur VAL Donkersloot, Pieter Abraham Donkersloot 1930-1986 COL editie VAL 2e herz.dr COL plaats VAL Delft COL uitgever VAL W.D. Meinema
#COL taal VAL ned COL jaar VAL 1976 COL titel VAL @Harold, de vriend van Bonifatius COL extra annotatie VAL [geïllustreerd door P. A. Donkersloot] COL hoofdauteur VAL Zeeuw, Pieter de Zeeuw J.Gzn. 1890-1968 COL 2e auteur VAL Donkersloot, Pieter Abraham Donkersloot 1930-1986 COL editie VAL 5e dr COL plaats VAL Amsterdam COL uitgever VAL Vereeniging tot verspreiding der Heilige Schrift






long_prompt = """
Je bent een data assistent die op basis van metadata boeken dezelfde WEMI expressies zijn, in tegenstelling tot vertalingen of herziene edities.

Belangrijke feiten:
1. Let goed op het veld "editie, hier kan bijvoorbeeld staan "herz. dr". Dit is dan altijd een andere expressie.
2. Verschillende edities kunnen tot dezelfde expressie behoren.
3. Auteursnamen en titels kunnen vaak verschillen en net anders geschreven staan, ook staat de auteur soms in andere velden, bijvoorbeeld auteur en hoofdauteur
4. Het gaat om significante tekstuele veranderingen, dus alleen ander voorwoord, andere uitgever of andere illustrator/coverart: ZELFDE expressie
5. Andere redactie en editors: NIET zelfde expressie


gedeelde ISBN kan een aanwijzing zijn voor 'Ja', vertalingen en herziene drukken en extra auteurs zijn belangrijkste aanwijzing voor 'Nee'

    Hier volgen wat vorobeelden.
    
    Voorbeeld 1:
    record A: COL taal VAL ned COL jaar VAL 1956 COL titel VAL @Van de boze koster! COL extra annotatie VAL met tek. van Tjeerd Bottema COL hoofdauteur VAL Hulst, Willem Gerrit van de Hulst 1879-1963 COL 2e auteur VAL Bottema, Tjeerd Bottema 1884-1978 COL editie VAL 13e dr COL plaats VAL Nijkerk COL uitgever VAL G.F. Callenbach
    record B: COL taal VAL ned COL jaar VAL 1966 COL titel VAL @Van de boze koster COL hoofdauteur VAL Hulst COL editie VAL 17e dr COL plaats VAL Nijkerk COL uitgever VAL G.F. Callenbach COL pagina’s VAL 48 p
    Behoren deze records tot dezelfde expressie? Antwoord Ja of Nee met een korte redenering: 
    reactie: Ja (ze delen bijna dezelfde titel en taal, geen vertaling dus, andere illustrator alleen zijn niet transformatief genoeg)

    Voorbeeld 2:
    record A: COL ISBN VAL 9026906544 COL taal VAL ned COL jaar VAL 1987 COL titel VAL Het @slot op de Hoef COL hoofdauteur VAL Kieviet, Cornelis Johannes Kieviet 1858-1931 COL 2e auteur VAL Poortvliet, Marinus Harm Poortvliet 1932-1995 COL 2e auteur VAL Minderhoud, Joost Minderhout COL editie VAL 15e, herz. dr COL plaats VAL Houten COL uitgever VAL Van Holkema & Warendorf COL omschrijving VAL Omslagillustratie van Joost Minderhoud COL Oorspr. titel VAL Het slot Op den Hoef : een verhaal uit den tijd van Ada van Holland. - Amersfoort : Valkhoff & Van den Dries, 1897
    record B: COL taal VAL ned COL jaar VAL 1937 COL titel VAL Het @slot op den Hoef COL ondertitel VAL een verhaal uit de tijd van Ada van Holland COL hoofdauteur VAL Kieviet, Cornelis Johannes Kieviet 1858-1931 COL 2e auteur VAL Reinderhoff, Marie-Louise Reinderhoff 1903-1991 COL editie VAL 7e dr COL plaats VAL Amsterdam COL uitgever VAL Van Holkema & Warendorf
    reactie: Nee (ze delen dezelfde titel en taal, maar de record A is een herziene druk, dat blijkt uit "herz. druk")
    
    Voorbeeld 3:
    record A: COL ISBN VAL 3855397937 COL taal VAL ned COL jaar VAL 1987 COL titel VAL De @varkenshoeder COL ondertitel VAL een sprookje van Hans Christian Andersen COL hoofdauteur VAL Andersen, Hans Christian Andersen 1805-1875 COL 2e auteur VAL Ris, Ineke Ris COL 2e auteur VAL Duntze, Dorothée Duntze 1960- COL plaats VAL Den Haag COL uitgever VAL De Vier Windstreken COL vertaling van VAL Der Schweinehirt. - Mönchaltorf : Nord-Süd Verlag, cop. 1987 COL auteur/primair VAL Andersen 
    record B: COL ISBN VAL 3851951212 COL taal VAL dui COL jaar VAL 1982 COL titel VAL Der @Schweinehirt COL hoofdauteur VAL Andersen, Hans Christian Andersen 1805-1875 COL 2e auteur VAL Zwerger, Lisbeth Zwerger 1954- COL plaats VAL Salzburg COL uitgever VAL Verlag Neugebauer Press 
    reactie: Nee (dit is een vertaling, dus altijd een andere expressie)

    Voorbeeld 4:
    record A: COL taal VAL fra COL jaar VAL 1920 COL titel VAL @Sans famille COL hoofdauteur VAL Malot, Hector Henri Malot 1830-1907 COL plaats VAL Paris COL uitgever VAL Fayard
    record B: COL taal VAL fra COL jaar VAL 189X COL titel VAL @Sans Familie COL extra annotatie VAL ill. par É. Bayard COL hoofdauteur VAL Malot, Hector Henri Malot 1830-1907 COL 2e auteur VAL Bayard, Émile Antoine Bayard 1837-1891 COL plaats VAL Paris COL uitgever VAL Librairie Hachette

    reactie: Ja (dit is allebei dezelfde Franse originele versie, enige verschil is illustraties door Bayard)

    Voorbeeld 5:
    record A: COL taal VAL ned COL jaar VAL 1834 COL titel VAL De @Kersavond COL ondertitel VAL een verhaal voor kinderen COL extra annotatie VAL [naar het Hoogduitsch] COL extra annotatie VAL met platen [door H.P. Oosterhuis, gegraveerd door H.W. Hoogkamer] COL hoofdauteur VAL Schmid, Johann Christoph Friedrich von Schmid 1768-1854 COL 2e auteur VAL Oosterhuis, Haatje Pieters Oosterhuis 1784-1854 COL 2e auteur VAL Hoogkamer, H.W. Hoogkamer COL editie VAL 2e dr COL plaats VAL Amsterdam COL uitgever VAL Ten Brink & De Vries COL vertaling van VAL Der Weihnachtsabend
    record B: COL taal VAL ned COL jaar VAL 1837 COL titel VAL De @kersavond COL ondertitel VAL een verhaal voor kinderen COL extra annotatie VAL met platen COL hoofdauteur VAL Schmid, Johann Christoph Friedrich von Schmid 1768-1854 COL 2e auteur VAL Oosterhuis, Haatje Pieters Oosterhuis 1784-1854 COL 2e auteur VAL Hoogkamer, H.W. Hoogkamer COL editie VAL 3e dr COL plaats VAL Amsterdam COL uitgever VAL Ten Brink & De Vries COL omschrijving VAL Vert. uit het Duits COL omschrijving VAL Met gegraveerd titelblad COL omschrijving VAL Ill. gesigneerd H.P. Oosterhuis en H.W. Hoogkamer COL vertaling van VAL Der Weihnachtsabend. - 1825
    reactie: Ja (Zelfde vertaling, enige verschil zit in de platen/illustratie, zelfde expressie)

    Voorbeeld 6:
    record A: COL taal VAL ned COL titel VAL @Veel geluk! COL ondertitel VAL nieuwjaars- en verjaardagsversjes voor kinderen COL hoofdauteur VAL Louwerse, Pieter Louwerse 1840-1908 COL editie VAL Vijfde, verbeterde druk COL uitgever VAL S. & W.N. van Nooten
    record B: COL taal VAL ned COL titel VAL @Veel geluk COL ondertitel VAL nieuwjaars- en verjaardagsversjes voor kinderen COL hoofdauteur VAL Louwerse, Pieter Louwerse 1840-1908 COL editie VAL Zesde druk COL uitgever VAL S. & W.N. van Nooten
    reactie: Ja (De zesde druk komt na de vijfde verbeterde druk, en is dus dezelfde expressie ondanks de verschillende titel)

    nu jij:

"""
short_prompt = """
    Je bent een data assistent. 
    Bepaal of twee boekenrecords tot dezelfde WEMI expressie behoren.
    Belangrijke feiten:
    1. Let goed op het veld "editie, hier kan bijvoorbeeld staan "herz. dr". Dit is dan altijd een andere expressie.
    2. Verschillende edities kunnen tot dezelfde expressie behoren.
    3. Auteursnamen en titels kunnen vaak verschillen en net anders geschreven staan, ook staat de auteur soms in andere velden, bijvoorbeeld auteur en hoofdauteur
    4. Het gaat om significante tekstuele veranderingen, dus alleen ander voorwoord, andere uitgever of andere illustrator/coverart: ZELFDE expressie
    5. Andere redactie en editors: NIET zelfde expressie

"""
correct_match=0
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
        model="gpt-3.5-turbo",
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
        if label == '0':
            correct_match += 1
        else:
            print('missed a not one')
            print(s1)
            print(s2)
            print('')
        pred.append(pr)

    else:
        pr='1'
        if label == '1':
            correct_match += 1
        else:
            print('missed one:')
            print(s1)
            print(s2)
            print('')
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