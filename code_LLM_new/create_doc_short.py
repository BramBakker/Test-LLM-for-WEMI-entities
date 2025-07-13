import re

# Fields you want to remove
fields_to_remove = {"ISBN", "omschrijving","annotatie","lokale annotatie","plaats","jaar"}  # Add more as needed
fields_to_remove_work = {"taal", "ISBN", "omschrijving","annotatie","lokale annotatie","uitgever","plaats","oorspronkelijk jaar","jaar","2e auteur vermelding","hoofdauteur vermelding"}  # Add more as needed


def clean_segment(segment):
    """Remove unwanted fields from a COL...VAL segment."""
    matches = re.findall(r'COL (.*?) VAL (.*?)(?=(?:COL |$))', segment)
    cleaned = [f"COL {k} VAL {v.strip()}" for k, v in matches if k.strip() not in fields_to_remove_work]
    return ' '.join(cleaned)

with open('test_pairs_expr.txt', 'r', encoding='utf-8') as infile, open('short_pairs_expr.txt', 'w', encoding='utf-8') as outfile:
    for line in infile:
        parts = line.strip().split('\t')

        if len(parts) != 3:
            # If the line is not a triplet, write as-is
            outfile.write(line)
            continue

        item1, item2, match = parts
        item1_cleaned = clean_segment(item1)
        item2_cleaned = clean_segment(item2)

        outfile.write(f"{item1_cleaned}\t{item2_cleaned}\t{match}\n")
