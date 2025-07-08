with open("test_pairs_expr.txt", "r", encoding="utf-8") as infile, open("expanded.txt", "w", encoding="utf-8") as outfile:
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            print('huh')
            continue  # skip malformed lines

        item1, item2, label = parts

        outfile.write(item1.strip() + '\n')
        outfile.write(item2.strip() + '\n')
        outfile.write(label.strip() + '\n')