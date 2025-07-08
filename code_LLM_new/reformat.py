with open("expanded.txt", "r", encoding="utf-8") as infile, open("output.txt", "w", encoding="utf-8") as outfile:
    lines = [line.strip() for line in infile if line.strip()]  # remove empty lines
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):  # ensure we have a full triplet
            a = lines[i]
            b = lines[i + 1]
            label = lines[i + 2]
            outfile.write(f"{a}\t{b}\t{label}\n")
