import csv, glob, os, sys
folder = sys.argv[1]
parts = sorted(glob.glob(os.path.join(folder, 'train_labels_part*.csv')))
assert parts, 'brak plikow train_labels_part*.csv w ' + folder
rows, header, ids = [], None, set()
for p in parts:
    with open(p, newline='') as fh:
        r = csv.reader(fh); h = next(r); header = header or h
        for row in r:
            if row[0] in ids:
                continue
            if not os.path.exists(os.path.join(folder, 'train', 'images', row[0] + '.png')):
                continue
            ids.add(row[0]); rows.append(row)
    print(f'{os.path.basename(p)}: razem {len(rows)}')
with open(os.path.join(folder, 'train_labels.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(header); w.writerows(rows)
print('zapisano', len(rows), 'obrazkow')
