import csv
import sys

attribute = sys.argv[1]
value = float(sys.argv[2])

with open("./generated/FB15K-237_dummy_kblrn/attr2id_min_max.txt", "r") as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)
    for row in reader:
        if row[1] == attribute:
            print(f"{row[0]}: {value*(float(row[3])-float(row[2]))+float(row[2]):.2f}")
            exit()
