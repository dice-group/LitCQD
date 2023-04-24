import csv
import os
import statistics

path = './generated/FB15K-237_dummy_kblrn/'

attr_values = dict()
with open(os.path.join(path, "attr_train2id.txt"), "r") as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)
    for row in reader:
        if int(row[1]) in attr_values:
            attr_values[int(row[1])].append(float(row[2]))
        else:
            attr_values[int(row[1])] = [float(row[2])]

with open(os.path.join(path, "attr_valid2id.txt"), "r") as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)
    for row in reader:
        if int(row[1]) in attr_values:
            attr_values[int(row[1])].append(float(row[2]))
        else:
            attr_values[int(row[1])] = [float(row[2])]

with open(os.path.join(path, "attr_test2id.txt"), "r") as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)
    for row in reader:
        if int(row[1]) in attr_values:
            attr_values[int(row[1])].append(float(row[2]))
        else:
            attr_values[int(row[1])] = [float(row[2])]


print('id\tmean\t\t\tmad')
means = list()
maedevs = list()
for attr, values in dict(sorted(attr_values.items())).items():
    means.append(statistics.mean(values))
    try:
        maedevs.append(sum([abs(means[-1]-v) for v in values])/len(values))
    except:
        maedevs.append(0.0)
    print(f"{attr}\t{means[-1]:.18f}\t{maedevs[-1]:.18f}")

print('='*20)
print('average mean per attribute:', statistics.mean(means))
print('average mad per attribute:', statistics.mean(maedevs))
all_values = list()
for attr, values in attr_values.items():
    for val in values:
        all_values.append(val)
print('mean:', statistics.mean(all_values))
print('mad:', statistics.stdev(all_values))
