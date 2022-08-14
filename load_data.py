import os

def load_data(filename):
    file = open(filename, 'rt', encoding='UTF8')

    features = []; X = []; y = []
    for line in file:
        line = line.replace('\n', '')
        row = [s.strip() for s in line.split(',')]
        if not features:
            features = row
        else:
            X.append([float(s) for s in row[:-1]])
            y.append(row[-1])
    file.close()
    return np.array(X), np.transpose(np.array([y])), features