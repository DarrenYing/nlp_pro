import csv

def read_csv(path, encoding='utf-8-sig', headers=None, sep=',', dropna=True):
    with open(path, 'r', encoding=encoding) as csv_file:
        f = csv.reader(csv_file, delimiter=sep)
        start_idx = 0

        if headers is None:
            headers = next(f)
            # print(headers)
            start_idx += 1


        # ID,txt,Label
        sentences = []
        labels = []
        for line_idx, line in enumerate(f, start_idx):
            contents = line

            _dict = {}
            for header, content in zip(headers, contents):
                if str(header).lower() == "label":
                    labels.append(content)
                else:
                    _dict[header] = str(content).lower()    #小写
            sentences.append(_dict)

    return sentences, labels, headers