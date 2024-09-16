import json
from typing import DefaultDict

with open("../dataset/Tweets17_corenlp/twitter17_train.json", 'r') as f:
    all_data = []
    data = json.load(f)
    for d in data:
        head = list(d['head'])
        max = len(head)
        tmp = [[0] * max for _ in range(max)]
        for i in range(max):
            j = head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = DefaultDict(list)

        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        leverl_degree = [[6] * max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    # print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                node_set.add(g)
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                        leverl_degree[i][q] = 4
                                        node_set.add(q)
                                        for h in tmp_dict[q]:
                                            if h not in node_set:
                                                leverl_degree[i][h] = 5
                                                node_set.add(h)
        d['short'] = leverl_degree

    wf = open('../dataset/Tweets17_corenlp/twitter17_train_write_6.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Tweets17_corenlp/twitter17_test.json", 'r') as f:
    all_data = []
    data = json.load(f)
    for d in data:
        head = list(d['head'])
        max = len(head)
        tmp = [[0] * max for _ in range(max)]
        for i in range(max):
            j = head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = DefaultDict(list)

        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        leverl_degree = [[6] * max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    # print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                node_set.add(g)
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                        leverl_degree[i][q] = 4
                                        node_set.add(q)
                                        for h in tmp_dict[q]:
                                            if h not in node_set:
                                                leverl_degree[i][h] = 5
                                                node_set.add(h)
        d['short'] = leverl_degree

    wf = open('../dataset/Tweets17_corenlp/twitter17_test_write_6.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Tweets17_corenlp/twitter17_val.json", 'r') as f:
    all_data = []
    data = json.load(f)
    for d in data:
        head = list(d['head'])
        max = len(head)
        tmp = [[0] * max for _ in range(max)]
        for i in range(max):
            j = head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = DefaultDict(list)

        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        leverl_degree = [[6] * max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    # print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                node_set.add(g)
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                        leverl_degree[i][q] = 4
                                        node_set.add(q)
                                        for h in tmp_dict[q]:
                                            if h not in node_set:
                                                leverl_degree[i][h] = 5
                                                node_set.add(h)
        d['short'] = leverl_degree

    wf = open('../dataset/Tweets17_corenlp/twitter17_val_write_6.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()
