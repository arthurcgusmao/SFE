import os
import numpy as np

def debug_get_name_of_els_in_list(list_of_els):
    """Prints a list of elements using their string method."""
    l = []
    for n in list_of_els:
        l.append(n.__str__())
    return l

def dfs_node_sequence_from_path(start, goal, edges_path, nodes_path=None, allow_cycles=False):
    """Performs a DFS following a restricted edges path. One may want to use this function
    to search the possible nodes one visits when following a sequence of edges.

    Arguments:
    - `start`: start node
    - `goal`: goal node
    - `edges_path` (list or string): a list containing the sequence of edge names you want to consider; it can also be a PRA-style string describing the sequence of edges.
    - `nodes_path` (optional): the current set of nodes the have been visited so far.
    - `allow_cycles` (bool): allow the search to expand to already visited nodes.
    """
    if nodes_path is None:
        nodes_path = [start]
    if type(edges_path) == str:
        edges_path = edges_path.split('-')[1:-1]
    if len(edges_path) == 0:
        if start == goal: yield nodes_path
    else:
        edge_string = edges_path[0]
        if edge_string[0] == '_':
            edge_type = edge_string[1:]
            neighbors = start.in_edge2neighbors.get(edge_type, set())
            inversed = True
        else:
            edge_type = edge_string
            neighbors = start.out_edge2neighbors.get(edge_type, set())
            inversed = False
        if allow_cycles:
            next_nodes = set(neighbors)
        else:
            next_nodes = set(neighbors) - set(nodes_path)
        for next_ in next_nodes:
            for p in dfs_node_sequence_from_path(next_, goal, edges_path[1:], nodes_path + [next_], allow_cycles=allow_cycles):
                yield p


####################################################################
##
## FEATURE SAVING/READING FUNCTIONS
##
def stringify_res(res):
    """This method will transform the results (`res`) of the `extract_features()` function to
    strings, so that they can be saved to disk in the same format that Gardner's PRA code uses.
    """
    for rel in res: # for each relation
        for i in range(len(res[rel])): # for each instance
            inst = res[rel].pop(0)
            # transform features into string
            stringified_paths = ['-' + '-'.join(path) + '-,1.0' for path in inst['features']]
            stringified_feats = ' -#- '.join(stringified_paths)
            # transform entity pair into string
            stringified_ent_pair = ','.join(inst['entity_pair'])

            inst = '{}\t{}\t{}\n'.format(stringified_ent_pair, inst['label'], stringified_feats)
            res[rel].append(inst)
        res[rel] = ''.join(res[rel])

def save_features_to_disk(res, output_dir, output_file_name):
    """
    Arguments:
    - `res`: output from `extract_features()` method, a dict of the form described above.
    - `output_dir`: name of the main directory where each relation directory will be placed
    - `output_file_name`: name that each output file will have (each is placed inside its relation directory)
    """
    stringify_res(res)
    # print res; return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for rel in res:
        rel_output_dir = os.path.join(output_dir, rel)
        if not os.path.exists(rel_output_dir):
            os.mkdir(rel_output_dir)
        rel_output_file_path = os.path.join(rel_output_dir, output_file_name)
        with open(rel_output_file_path, 'a') as f:
            f.write(res[rel])

def parse_feature_matrix(filepath):
    """Returns four objects: three lists (of heads, tails and labels) and a sparse matrix (of
    features) for the input (a path to a feature matrix file).
    """
    heads = []
    tails = []
    labels = []
    feat_dicts = []
    with open(filepath, 'r') as f:
        for line in f:
            ent_pair, label, features = line.replace('\n', '').split('\t')
            head, tail = ent_pair.split(',')
            d = {}
            if features:
                for feat in features.split(' -#- '):
                    feat_name, value = feat.split(',')
                    d[feat_name] = float(value)

            heads.append(head)
            tails.append(tail)
            labels.append(int(label))
            feat_dicts.append(d)

    return np.array(heads), np.array(tails), np.array(labels), feat_dicts


##################################################################
##
## PROCESS DATASET TO BUILD GRAPH INPUT
##
def build_graph_input_from_benchmark(benchmark_path):
    graph_input_path = os.path.join(benchmark_path, 'pra_graph_input')
    if os.path.exists(graph_input_path):
        print 'Graph input already exists for current benchmark, not (re)building it.\n'
        return
    
    import pandas as pd
    train2id = pd.read_csv(benchmark_path + '/train2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    valid2id = pd.read_csv(benchmark_path + '/valid2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])
    test2id = pd.read_csv(benchmark_path + '/test2id.txt', sep=' ', skiprows=1, names=['head', 'tail', 'relation'])

    from tools import dataset_tools
    entity2id, id2entity     = dataset_tools.read_name2id_file(benchmark_path + '/entity2id.txt')
    relation2id, id2relation = dataset_tools.read_name2id_file(benchmark_path + '/relation2id.txt')

    train = pd.read_csv(benchmark_path + '/train.txt', sep='\t', skiprows=0, names=['head', 'relation', 'tail'])
    valid = pd.read_csv(benchmark_path + '/valid.txt', sep='\t', skiprows=0, names=['head', 'relation', 'tail', 'label'])
    test = pd.read_csv(benchmark_path + '/test.txt', sep='\t', skiprows=0, names=['head', 'relation', 'tail', 'label'])

    valid_pos = valid.loc[valid['label'] == 1]

    os.mkdir(graph_input_path)
    train.to_csv(os.path.join(graph_input_path, 'train.tsv'), index=False, header=False, sep='\t', columns=['head', 'relation', 'tail'])
    valid_pos.to_csv(os.path.join(graph_input_path, 'valid.tsv'), index=False, header=False, sep='\t', columns=['head', 'relation', 'tail'])


##################################################################
##
## COMPARE TWO EXTRACTED FEATURES MATRICES
##
def compare_feature_matrices(filepath1, filepath2):
    heads, tails, labels, feat_dicts = parse_feature_matrix(filepath1)
    zipped1 = zip(list(heads),list(tails),list(labels),feat_dicts)
    heads, tails, labels, feat_dicts = parse_feature_matrix(filepath2)
    zipped2 = zip(list(heads),list(tails),list(labels),feat_dicts)

    zipped1.sort()
    zipped2.sort()

    if zipped1 == zipped2:
        print "Feature matrices are equivalent."
        return

    # else, return rows that are different
    z1_diff_z2 = []
    for inst in zipped1:
        if not inst in zipped2:
            z1_diff_z2.append(inst)
    z2_diff_z1 = []
    for inst in zipped2:
        if not inst in zipped1:
            z2_diff_z1.append(inst)

    z1_diff_z2.sort()
    z2_diff_z1.sort()

    print "Differences found.\n{} examples in FM1 diverge from FM2;\n{} examples in FM2 diverge from FM1.\nDifferent entity pairs will be described next. FM1 and FM2 refers to the feature matrices in filepath1 and filepath2, respectively.".format(len(z1_diff_z2), len(z2_diff_z1))

    # unzip elements to separate entity pairs,
    heads, tails, labels1, feat_dicts1 = zip(*z1_diff_z2)
    ent_pairs1 = zip(heads, tails)
    heads, tails, labels2, feat_dicts2 = zip(*z2_diff_z1)
    ent_pairs2 = zip(heads, tails)

    compare_instances(ent_pairs1, labels1, feat_dicts1, ent_pairs2, labels2, feat_dicts2, order=['FM1', 'FM2'])
    compare_instances(ent_pairs2, labels2, feat_dicts2, ent_pairs1, labels1, feat_dicts1, order=['FM2', 'FM1'])

def compare_instances(ent_pairs1, labels1, feat_dicts1, ent_pairs2, labels2, feat_dicts2, order=['FM1', 'FM2']):
    for ent_pair in ent_pairs1:
        print("\n\n\n-----------------------------------------------------------------------")
        print("Entity pair:\n\t{}".format(ent_pair))
        if ent_pair in ent_pairs2:
            idx1 = ent_pairs1.index(ent_pair)
            idx2 = ent_pairs2.index(ent_pair)
            label1 = labels1[idx1]
            label2 = labels2[idx2]
            feats1 = feat_dicts1[idx1]
            feats2 = feat_dicts2[idx2]
            if label1 != label2: print("\nLabels are different: {} ({}) and {} ({})".format(label1, order[0], label2, order[1]))
            dict_compare(feats1, feats2, order)
        else:
            print('Entity pair in {} not present in {}.'.format(order[0], order[1]))

def dict_compare(d1, d2, order):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    print("\n{} diff {}:\n{}\n\n{} diff {}:\n{}\n\nModified:\n{}\n\nSame:\n{}".format(order[0], order[1], added, order[1], order[0], removed, modified, same))
