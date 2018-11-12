import os

def dfs_node_sequence_from_path(start, goal, edges_path, nodes_path=None):
    """Performs a DFS following a restricted edges path. One may want to use this function
    to search the possible nodes one visits when following a sequence of edges.

    Arguments:
    - `start`: start node
    - `goal`: goal node
    - `edges_path`: a list containing the sequence of edge names you want to consider.
    - `nodes_path` (optional): the current set of nodes the have been visited so far.
    """
    if nodes_path is None:
        nodes_path = [start]
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
        for next_ in set(neighbors) - set(nodes_path):
            for p in dfs_node_sequence_from_path(next_, goal, edges_path[1:], nodes_path + [next_]):
                yield p


####################################################################
##
## FEATURE SAVING FUNCTIONS
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



##################################################################
##
## PROCESS DATASET TO BUILD GRAPH INPUT
##
def build_graph_input_from_benchmark(benchmark_path):
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

    # @TODO: the dataframes to save are `valid_pos` and `train`. They must be saved into a folder called `pra_graph_input`, e.g., XKE/benchmarks/NELL186/pra_graph_input/train.tsv and valid.tsv
