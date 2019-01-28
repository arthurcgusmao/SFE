import os
import pandas as pd
import multiprocessing as mp

from sfe.sfe import Graph, SFE
from helpers import save_features_to_disk


def pipeline(
    # Files & Directories paths
    pra_graph_input_path, # list containing the path to where the tsv files containing set of triples to be used to build the graph are, e.g., `XKE/benchmarks/FB13/pra_graph_input/`. There should be `train.tsv` and `valid.tsv` files. Each line should contain the head, relation and tail of an existing triple, in the mentioned order.
    datasets_paths, # list containing the path for each dataset for which features will be extracted. The dataset file should be a TSV file containing the columns (in the order): head, relation, tail, label
    output_dir, # path for the output dir, the directory where results will be saved
    # @TODO: add timestamp to output_dir (and also save a text file in it containing the parameters (SFE options) used in the extraction)

    # SFE options
    max_depth=2,
    max_fan_out=100,
    bfs_memory_size=1000,
    batch_size=10000, # number of features that will be processed in a row before saving them to disk (and freeing up memory space). Notice that this applies to each Process, so in practice this number is multiplied by the number of cores.
):

    # BUILD GRAPH
    g = Graph()
    train = pd.read_csv(os.path.join(pra_graph_input_path, 'train.tsv'), sep='\t', skiprows=0, names=['head', 'relation', 'tail'])
    valid = pd.read_csv(os.path.join(pra_graph_input_path, 'valid.tsv'), sep='\t', skiprows=0, names=['head', 'relation', 'tail'])
    print("Building graph..."),
    g.partial_build_from_df(train)
    g.partial_build_from_df(valid)
    print("Built.")


    # EXTRACT FEATURES
    sfe = SFE(g, )
    for filepath in datasets_paths:
        print("\nStarting feature extraction for `{}` ...".format(filepath))
        df = pd.read_csv(filepath, sep='\t', skiprows=0, names=['head', 'relation', 'tail', 'label'])
        df = df.sort_values(by=['head', 'tail']) # this is important if multiprocessing to help SFE save computing time using the BFS memory
        output_file_name = os.path.basename(filepath).replace('.txt', '.tsv')

        # @TODO: put this into multiprocessing (or maybe threading)
        count = 0
        for res in sfe.extract_features(df, batch_size=batch_size):
            save_features_to_disk(res, output_dir, output_file_name)
            count += batch_size
            print("{} examples processed...".format(count))

    print("\nPipeline finished.")
