import os
import time
import pandas as pd
import numpy as np
import itertools

## for debugging purposes
def debug_print(list_of_els, something=''):
    l = []
    for n in list_of_els:
        l.append(n.__str__())
    print("{} --- {}".format(something, l))

class Edge(object):
    def __init__(self, start, end, label, direction):
        self.label = str(label)
        self.start = start
        self.end = end
        self.direction = direction

    def __str__(self):
        return self.label if self.direction == 1 else '_' + self.label


class Node(object):
    def __init__(self, name):
        self.name = name
        self.neighbors = set()
        self.neighbor2edges = {} # a dict of the form {node: [edge1, edge2, ...]}
        self.edge_fan_out = {} # store the fan out for each edge label, a dict of the form {edge_label: fan_out (int)}

    def __str__(self):
        return "Node({})".format(self.name)

    def add_edge(self, edge):
        self.neighbors.add(edge.end)
        # if not edge.end in self.neighbor2edges:
        #     self.neighbor2edges[edge.end] = []
        # self.neighbor2edges[edge.end].append(edge)
        self.neighbor2edges[edge.end] = self.neighbor2edges.get(edge.end, []) + [edge]
        self.edge_fan_out[edge.label] = self.edge_fan_out.get(edge.label, 0) + 1



class Graph(object):
    def __init__(self):
        self.nodes = {} # nodes are indexed by name

    def get_node(self, name, create=False):
        if create and not name in self.nodes:
            self.nodes[name] = Node(name)
        return self.nodes[name]

    def partial_build_from_df(self, df):
        """Arguments:
        - `df` (pandas.DataFrame): DataFrame with three columns: head, tail and relation.
        Each row represents a triple in a knowledge graph.
        """
        for idx,row in df.iterrows():
            head = self.get_node(row['head'], create=True)
            tail = self.get_node(row['tail'], create=True)
            relation = row['relation']
            head.add_edge(Edge(head, tail, relation, +1))
            tail.add_edge(Edge(tail, head, relation, -1))


class SFE(object):
    def __init__(self, graph):
        self.graph = graph

    def bfs_node_seqs(self, start_node, max_depth):
        """Generates all possible sequences of nodes of max-depth `max_depth` one can walk
        to get to a set of goal nodes.

        Arguments:
        - `start_node` (Node): the initial node from where to walk from
        - `goal_nodes` (list of Nodes): a list of end nodes, every path has to end in one of these nodes
        - `max_depth` (int): max-depth for the breadth-first search done to construct the subgraph
        for each node (head and tail).

        Yields node sequences; each node sequence (list) is a nodes list that defines a set of possible
        edge sequences (paths).
        """
        output = {} # indexed by end node; {end_node: [node sequence], ...}
        queue = [(start_node, [start_node], 0)]
        while queue:
            (vertex, path, level) = queue.pop(0)
            for node in vertex.neighbors - set(path):
                output[node] = output.get(node, []) + [path + [node]]
                if level+1 < max_depth:
                    queue.append((node, path + [node], level+1))
        return output


    def get_edge_seqs(self, node_seqs, invert=False):
        """Returns all possible sequences of edges (paths) one can walk when following a set of node sequences.

        Arguments:
        - `node_seqs` (iterable): an iterable where each element should be a list of Nodes.

        Returns a list of edge sequences; each edge sequence is a list of edges that defines a path.
        """
        edge_seqs = []
        for node_seq in node_seqs:
            # for node in node_seq:
            #     debug_print(node)
            possible_edges_seqs = []
            for i in range(1, len(node_seq)):
                possible_edges_seqs.append(node_seq[i-1].neighbor2edges[node_seq[i]])
            edge_seqs.extend(list(itertools.product(*possible_edges_seqs)))
        return edge_seqs

        # all_edges_seqs = {} # of the form: {end_node: [list of paths]}
        # for node_seq in node_seqs:
        #     possible_edges_seqs = []
        #     for i in range(1, len(node_seq)):
        #         if invert:
        #             possible_edges_seqs.append(node_seq[i].neighbor2edges[node_seq[i-1]])
        #         else:
        #             possible_edges_seqs.append(node_seq[i-1].neighbor2edges[node_seq[i]])
        #     end_node = node_seq[-1]
        #     if not end_node in all_edges_seqs:
        #         all_edges_seqs[end_node] = []
        #     all_edges_seqs[end_node].extend(list(itertools.product(*possible_edges_seqs)))
        # return all_edges_seqs

    def get_features(self, paths, relation):
        """Returns a list of strings representing the feature names from a set of paths.
        The feature that has the current relation as the only path is removed.
        """
        features = []
        for path in paths:
            edges = []
            for edge in path:
                edges.append(edge.__str__())
            features.append(edges)
        if [str(relation)] in features: features.remove([str(relation)])
        return features

    def merge_node_sequences(self, head, tail, head_node_seqs, tail_node_seqs):
        node_seqs = set()
        for end_node in head_node_seqs:
            for head_node_seq in head_node_seqs.get(end_node, []):
                if end_node == tail:
                    node_seqs.add(tuple(head_node_seq))
                else:
                    for tail_node_seq in tail_node_seqs.get(end_node, []):
                        if len(set(tail_node_seq).intersection(set(head_node_seq))) == 1: # check for acyclicity
                            node_seqs.add(tuple(head_node_seq[:-1] + list(reversed(tail_node_seq))))
        return node_seqs

    def search_paths(self, head_name, tail_name, max_depth):
        """Search paths between two nodes using the current graph.

        Arguments:
        - `head_name` (string): head entity name.
        - `tail_name` (string): tail entity name.
        - `max_depth` (int): max-depth for the breadth-first search done to construct the subgraph
        for each node (head and tail).
        """
        head = self.graph.get_node(head_name)
        tail = self.graph.get_node(tail_name)

        head_node_seqs = self.bfs_node_seqs(head, max_depth) # indexed by end node
        tail_node_seqs = self.bfs_node_seqs(tail, max_depth) # indexed by end node

        node_seqs = self.merge_node_sequences(head, tail, head_node_seqs, tail_node_seqs)
        edge_seqs = self.get_edge_seqs(node_seqs)

        return edge_seqs

    def generate_features(self, df, max_depth, batch_size=999999):
        """Run SFE for a set of triples.

        Arguments:
        - `df` (pandas.DataFrame): DataFrame with three columns: head, tail and relation.
        Each row represents a triple in a knowledge graph.
        """
        # avoid unnecessary runs by running once for each node pair, indepedently of relation
        df = df.sort_values(by=['head', 'tail'])
        last = {'head': None, 'tail': None}; paths = None
        features = []
        for idx,row in df.iterrows():
            if last['head'] == row['head'] and last['tail'] == row['tail']:
                pass
            else:
                paths = self.search_paths(row['head'], row['tail'], max_depth)
            features.append((idx, self.get_features(paths, row['relation'])))
            if len(features) == batch_size:
                yield features
                features = []
        yield features
