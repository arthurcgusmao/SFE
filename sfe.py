import os
import time
import math
import pandas as pd
import numpy as np
import itertools


class Edge(object):
    def __init__(self, start, end, label, direction):
        self.label = str(label)
        self.start = start
        self.end = end
        self.direction = direction
        self.str = self.__str__()

    def __str__(self):
        return self.label if self.direction == 1 else '_' + self.label


class Node(object):
    def __init__(self, name):
        self.name = name
        self.neighbors = set()
        # self.neighbor2edges = {} # a dict of the form {node: [edge1, edge2, ...]}
        self.neighbor2edgesstr = {} # a dict of the form {node: [edge1.str, edge2.str, ...]}
        # self.edgestr2neighbors = {} # a dict of the form {edge1.str: [node1, node2, ...]}
        self.fan_out = 0

    def __str__(self):
        return "Node({})".format(self.name)

    def add_edge(self, edge):
        self.neighbors.add(edge.end)
        # self.neighbor2edges[edge.end] = self.neighbor2edges.get(edge.end, []) + [edge]
        self.neighbor2edgesstr[edge.end] = self.neighbor2edgesstr.get(edge.end, []) + [edge.str]
        # self.edgestr2neighbors[edge.str] = self.edgestr2neighbors.get(edge.str, []) + [edge.end]
        self.fan_out += 1

    def get_edgestr2neighbors(self, edgestr):
        """This method works like a dict, mapping edge labels to nodes that are related to this
        node through that edge. Notice that reversed edges are considered different than their
        regular version, since we are dealing with its final string.

        The goal of creating a method for this is to save space. Thus, the actual dict is only
        created when necessary.
        """
        if not hasattr(self, 'edgestr2neighbors'):
            self.edgestr2neighbors = {} # a dict of the form {edge1.str: [node1, node2, ...]}
            for neighbor,edgesstrs in self.neighbor2edgesstr.iteritems():
                for edgestr in edgesstrs:
                    self.edgestr2neighbors[edgestr] = self.edgestr2neighbors.get(edgestr, []) + [neighbor]
        return self.edgestr2neighbors.get(edgestr, [])


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
    def __init__(self, graph, max_depth=2, max_fan_out=None):
        """Init method.

        Arguments:
        - `max_depth` (int): max-depth for the breadth-first search done to construct the subgraph
        for each node (head and tail).
        """
        self.graph = graph
        self.max_depth = max_depth
        self.max_fan_out = max_fan_out if max_fan_out != None else float('inf')

    def bfs_node_seqs(self, start_node):
        """Generates all possible sequences of nodes of max-depth `max_depth` one can walk
        from a start node.
        Nodes whose fan-out exceeds max_fan_out are not expanded.

        Arguments:
        - `start_node` (Node): the initial node from where to walk from
        - `goal_nodes` (list of Nodes): a list of end nodes, every path has to end in one of these nodes

        Yields node sequences; each node sequence (list) is a nodes list that defines a set of possible
        edge sequences (paths).
        """
        output = {} # indexed by end node; {end_node: [node sequence], ...}
        queue = [(start_node, [start_node], 0)]
        while queue:
            (vertex, path, level) = queue.pop(0)
            # only expand nodes whose fan_out does not exceed max
            if vertex.fan_out <= self.max_fan_out:
                for node in vertex.neighbors - set(path):
                    output[node] = output.get(node, []) + [path + [node]]
                    if level+1 < self.max_depth:
                        queue.append((node, path + [node], level+1))
        return output

    # def get_edge_seqs(self, node_seqs, invert=False):
    #     """Returns all possible sequences of edges (paths) one can walk when following a set of node sequences.
    #
    #     Arguments:
    #     - `node_seqs` (iterable): an iterable where each element should be a list of Nodes.
    #
    #     Returns a list of edge sequences; each edge sequence is a list of edges that defines a path.
    #     """
    #     edge_seqs = set()
    #     for node_seq in node_seqs:
    #         possible_edges_seqs = []
    #         for i in range(1, len(node_seq)):
    #             possible_edges_seqs.append(node_seq[i-1].neighbor2edges[node_seq[i]])
    #         edge_seqs.update(itertools.product(*possible_edges_seqs))
    #     return edge_seqs

    def get_paths(self, node_seqs):
        """Outputs all possible sequences of edges (paths) one can walk when following a set of node sequences.
        Returns a list of edge sequences; each edge sequence is a list of edge strings that defines a path.

        Arguments:
        - `node_seqs` (iterable): an iterable where each element should be a list of Nodes.
        """
        paths = set()
        for node_seq in node_seqs:
            possible_paths = []
            for i in range(1, len(node_seq)):
                possible_paths.append(node_seq[i-1].neighbor2edgesstr[node_seq[i]])
            paths.update(itertools.product(*possible_paths))
        return paths

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

    def search_paths(self, head_name, tail_name):
        """Search paths between two nodes using the current graph.

        Arguments:
        - `head_name` (string): head entity name.
        - `tail_name` (string): tail entity name.
        - `max_depth` (int): max-depth for the breadth-first search done to construct the subgraph
        for each node (head and tail).
        """
        last_time = time.time()
        head = self.graph.get_node(head_name)
        tail = self.graph.get_node(tail_name)
        print "time get nodes: {}".format(time.time() - last_time); last_time = time.time()
        head_node_seqs = self.bfs_node_seqs(head) # indexed by end node
        tail_node_seqs = self.bfs_node_seqs(tail) # indexed by end node
        print "time to find node sequences: {}".format(time.time() - last_time); last_time = time.time()
        node_seqs = self.merge_node_sequences(head, tail, head_node_seqs, tail_node_seqs)
        print "time to merge node sequences: {}".format(time.time() - last_time); last_time = time.time()
        paths = self.get_paths(node_seqs)
        print "time to get paths: {}".format(time.time() - last_time); last_time = time.time()

        return paths

    def generate_features(self, df, batch_size=999999):
        """Run SFE for a set of triples.

        Arguments:
        - `df` (pandas.DataFrame): DataFrame with three columns: head, tail and relation.
        Each row represents a triple in a knowledge graph.
        """
        # avoid unnecessary runs by running once for each node pair, independently of relation
        # run speed can be further improved by storing a bunch of BFS subgraphs and processing
        # close together.
        df = df.sort_values(by=['head', 'tail'])
        last = {'head': None, 'tail': None}; paths = None
        features = []
        for idx,row in df.iterrows():
            if last['head'] == row['head'] and last['tail'] == row['tail']:
                pass
            else:
                paths = self.search_paths(row['head'], row['tail'])
            features.append((idx, paths - {(row['relation'],)}))
            if len(features) == batch_size:
                yield features
                features = []
        yield features
