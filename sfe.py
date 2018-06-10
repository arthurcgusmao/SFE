import os
import time
import pandas as pd
import numpy as np
import itertools

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

    def __str__(self):
        return "Node({})".format(self.name)

    def add_edge(self, edge):
        self.neighbors.add(edge.end)
        if not edge.end in self.neighbor2edges:
            self.neighbor2edges[edge.end] = []
        self.neighbor2edges[edge.end].append(edge)


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

    def get_subgraph_nodes(self, node, max_depth, init=True):
        """Returns all nodes present in a subgraph defined by walking at most `max_depth` edges
        from an initial node.

        Arguments:
        - `node` (Node): the initial, central node for the subgraph
        - `max_depth` (int): max-depth for the breadth-first search done to construct the subgraph
        for each node (head and tail).
        """
        if init:
            self.expanded_nodes = set() # save nodes that have been expanded
        self.expanded_nodes.add(node)
        output = set([node])
        if max_depth > 1:
            for n in node.neighbors.difference(self.expanded_nodes): # loop through elements in neighbors that are not in expanded_nodes
                output.union(self.get_subgraph_nodes(n, max_depth-1, init=False))
            self.expanded_nodes.union(node.neighbors)
        return output.union(node.neighbors)

    def bfs_node_seqs(self, start_node, goal_nodes, max_depth):
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
        queue = [(start_node, [start_node])]
        depth = 0
        while queue:
            (vertex, path) = queue.pop(0)
            for node in vertex.neighbors - set(path):
                depth += 1
                if node in goal_nodes:
                    yield path + [node]
                elif depth < max_depth:
                    queue.append((node, path + [node]))

    def get_edge_seqs(self, node_seqs):
        """Returns all possible sequences of edges (paths) one can walk when following a set of node sequences.

        Arguments:
        - `node_seqs` (iterable): an iterable where each element should be a list of Nodes.

        Returns a list of edge sequences; each edge sequence is a list of edges that defines a path.
        """
        all_edges_seqs = {} # of the form: {end_node: [list of paths]}
        for node_seq in node_seqs:
            possible_edges_seqs = []
            for i in range(1, len(node_seq)):
                possible_edges_seqs.append(node_seq[i-1].neighbor2edges[node_seq[i]])
            end_node = node_seq[-1]
            if not end_node in all_edges_seqs:
                all_edges_seqs[end_node] = []
            all_edges_seqs[end_node].extend(list(itertools.product(*possible_edges_seqs)))
        return all_edges_seqs

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

    def search_paths(self, head_name, tail_name, max_depth):
        """Extract features using the current graph for an entity (head, tail) pair.

        Arguments:
        - `head_name` (string): head entity name.
        - `tail_name` (string): tail entity name.
        - `max_depth` (int): max-depth for the breadth-first search done to construct the subgraph
        for each node (head and tail).
        """
        head = self.graph.get_node(head_name)
        tail = self.graph.get_node(tail_name)

        nodes_subgraph_head = self.get_subgraph_nodes(head, max_depth)
        nodes_subgraph_tail = self.get_subgraph_nodes(tail, max_depth)

        nodes_intersect = nodes_subgraph_head.intersection(nodes_subgraph_tail)

        # now we know all nodes at the intersection of both subgraphs.
        # we must now find paths between head and intersects and tail and intersects
        head_node_seqs = self.bfs_node_seqs(head, nodes_intersect, max_depth)
        tail_node_seqs = self.bfs_node_seqs(tail, nodes_intersect, max_depth)

        # get intermediate paths
        head_inter_paths = sfe.get_edge_seqs(head_node_seqs)
        tail_inter_paths = sfe.get_edge_seqs(tail_node_seqs)

        all_edge_seqs = []
        for inter_node in head_inter_paths:
            head_edge_seqs = head_inter_paths[inter_node]
            tail_edge_seqs = tail_inter_paths.get(inter_node, None) # tail may not have an inter_node because the end_node may be the tail itself.
            if tail_edge_seqs == None:
                edge_seqs = head_edge_seqs
            else:
                edge_seqs = []
                for seq in itertools.product(head_edge_seqs, tail_edge_seqs):
                    edge_seqs.append(seq[0] + seq[1])
            all_edge_seqs.extend(edge_seqs)

        return all_edge_seqs

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
