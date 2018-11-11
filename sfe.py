import os
import time
import math
import pandas as pd
import numpy as np
import itertools


def debug_get_name_of_els_in_list(list_of_els):
    """Prints a list of elements using their string method."""
    l = []
    for n in list_of_els:
        l.append(n.__str__())
    return l


class Edge(object):
    def __init__(self, start, end, type):
        self.type = str(type)
        self.start = start
        self.end = end

class Node(object):
    def __init__(self, name):
        self.name = name
        self.fan_out = 0
        self.edge_fan_out = {} # indexed by edge_type; {edge_type: [fan out count]}
        self.in_edge2neighbors = {} # indexed by edge_type; {edge_type: [list of neighbors]}
        self.out_edge2neighbors = {} # indexed by edge_type; {edge_type: [list of neighbors]}

    def __str__(self):
        return "Node({})".format(self.name)

    def add_edge(self, edge, direction):
        if direction == 'in':
            self.in_edge2neighbors[edge.type] = self.in_edge2neighbors.get(edge.type, set()).union([edge.start])
        else:
            self.out_edge2neighbors[edge.type] = self.out_edge2neighbors.get(edge.type, set()).union([edge.end])
        self.fan_out += 1
        self.edge_fan_out[edge.type] = self.edge_fan_out.get(edge.type, 0) + 1 # we count the fan out per edge type, regardless whether it's an incoming or outgoing edge


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
            edge = Edge(head, tail, relation)
            head.add_edge(edge, 'out')
            tail.add_edge(edge, 'in')


class SFE(object):
    def __init__(self, graph, max_depth=2, max_fan_out=100):
        """Init method.

        Arguments:
        - `max_depth` (int): max-depth for the breadth-first search done to construct the subgraph
        for each node (head and tail).
        """
        self.graph = graph
        self.max_depth = max_depth
        self.max_fan_out = max_fan_out if max_fan_out != None else float('inf')

    def bfs_edge_seqs(self, start_node, is_tail=False):
        """Generates all possible sequence of edges of max-depth `max_depth` one can walk
        from a start node.
        Edges whose fan-out exceeds max_fan_out are not expanded.

        Arguments:
        - `start_node` (Node): the initial node from where to walk from
        - `is_tail` (Bool): indicates how the relation strings should be directed
        """
        output = {} # indexed by end node; {end_node: {(node sequence): [(edge sequence 1), ...]}}
        queue = [(start_node, (start_node,), (), 0)]
        while queue:
            (vertex, node_seq, edge_seq, level) = queue.pop(0)
            for edge_type in vertex.edge_fan_out:
                if vertex.edge_fan_out[edge_type] <= self.max_fan_out: # only expand nodes whose fan_out does not exceed max
                    # loop for outgoing edges
                    for node in vertex.out_edge2neighbors.get(edge_type, set()) - set(node_seq):
                        new_node_seq = node_seq + (node,)
                        new_edge_seq = edge_seq + (edge_type,) if not is_tail else edge_seq + ('_' + edge_type,) # edges preceded by '_' are incoming edges
                        node_seq2edge_seqs = output.get(node, {})
                        node_seq2edge_seqs[new_node_seq] = node_seq2edge_seqs.get(new_node_seq, set()).union([new_edge_seq])
                        output[node] = node_seq2edge_seqs
                        if level+1 < self.max_depth:
                            queue.append((node, new_node_seq, new_edge_seq, level+1))
                    # loop for incoming edges
                    for node in vertex.in_edge2neighbors.get(edge_type, set()) - set(node_seq):
                        new_node_seq = node_seq + (node,)
                        new_edge_seq = edge_seq + ('_' + edge_type,) if not is_tail else edge_seq + (edge_type,) # edges preceded by '_' are incoming edges
                        node_seq2edge_seqs = output.get(node, {})
                        node_seq2edge_seqs[new_node_seq] = node_seq2edge_seqs.get(new_node_seq, set()).union([new_edge_seq])
                        output[node] = node_seq2edge_seqs
                        if level+1 < self.max_depth:
                            queue.append((node, new_node_seq, new_edge_seq, level+1))
        return output

    def merge_edge_sequences(self, head, tail, head_bfs_res, tail_bfs_res):
        output = set() # set of edge sequences between the two nodes
        for end_node in head_bfs_res:
            # print '\n----- for end node = {} -----'.format(end_node)
            head__node_seq2edge_seqs = head_bfs_res.get(end_node, {})
            if end_node == tail:
                for edge_seq in head__node_seq2edge_seqs.values():
                    # print 'end_node = tail; edge_seq = ', edge_seq
                    output = output.union(edge_seq)
            else:
                tail__node_seq2edge_seqs = tail_bfs_res.get(end_node, {})
                # print("!!!### {} ||| {} ###!!!".format(head__node_seq2edge_seqs, type(head__node_seq2edge_seqs)))
                for head_node_seq in head__node_seq2edge_seqs:
                    for tail_node_seq in tail__node_seq2edge_seqs:
                        # print '`head_node_seq`:', head_node_seq, type(head_node_seq)
                        # print '`tail_node_seq`:', tail_node_seq, type(tail_node_seq)
                        if len(set(tail_node_seq).intersection(set(head_node_seq))) == 1: # check for acyclicity
                            # print '`head_node_seq`:', head_node_seq, type(head_node_seq)
                            # print '`tail_node_seq`:', tail_node_seq, type(tail_node_seq)
                            for head_edge_seq in head__node_seq2edge_seqs[head_node_seq]:
                                for tail_edge_seq in tail__node_seq2edge_seqs[tail_node_seq]:
                                    # print '`head_node_seq`: {} \t `head_edge_seq`: {}'.format(debug_get_name_of_els_in_list(head_node_seq), head_edge_seq)
                                    # print '`tail_node_seq`: {} \t `tail_edge_seq`: {}'.format(debug_get_name_of_els_in_list(tail_node_seq), tail_edge_seq)
                                    output.add(head_edge_seq + tuple(reversed(tail_edge_seq)))
        return output

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
        # print "time get nodes: {}".format(time.time() - last_time); last_time = time.time()
        head_bfs_res = self.bfs_edge_seqs(head) # indexed by end node
        tail_bfs_res = self.bfs_edge_seqs(tail, is_tail=True) # indexed by end node
        # print "time to perform BFS on both nodes: {}".format(time.time() - last_time); last_time = time.time()
        edge_seqs = self.merge_edge_sequences(head, tail, head_bfs_res, tail_bfs_res)
        # print "time to merge edge sequences: {}".format(time.time() - last_time); last_time = time.time()
        return edge_seqs

    def extract_features(self, df, batch_size=999999):
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
