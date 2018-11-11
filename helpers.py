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
