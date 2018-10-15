def dfs_node_sequence_from_path(start, goal, edges_path, nodes_path=None):
    """Performs a DFS following a restricted edges path. One may want to use this function
    in order to search the possible nodes one visits when following a sequence of edges.

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
        for next_ in set(start.get_edgestr2neighbors(edges_path[0])) - set(nodes_path):
            for p in dfs_node_sequence_from_path(next_, goal, edges_path[1:], nodes_path + [next_]):
                yield p
