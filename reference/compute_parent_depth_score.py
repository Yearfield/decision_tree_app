def compute_parent_depth_score(nodes, parent_id=None, depth=0, scores=None):
    """
    Recursively compute the depth score of each node in a tree.

    Args:
        nodes (dict): Mapping of node_id -> node dict, where each node may have "parent" and "children".
        parent_id (str): The current parent node id being processed.
        depth (int): Current depth in the tree.
        scores (dict): Accumulator for depth scores.

    Returns:
        dict: Mapping of node_id -> depth score
    """
    if scores is None:
        scores = {}

    for node_id, node in nodes.items():
        if node.get("parent") == parent_id:
            scores[node_id] = depth
            # Recursively compute for children
            compute_parent_depth_score(nodes, parent_id=node_id, depth=depth + 1, scores=scores)

    return scores
