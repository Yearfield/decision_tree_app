def detect_orphan_nodes(nodes):
    """
    Detect orphan nodes in the decision tree (nodes whose parent does not exist).

    Args:
        nodes (dict): Mapping of node_id -> node dict, where each node may have a "parent".

    Returns:
        list: List of orphan node_ids
    """
    node_ids = set(nodes.keys())
    orphan_nodes = []

    for node_id, node in nodes.items():
        parent_id = node.get("parent")
        if parent_id and parent_id not in node_ids:
            orphan_nodes.append(node_id)

    return orphan_nodes
