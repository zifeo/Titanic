from scipy.sparse.linalg import eigsh
from scipy.sparse import linalg
from scipy import sparse, stats
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
import numpy as np

def calc_weight(n, root, pred, local_tree):
    
    if n==root:
        return 0, root
    
    parent = pred[root, n]

    w_p = local_tree[parent, n]
    
    gparent = pred[root, parent]
    
    if gparent != -9999:
        w_d = local_tree[gparent, parent]
    else:
        w_d = 1
        gparent = n
        
    w = 2./(1./w_p + 1./w_d)

    return w, gparent

def mst(graph, levels=2):
    
    G = [graph]
    
    for _ in range(levels):
        test_dist_triu = np.triu(graph)
        Tree = minimum_spanning_tree(test_dist_triu)
        Tree = Tree + Tree.T

        local_tree = Tree.todense()
        distance_matrix, pred = dijkstra(local_tree, directed=False, unweighted=True, return_predecessors=True)

        root = np.random.choice(np.arange(distance_matrix.shape[0]))
        even_nodes = distance_matrix[:, root] % 2 == 0
        even_nodes = np.arange(graph.shape[0])[even_nodes]

        weight_tree = np.zeros(local_tree.shape)
        

        for n in even_nodes:
            print("node : ",n)
            out = calc_weight(n, root, pred, local_tree)
            print("out : ", out)
            weight_tree[n, out[1]] = out[0]
            weight_tree[out[1], n] = out[0]

        graph = weight_tree
        G.append(graph)
        
    return G