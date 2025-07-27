import heapq

class HuffmanNode:
    def __init__(self, freq, idx=None, left=None, right=None):
        self.freq = freq
        self.idx = idx  # index of word if it's a leaf
        self.left = left
        self.right = right
        self.code = None  # will assign later
        self.path = None  # will assign later
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(token_freqs):
    # token_freqs: list of (token, freq), idx = position in vocab
    heap = [HuffmanNode(freq, idx=i) for i, (token, freq) in enumerate(token_freqs)]
    heapq.heapify(heap)
    nodes = list(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        parent = HuffmanNode(n1.freq + n2.freq, left=n1, right=n2)
        heapq.heappush(heap, parent)
        nodes.append(parent)
    root = heap[0]
    # Assign code/path for each leaf
    def assign_code(node, code, path):
        node.code = code
        node.path = path
        if node.left:
            assign_code(node.left, code + [0], path + [node])
        if node.right:
            assign_code(node.right, code + [1], path + [node])
    assign_code(root, [], [])
    # Create mapping from word idx to (code, path)
    idx2huffman = {}
    for node in nodes:
        if node.idx is not None:
            idx2huffman[int(node.idx)] = (node.code, node.path)
    # List of internal nodes and mapping to embedding indices
    internal_nodes = [n for n in nodes if n.idx is None]
    internal_node2idx = {n: i for i, n in enumerate(internal_nodes)}
    return idx2huffman, internal_nodes, internal_node2idx 