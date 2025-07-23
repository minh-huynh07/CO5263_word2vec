import heapq

class HuffmanNode:
    def __init__(self, freq, idx=None, left=None, right=None):
        self.freq = freq
        self.idx = idx  # index của từ nếu là lá
        self.left = left
        self.right = right
        self.code = None  # sẽ gán sau
        self.path = None  # sẽ gán sau
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(token_freqs):
    # token_freqs: list of (token, freq), idx = vị trí trong vocab
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
    # Gán code/path cho từng lá
    def assign_code(node, code, path):
        node.code = code
        node.path = path
        if node.left:
            assign_code(node.left, code + [0], path + [node])
        if node.right:
            assign_code(node.right, code + [1], path + [node])
    assign_code(root, [], [])
    # Tạo mapping từ idx từ sang (code, path)
    idx2huffman = {}
    for node in nodes:
        if node.idx is not None:
            idx2huffman[int(node.idx)] = (node.code, node.path)
    # Danh sách node nội bộ và mapping sang chỉ số embedding
    internal_nodes = [n for n in nodes if n.idx is None]
    internal_node2idx = {n: i for i, n in enumerate(internal_nodes)}
    return idx2huffman, internal_nodes, internal_node2idx 