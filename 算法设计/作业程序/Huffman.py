import heapq

class Node:
    def __init__(self, char, frequency):
        self.char = char
        self.frequency = frequency
        self.left = None
        self.right = None

    # 定义比较操作，用于最小堆的排序
    def __lt__(self, other):
        return self.frequency < other.frequency

def dfsHuffmanTree(current_string, current_node, Huffman):
    if current_node.left is None and current_node.right is None:  # 到达叶子节点
        Huffman[current_node.char] = current_string  # 存储字符的哈夫曼编码
        return
    if current_node.left is not None:
        dfsHuffmanTree(current_string + "0", current_node.left, Huffman)
    if current_node.right is not None:
        dfsHuffmanTree(current_string + "1", current_node.right, Huffman)

def buildHuffmanTree(frequency):
    char_set = list(frequency.keys())
    Smallest_Value_Heap = []
    # 初始化堆
    for char in char_set:
        node = Node(char, frequency[char])
        heapq.heappush(Smallest_Value_Heap, node)

    # 构建哈夫曼树
    while len(Smallest_Value_Heap) > 1:
        smallest_value_node_0 = heapq.heappop(Smallest_Value_Heap)
        smallest_value_node_1 = heapq.heappop(Smallest_Value_Heap)
        newnode = Node(None, smallest_value_node_0.frequency + smallest_value_node_1.frequency)
        newnode.left = smallest_value_node_0
        newnode.right = smallest_value_node_1
        heapq.heappush(Smallest_Value_Heap, newnode)

    # 获取哈夫曼树的根节点
    HuffmanTreeHead = heapq.heappop(Smallest_Value_Heap)
    Huffman = {}
    dfsHuffmanTree("", HuffmanTreeHead, Huffman)
    return Huffman

# 示例
frequency = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
# a~h的频率遵循斐波那契数列
# frequency = {'a':1, 'b':1, 'c':2, 'd':3, 'e':5, 'f':8, 'g':13, 'h':21}
Huffman = buildHuffmanTree(frequency)
print(Huffman)