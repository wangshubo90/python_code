class _BinTreeNode :
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def preOrderTrav(subtree):
    if subtree is not None:
        print(subtree.data)
        preOrderTrav(subtree.left)
        preOrderTrav(subtree.right)

def inOrderTrav(subtree):
    if subtree is not None:
        inOrderTrav(subtree.left)
        print(subtree.data)
        inOrderTrav(subtree.right)

def postOrderTrav(subtree):
    if subtree is not None:
        postOrderTrav(subtree.left)
        postOrderTrav(subtree.right)
        print(subtree.data)

from Queue import Queue_lk as Queue

def breadthFirstTrav(subtree):
    q = Queue()
    q.enQueue(subtree)

    while not q._isEmpty:
        node = q.deQueue()
        print(node.data)

        if node.left is not None:
            q.enQueue(node.left)
        if node.right is not None:
            q.enQueue(node.right)

        