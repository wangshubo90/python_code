class ExpressionTree:
    def __init__(self, expStr):
        self._expTree = None
        self._buildTree(expStr)

    def evaluate(self, varMap):
        return self._evalTree(self._expTree, varMap)

    def __str__(self):
        return self._buildString(self._expTree)

    def _buildString(self, treeNode):
        if treeNode.left is None and treeNode.right is None:
            return str(treeNode.element)
        else:
            expStr = '('
            expStr += self._buildString(treeNode.left)
            expStr += str(treeNode.element)
            expStr += self._buildString(treeNode.right)
            expStr += ')'
            return expStr

    def _evalTree(self, subtree, varDict):
        if subtree.left is None and subtree.right is None:
            if subtree.element >= '0' and subtree.element <= '9' :
                return int(subtree.element)
            else:
                assert subtree.element in varDict, "Invalid variable"
                return varDict[subtree.element]

        else:
            lvalue = _evalTree(subtree.left, varDict)
            rvalue = _evalTree(subtree.left, varDict)
            return computOp(lvalue, subtree.element, rvalue)

    def _computOp(left, op, right):

        


class _ExpTreeNode:
    def __init__(self, data):
        self.element = data
        self.left = None
        self.right = None