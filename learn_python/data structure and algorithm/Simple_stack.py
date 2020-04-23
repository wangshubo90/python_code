class Stack_ls():
    def __init__(self):
        self._theItems = list()

    def __len__(self):
        return len(self._theItems)

    def isEmpty(self):
        return self.__len__() == 0
        
    def peek(self):
        assert not self.isEmpty(), "Cannot peek at an empty stack"
        return self._theItems[-1]
    
    def pop(self):
        assert not self.isEmpty(), "Cannot pop from an empty stack"
        return self._theItems.pop()

    def push(self, item):
        self._theItems.append(item)

class Stack_linked():
    def __init__(self):
        self._top = None
        self._size = 0

    def __len__(self):
        return self._size

    def isEmpty(self):
        return self._size == 0

    def peek(self):
        assert not self.isEmpty(), "Cannot peek at an empty stack"
        return self._top.item

    def pop(self):
        assert not self.isEmpty(), "Cannot peek at an empty stack"
        node = self._top
        self._top = self._top.next
        self._size += -1
        return node.item

    def push(self, item):
        self._top = _StackNode(item, self._top)
        self._size += 1

class _StackNode():
    def __init__(self,item,link):
        self.item = item
        self.next = link

if __name__ == "__main__":
    a = range(10)
    stack = Stack_ls()
    for i in a:
        stack.push(i)
    stack.pop()
    stack.peek()
    