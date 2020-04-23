from oned_array import Array
# an implementation using circular list
class Queue: 
    def __init__(self, maxsize):
        self._count = 0
        self._front = 0
        self._back = maxsize-1
        self._qArray = Array(maxsize)

    def isEmpty(self):
        return self._count == 0

    def isFull(self):
        return self._count == len(self._qArray)

    def __len__(self):
        return self._count

    def enqueue(self, item):
        assert not self.isFull(), "Cannot enqueue to a full queue."
        maxsize = len(self._qArray)
        self._back = (self._back + 1) % maxsize
        self._qArray[self._back] = item
        self._count += 1

    def dequeue(self,):
        assert not self.isEmpty(), "Cannot dequeue an empty queue."
        maxsize = len(self._qArray)
        item = self._qArray[self._front]
        self._front = (self._front + 1) % maxsize
        self._count += -1 
        return item

class Queue_lk():
    def __init__(self):
        self._head = None
        self._tail = None
        self._count = 0

    def __len__(self):
        return self._count
    
    def _isEmpty(self):
        return self._count == 0

    def enQueue(self, item):
        node = _qNode(item)
        if self._isEmpty():
            self._head = node
        else:
            self._tail.next = node

        self._count += 1
        self._tail = node

    def deQueue(self):
        assert not self._isEmpty(), "Can not deQueue an empty queue"
        self._count += -1
        item = self._head.item
        if self._head == self._tail:
            self._tail = None
        self._head = self._head.next
        return item

class _qNode():
    def __init__(self, item):
        self.item = item
        self.next = None
