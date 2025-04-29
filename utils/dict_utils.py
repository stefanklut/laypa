from collections import deque


# Based on https://stackoverflow.com/a/76757349
class FIFOdict(dict):

    def __init__(self, maxSize):
        super().__init__()
        self.maxSize = maxSize
        self.addOrder = deque()

    def __setitem__(self, key, value):
        if len(self) == self.maxSize:  # max reached
            self.__delitem__(self.addOrder[0])  # remove first added
        super().__setitem__(key, value)
        self.addOrder.append(key)

    def __delitem__(self, key):  # removing 1st will be fast
        super().__delitem__(key)
        self.addOrder.remove(key)


if __name__ == "__main__":
    fifo_dict = FIFOdict(maxSize=3)
    fifo_dict["a"] = 1
    fifo_dict["b"] = 2
    fifo_dict["c"] = 3
    print(fifo_dict)  # {'a': 1, 'b': 2, 'c': 3}

    fifo_dict["d"] = 4
    print(fifo_dict)  # {'b': 2, 'c': 3, 'd': 4}

    fifo_dict["e"] = 5
    print(fifo_dict)  # {'c': 3, 'd': 4, 'e': 5}
