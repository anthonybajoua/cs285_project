class Scheduler:
    """
    Parent class of any learning scheduler method.
    """
    
    def __init__(self, num_items):
        pass
    
    def next_item(self):
        pass
    
    def update(self, item, outcome):
        pass
    

class Random(Scheduler):
    """
    Scheduler that selects random items to present.
    """
    def __init(self, num_items):
        self.n = num_items
    
    def next_item(self):
        return np.random.randint(0, num_items)
    
    def update(self, item, outcome):
        pass
        
        

class Leitner(Scheduler): 
    """
    This class implements a Leitner scheduler that samples from 
    boxes with exponentially decreasing probability. Cards enter
    in box 0 and leave when they are correctly answered after entering 
    the final box
    """
    def __init__(self, nb):
        '''
        :param nb: Number of boxes
        boxes is a list of queues representing the boxes.
        dist_boxes is sampling distribution for which box to select fromr
        cards is a set of items in the boxes currently.
        '''
        self.boxes = [deque() for _ in nb]
        self.dist_boxes = np.array([1/2**i for i in range(nb)]) / sum([1/2**i for i in range(nb)])
        self.cards = set()
        
    
    def next_item(self):
        """
        Gets the next item in the learning sequence.
        """
        self.recent_box = np.random.multinomial(1, self.dist_boxes).argmax()
        
        if len(self.boxes[self.recent_box]):
            return self.boxes[self.recent_box].pop()
        else:
            return self.next_item()
    
    def update(self, item, outcome, thresh=.9):
        """
        Updates the most recent item from the sequence
        by putting it back depending on the outcome.
        """
        if outcome > thresh:
            new_box = self.recent_box + 1
            if new_box >= len(self.boxes):
                self.cards.remove(item)
            else:
                self.boxes[new_box].appendleft(item)
        else:
            new_box = max(self.recent_box - 1, 0)    
            self.boxes[new_box].appendleft(item)
        