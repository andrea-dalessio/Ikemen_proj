import numpy as np
class MemoryStack:
    def __init__(self, stackSize=4):
        self.data = np.zeros((stackSize, 96, 96, 3), dtype=np.uint8)
        self.size = stackSize
        self.used = 0
        self.current = 0
        
    def record(self, state):
        if not self.ready():
            while not self.ready():
                self.data[self.current] = state
                self.current = (self.current + 1) % self.size
                self.used = min(self.size, self.used + 1)
                
        self.data[self.current] = state
        self.current = (self.current + 1) % self.size
        self.used = min(self.size, self.used + 1)
    
    def reset(self):
        self.data = np.zeros((self.size, 96, 96, 3), dtype=np.uint8)
        self.used = 0
        self.current = 0
        
    def get(self):
        indices = list(range(self.current, self.size)) + list(range(0, self.current))
        frames = [self.data[i] for i in indices]
        return np.concatenate(frames, axis=2)

    def ready(self):
        return self.used == self.size