class EarlyStopping:
    def __init__(self, patience=0, delta=0):
        """Class for early stopper

        Args:
            patience (int, optional): The number of unchanged loss values step. Defaults to 0.
            delta (int, optional): Minimum distance between loss and a better loss. Defaults to 0.
        """
        self.patience = patience
        self.delta = delta
        self.wait = 0
        self.mode = "inactive"
        self.counter = 0
    def set_mode(self, mode):
        self.mode = mode
        
    def get_mode(self):
        return self.mode
  
    def track(self, old_loss, new_loss):
        if self.mode == "inactive":
            if new_loss >= old_loss - self.delta:
                if self.counter == 0:
                    self.mode = "active"
                    self.counter = self.patience
                else:
                    self.counter -= 1
            return
    def invest(self, old_loss, new_loss):
        if new_loss < old_loss - self.delta /10:
            self.mode = "inactive"
            return True
        else:
            return False
            