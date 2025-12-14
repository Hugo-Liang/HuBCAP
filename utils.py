def hunk_empty(hunk):
    if hunk.strip() == '':
        return True

    for line in hunk.split('\n'):
        if line[1:].strip() != '':
            return False

    return True

class EarlyStopping():
    """
    早停策略：当loss经过某些epochs后没有提升就停止训练
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def get_hunk_from_diff(diff:str):
    hunk_list = []
    hunk = ''
    add_hunk = ''
    del_hunk = ''

    for line in diff.split('\n'):
        if line.startswith(('+')):
            add_hunk = hunk + line[1:] + '\n'
        elif line.startswith(('-')):
            del_hunk = hunk + line[1:] + '\n'
        else:
            if not hunk_empty(hunk):    # finish a hunk
                hunk = hunk[:-1]
                hunk_list.append(hunk)
                hunk = ''

    if not hunk_empty(hunk):
        hunk_list.append(hunk)

    return hunk_list