from tqdm import tqdm
import time


class ProgressBar(object):  # Python3+ '(object)' can be omitted
    def __init__(self, max_value, disable=True):
        self.max_value = max_value
        self.disable = disable
        self.p = self.pbar()

    def pbar(self):
        return tqdm(
            total=self.max_value,
            desc='Step: ',
            disable=self.disable
        )

    def update(self, update_value):
        self.p.update(update_value)

    def close(self):
        self.p.close()
