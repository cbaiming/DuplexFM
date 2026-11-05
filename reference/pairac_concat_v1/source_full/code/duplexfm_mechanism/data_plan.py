import math
import numpy as np
SEED=42

class BatchSchedule:
    """Original epoch/batch sizes, even when a cleaned pool needs repeated passes.

    A step is an unambiguous cursor. Pool shuffles use independent RNG; restoring
    a model/optimizer RNG never changes which examples the next step consumes.
    """
    def __init__(self, fit, reference_rows, batch_size, seed=SEED):
        self.fit = np.asarray(fit, dtype=np.int64)
        if (self.fit.ndim != 1 or len(self.fit) == 0 or len(np.unique(self.fit)) != len(self.fit)
                or reference_rows < len(self.fit) or batch_size < 1):
            raise ValueError("Invalid fixed training schedule")
        self.reference_rows = int(reference_rows)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = math.ceil(reference_rows / batch_size)
        self.seed = seed
        self._cycle = None
        self._order = None

    def at(self, step):
        if step < 0:
            raise ValueError("Negative training cursor")
        epoch, block = divmod(int(step), self.steps_per_epoch)
        begin = epoch * self.reference_rows + block * self.batch_size
        count = min(self.batch_size, self.reference_rows - block * self.batch_size)
        pieces = []
        while count:
            cycle, offset = divmod(begin, len(self.fit))
            if self._cycle != cycle:
                self._order = np.random.default_rng(self.seed * 100003 + cycle + 1).permutation(self.fit)
                self._cycle = cycle
            take = min(count, len(self.fit) - offset)
            pieces.append(self._order[offset:offset + take].copy())
            begin += take
            count -= take
        return np.concatenate(pieces)
