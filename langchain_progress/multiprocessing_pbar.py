from multiprocessing import Manager, Process, Value

from .utils.import_utils import is_tqdm_installed

if is_tqdm_installed():
    from tqdm import tqdm


class MultiprocessingPBar:
    '''Handles the creation of the multiprocessing queue and process to update the progress bar.'''
    def __init__(self, pbar=None, total=None):
        self.pbar = pbar
        self.total = total
        self.context_alive = Value('b', False)

        if pbar is None and not is_tqdm_installed():
            raise ImportError(
                '`tqdm` is not installed. Please install `tqdm` to use MultiprocessingPBar.')
        elif pbar is None:
            self.pbar = tqdm(total=total)
        elif not isinstance(pbar, tqdm):
            raise TypeError('`pbar` must be an instance of `tqdm.tqdm`.')

    def _tqdm_process(self):
        '''Update the progress bar with the value from the queue.'''
        while self.context_alive.value:
            if not self.queue.empty():
                update_value = self.queue.get()
                self.pbar.update(update_value)

        self.pbar.close()

    def __enter__(self):
        self.context_alive.value = True

        self.manager = Manager()
        self.queue = self.manager.Queue()

        self.tqdm_process = Process(target=self._tqdm_process)
        self.tqdm_process.start()

        return self.queue
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.context_alive.value = False
        self.queue.put(0)
        self.tqdm_process.join()
        self.manager.shutdown()