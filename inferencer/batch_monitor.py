

class BatchMonitor():
    n_images_completed: int = 0
    n_images_total: int = 0
    ontick_callback_func = lambda x, y: None
    n_batches: int = 1
    cancelled: bool = False
    finished: bool = False
    alt_msg: str = ''

    def set_cancelled(self):
        self.cancelled = True

    def get_total_images(self):
        return self.n_images_total

    def set_total_images(self, n_images_total):
        self.n_images_total = n_images_total

    def get_completed_images(self):
        return self.n_images_completed

    def initialise_tick(self):
        if self.n_images_total > 0:
            self.on_tick(0, self.n_images_total)

    def tick(self):
        self.n_images_completed += 1
        self.on_tick(self.n_images_completed, self.n_images_total)

    def tick(self, i):
        self.n_images_completed += i
        self.on_tick(self.n_images_completed, self.n_images_total)

    def set_callback_on_tick(self, callback_func):
        self.ontick_callback_func = callback_func

    def on_tick(self, completed, total):
        self.ontick_callback_func(completed, total)

    def log(self, msg):
        self.alt_msg = self.alt_msg + str(msg) + '\n'