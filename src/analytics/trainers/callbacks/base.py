

class Callback(object):
    """Callback basic implementation"""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self):
        pass

    def on_before_eval(self):
        pass

    def on_train_end(self):
        pass
