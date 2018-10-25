class LossHandler:
    def __init__(self):
        # name -> epoch -> list (per batch)
        self.logs = defaultdict(lambda: defaultdict(list))

    def __getitem__(self, *args, **kwargs):
        return self.logs.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.logs.__setitem__(*args, **kwargs)

    def log_epoch(self, writer, epoch):
        for k, v in self.logs.items():
            if epoch in v:
                print("{}[{}]: {}".format(k, epoch, np.mean(v[epoch])))
                writer.add_scalar(k, np.mean(v[epoch]), epoch)
