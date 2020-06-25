from types import SimpleNamespace


class SimakovOptim:
    def __init__(self,
                 pm,
                 *,
                 auto_thresh=0.99,
                 nb_max_it=10,
                 reuse_nn=True,
                 bidir=True,
                 cb=None):
        self.pm = pm
        self.auto_thresh = auto_thresh
        self.nb_max_it = nb_max_it
        self.reuse_nn = reuse_nn
        self.bidir = bidir
        self.cb = cb

    def one_pass(self, target, reference, nn_forward=None, nn_backward=None):
        arena = self.pm.arena(target.shape)
        if self.bidir:
            (nearest, ts), (nearest2, st) = self.pm.bidir(target, reference)
            arena = self.pm.vote(nearest, arena, reference)
            arena = self.pm.reverse_vote(nearest2, arena, reference)
        else:
            (nearest, ts) = self.pm.forward(target, reference)
            nearest2, st = None, 0
            arena = self.pm.vote(nearest, arena, reference)
        target = self.pm.render_arena(arena)
        return SimpleNamespace(target=target,
                               total=st + ts,
                               nn_forward=nearest,
                               nn_backward=nearest2)

    def __call__(self, target, reference, nn_forward=None, nn_backward=None):
        prev_total = 1e25
        for i in range(self.nb_max_it + 1):
            out = self.one_pass(target, reference, nn_forward, nn_backward)
            target, total = out.target, out.total
            nn_forward, nn_backward = out.nn_forward, out.nn_backward
            if self.cb:
                self.cb(target)
            print('em', i, total / (prev_total + 1e-8))
            if (total / (prev_total + 1e-8) > self.auto_thresh
                    or i >= self.nb_max_it):
                break
            if not self.reuse_nn:
                nn_forward = None
                nn_backward = None
            prev_total = total
        return target, nn_forward, nn_backward
