import sys
import time



# Modified from https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping
class _ProgressBar(object):

    def __init__(self, iterations, animation_interval=0.5):
        self.iterations = iterations
        self.start = time.time()
        self.last = 0
        self.animation_interval = animation_interval

    def percentage(self, i):
        return 100 * i / float(self.iterations)

    def animate(self, i, e):
        pass

    def update(self, i):
        elapsed = time.time() - self.start
        i = i + 1

        if elapsed - self.last > self.animation_interval:
            self.animate(i + 1, elapsed)
            self.last = elapsed
        elif i == self.iterations:
            self.animate(i, elapsed)


# Modified from https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping
class _TextProgressBar(_ProgressBar):

    def __init__(self, iterations, printer):
        self.fill_char = '-'
        self.width = 40
        self.printer = printer

        _ProgressBar.__init__(self, iterations)
        self.update(0)

    def animate(self, i, elapsed):
        self.printer(self.progbar(i, elapsed))

    def progbar(self, i, elapsed):
        bar = self.bar(self.percentage(i))
        return "[%s] %i of %i complete in %.1f sec" % (
            bar, i, self.iterations, round(elapsed, 1))

    def bar(self, percent):
        all_full = self.width - 2
        num_hashes = int(percent / 100 * all_full)

        bar = self.fill_char * num_hashes + ' ' * (all_full - num_hashes)

        info = '%d%%' % percent
        loc = (len(bar) - len(info)) // 2
        return replace_at(bar, info, loc, loc + len(info))


def replace_at(str, new, start, stop):
    return str[:start] + new + str[stop:]


def consoleprint(s):
    if sys.platform.lower().startswith('nt'):
        print(s, '\r', end='')
    else:
        print(s)


def progress_bar(iters):
    return _TextProgressBar(iters, consoleprint)
