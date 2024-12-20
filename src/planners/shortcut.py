import numpy as np
from tqdm.auto import tqdm

class ShortcutOptions:
    def __init__(self, max_iters=1e2, check_size=1e-2):
        self.max_iters = int(max_iters)
        self.check_size = check_size

class Shortcut:
    def __init__(self, Metric, Interpolator, CollisionChecker, options):
        self.Metric = Metric
        self.Interpolator = Interpolator
        self.CollisionChecker = CollisionChecker
        self.options = options

    def shortcut(self, path, verbose=False):
        segment_lengths = [self.Metric(qi, qj)[0] for qi, qj in path]
        total_distance = np.sum(segment_lengths)
        weights = segment_lengths / total_distance

        total_shortcuts = 0
        for _ in tqdm(range(self.options.max_iters), disable=not verbose):
            start, stop = np.random.choice(len(path), size=2, replace=False, p=weights)
            if stop < start:
                start, stop = stop, start

            s, t = np.random.random(2)
            q1 = s * path[start][0] + (1 - s) * path[start][1]
            q2 = s * path[stop][0] + (1 - s) * path[stop][1]
            dist, q2_local = self.Metric(q1, q2)

            if self.CollisionChecker.CheckEdgeCollisionFreeParallel(q1, q2_local):
                path[start] = (path[start][0], q1)
                path[stop] = (q2, path[stop][1])
                del path[start+1:stop]
                path.insert(start+1, (q1, q2_local))

                # total_distance -= np.sum(segment_lengths[start:stop+1])

                segment_lengths[start] = self.Metric(path[start][0], path[start][1])[0]
                segment_lengths[start+2] = self.Metric(path[start+2][0], path[start+2][1])[0]
                del segment_lengths[start+1:stop]
                segment_lengths.insert(start+1, dist)

                # total_distance += np.sum(segment_lengths[start:start+3])

                total_distance = np.sum(segment_lengths)
                weights = segment_lengths / total_distance
                total_shortcuts += 1

        if verbose:
            print("Applied %d shortcuts" % total_shortcuts)
        return path