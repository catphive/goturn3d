import sys
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud


def save_points(filename, points):
    """Assumes point in columns"""
    frame = pd.DataFrame(points[:, 0:3], columns=['x', 'y', 'z'])
    print(frame)
    cloud = PyntCloud(frame)

    cloud.to_file(filename)


# Print iterations progress
def print_progress(iteration, total):
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------

    iteration :
                Current iteration (Int)
    total     :
                Total iterations (Int)
    """
    str_format = "{0:.0f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(100 * iteration / float(total)))
    bar = '█' * filled_length + '-' * (100 - filled_length)

    sys.stdout.write('\r |%s| %s%%' % (bar, percents)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()