# -*- coding: utf-8 -*-

"""
author: Diogo Silva
Utils.
"""

from timeit import default_timer as timer

import sys

class Timer():
    def __init__(self, ID = None):
        if ID is not None:
              self.id = str(ID)
        self.start = None
        self.end = None
        self.elapsed = 0

    def tic(self):
        self.start = timer()

    def tac(self):
        self.end = timer()
        self.elapsed += self.end - self.start
        return self.elapsed

    def reset(self):
        self.elapsed = 0

    def wrap_function(self, fn):
        """Returns a new function that is wrapped around the timer. Whenever the
        new function is called it starts the timer with the _tic_ method, calls
        the original function _fn_ with the received arguments, saves the return
        values, stops the timer with the _tac_ method and finally returns the 
        return values. The elapsed time is stored in the timer.
        """
        def timed_fn(*args,**kwargs):
            self.tic()
            returnvals = fn(*args,**kwargs)
            self.tac()
            return returnvals
        return timed_fn

#    def __repr__(self):

    def __str__(self):
        id_str = ""
        if hasattr(self, 'id'):
            id_str = self.id + " : "

        if hasattr(self, 'elapsed'):
            return_str = ("{} Elapsed time: {} s, "
                         "{} ms".format(id_str,
                                        self.elapsed,
                                        self.elapsed * 1000))
        else:
            return_str = "{} No time recorded yet.".format(id_str)

        return return_str



if __name__ == "__main__":
    sys.exit(0)







