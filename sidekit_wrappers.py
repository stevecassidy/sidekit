# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2015 Anthony Larcher

:mod:`sidekit_wrappers` provides wrappers for different purposes.
The aim when using wrappers is to simplify the development of new function
in an efficient manner
"""

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2015 Anthony Larcher"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'

import os
import numpy as np
import logging
import copy
from sidekit import PARALLEL_MODULE

def check_path_existance(func):
    """ Decorator for a function wich prototype is:
    
        func(features, outputFileName)
        
        This decorator get the path included in 'outputFileName' if any 
        and check if this path exists; if not the path is created.
    """
    def wrapper(*args, **kwargs):
        dir_name = os.path.dirname(args[1])  # get the path
        # Create the directory if it dosn't exist
        if not os.path.exists(dir_name) and (dir_name is not ''):
            os.makedirs(dir_name)            
        # Do the job
        func(*args, **kwargs)
    return wrapper


def process_parallel_lists(func):
    print("We are going to parallelize {} using the {} module".format(func.__name__, PARALLEL_MODULE))
    def wrapper(*args, **kwargs):
        
        numThread = 1
        if "numThread" in kwargs.keys():
            numThread = kwargs["numThread"] 
        
        # On créé un dictionnaire de paramètres kwargs pour chaque thread
        if PARALLEL_MODULE in ['threading', 'multiprocessing']:# and numThread > 1:
            
            print("Run {} process in parallel".format(numThread))
            
            # Create a list of dictionaries, one per thread, and initialize
            # them with the keys
            parallel_kwargs = []
            for ii in range(numThread):
                parallel_kwargs.append(dict(zip(kwargs.keys(), 
                                            [None]*len(kwargs.keys()))))
            
            for k, v in kwargs.iteritems():
                
                # If v is a list or a numpy.array
                if k.endswith("_list") or k.endswith("_indices"):
                    sub_lists = np.array_split(v, numThread)
                    for ii in range(numThread):
                        parallel_kwargs[ii][k] = sub_lists[ii]  # the ii-th sub_list is used for the thread ii
 
                # If v is an accumulator (meaning k ends with "_acc")
                # v is duplicated for each thread
                elif k.endswith("_acc"):
                    for ii in range(numThread):
                        #parallel_kwargs[ii][k] = copy.deepcopy(v)
                        parallel_kwargs[ii][k] = v

                # All other parameters are just given to each thread
                else:
                    for ii in range(numThread):
                        parallel_kwargs[ii][k] = v;
            
            if PARALLEL_MODULE is 'multiprocessing':
                import multiprocessing
                jobs = []
                multiprocessing.freeze_support()
                for idx in range(numThread):
                    p = multiprocessing.Process(target=func,
                            args=args, kwargs=parallel_kwargs[idx])
                    jobs.append(p)
                    p.start()
                for p in jobs:
                    p.join()
            
            elif PARALLEL_MODULE is 'threading':
                print("kwargs['llk_acc'] avant threading = {}".format(kwargs['llk_acc']))
                import threading
                jobs = []
                for idx in range(numThread):
                    p = threading.Thread(target=func, 
                           args=args, kwargs=parallel_kwargs[idx])
                    jobs.append(p)
                    p.start()
                for p in jobs:
                    p.join()
                print("kwargs['llk_acc'] apres threading = {}".format(kwargs['llk_acc']))
        
            elif PARALLEL_MODULE is 'MPI':
                # TODO
                print("ParallelProcess using MPI is not implemented yet")
                pass
            
            # Sum accumulators if any
            for k, v in kwargs.iteritems():
                if k.endswith("_acc"):
                    for ii in range(numThread):
                        if isinstance(kwargs[k], list):
                            kwargs[k][0] += parallel_kwargs[ii][k][0]
                        else:
                            kwargs[k] += parallel_kwargs[ii][k]

        else:
            print("No Parallel processing with this module")
            func(*args, **kwargs)
        
    return wrapper




