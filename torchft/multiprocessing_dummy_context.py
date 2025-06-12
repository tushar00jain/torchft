# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiprocessing Dummy Context
=========================

This module provides a context-like interface for multiprocessing.dummy,
which is a wrapper around the threading module that provides a multiprocessing-like
interface but uses threads instead of processes.

This allows code that uses multiprocessing.get_context() to work with
multiprocessing.dummy by providing a compatible interface.
"""

import multiprocessing.dummy as mp


class DummyContext:
    """
    A context-like class for multiprocessing.dummy that mimics the interface
    of a context returned by multiprocessing.get_context().
    """

    def __init__(self, method=None):
        """
        Initialize the dummy context.

        Args:
            method: Ignored, only for compatibility with multiprocessing.get_context()
        """
        pass

    def Process(self, *args, **kwargs):
        """
        Create a Process using multiprocessing.dummy.Process.
        """
        return mp.Process(*args, **kwargs)

    def Pipe(self, duplex=True):
        """
        Create a Pipe using multiprocessing.dummy.Pipe.
        """
        return mp.Pipe(duplex)

    def Queue(self, maxsize=0):
        """
        Create a Queue using multiprocessing.dummy.Queue.
        """
        return mp.Queue(maxsize)

    def Event(self):
        """
        Create an Event using multiprocessing.dummy.Event.
        """
        return mp.Event()

    def Lock(self):
        """
        Create a Lock using multiprocessing.dummy.Lock.
        """
        return mp.Lock()

    def RLock(self):
        """
        Create an RLock using multiprocessing.dummy.RLock.
        """
        return mp.RLock()

    def Semaphore(self, value=1):
        """
        Create a Semaphore using multiprocessing.dummy.Semaphore.
        """
        return mp.Semaphore(value)

    def BoundedSemaphore(self, value=1):
        """
        Create a BoundedSemaphore using multiprocessing.dummy.BoundedSemaphore.
        """
        return mp.BoundedSemaphore(value)

    def Condition(self, lock=None):
        """
        Create a Condition using multiprocessing.dummy.Condition.
        """
        return mp.Condition(lock)

    def Manager(self):
        """
        Create a Manager using multiprocessing.dummy.Manager.
        """
        return mp.Manager()


def get_context(method=None):
    """
    Return a context object for multiprocessing.dummy.

    This function mimics multiprocessing.get_context() but returns a DummyContext
    that works with multiprocessing.dummy. This can be used to patch
    multiprocessing.dummy like so


    ```
    import multiprocessing.dummy as mp
    from torchft.multiprocessing_dummy_context import get_context
    mp.get_context = get_context
    ```

    Args:
        method: Ignored, only for compatibility with multiprocessing.get_context()

    Returns:
        A DummyContext instance
    """
    return DummyContext(method)
