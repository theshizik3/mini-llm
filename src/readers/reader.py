from abc import ABC, abstractmethod


class Reader:
    @abstractmethod
    def __iter__(self):
        pass
