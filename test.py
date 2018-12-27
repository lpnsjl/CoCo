# -*- coding:utf-8 -*-


import threading


class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self, a):
        self.a = a

    def __new__(cls, *args, **kwargs):
        if not hasattr(Singleton, "_instance"):
            with Singleton._instance_lock:
                if not hasattr(Singleton, "_instance"):
                    Singleton._instance = object.__new__(cls)
        return Singleton._instance


obj1 = Singleton(3)
obj2 = Singleton(4)
print(obj1, obj2)
print(obj1.a, obj2.a)


def task(arg):
    obj = Singleton(3)
    print(obj.a)


for i in range(10):
    t = threading.Thread(target=task,args=[i,])
    t.start()
