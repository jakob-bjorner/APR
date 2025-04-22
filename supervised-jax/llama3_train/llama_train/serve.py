from typing import Generator, Any, Optional
import redis
import time
import pickle as pkl
import multiprocessing as mp
from functools import partial
import six
import json
from collections import OrderedDict
# config for server

class Config:
    redis_host = 'localhost'
    redis_port = 6379
    redis_db = 0
    client_refresh_delay = 0.01
    self_indicator = '__self__'
    init_message = '__init_message__'
    sse_channel_prefix = "__sse_channel__"
    sse_exit_type = "__EXIT__"

"""
=====
Below is the code for running a class on a seperate process.
Each call to a method on the class is executed in a queue.
You want to do this when serving models to process 1 request at a time.
=====
"""

def serve_class(cls):
    cache_cls = pkl.dumps(cls)
    
    class WrappedModel:
        def __init__(self, *args, **kwargs):
            self.r = redis.Redis(host=Config.redis_host, port=Config.redis_port, db=Config.redis_db)
            self.Q = initalize_server(self, super().__getattribute__('r'), cache_cls, args, kwargs)
        
        def __getattribute__(self, name):
            return partial(build_method(name, super().__getattribute__('r'), super().__getattribute__('Q')), self)
    
        def __call__(self, *args, **kwargs):
            return build_method('__call__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self, *args, **kwargs)
        
        def __getitem__(self, key):
            return build_method('__getitem__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self, key)
        
        def __setitem__(self, key, value):
            return build_method('__setitem__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self, key, value)
        
        def __contains__(self, key):
            return build_method('__contains__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self, key)
        
        def __len__(self):
            return build_method('__len__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self)
    
    return WrappedModel

def build_method(method, r, Q):
    def call_method(self, *args, **kwargs):
        request_id = int(r.incr('request_id_counter'))
        Q.put((request_id, method, args, kwargs,))
        while not r.exists(f'result_{request_id}'):
            time.sleep(Config.client_refresh_delay)
        result = pkl.loads(r.get(f'result_{request_id}'))
        r.delete(f'result_{request_id}')
        if result == Config.self_indicator:
            return self
        return result
    return call_method

def server_process(Q, cls_pkl, args, kwargs):
    r = redis.Redis(host=Config.redis_host, port=Config.redis_port, db=Config.redis_db)
    model = pkl.loads(cls_pkl)(*args, **kwargs)
    while True:
        try:
            request_id, method, args, kwargs = Q.get()
            if method == Config.init_message:
                r.set(f'result_{request_id}', pkl.dumps(method))
                continue
            result = getattr(model, method)(*args, **kwargs)
            if isinstance(result, Generator):
                result = tuple(result)
            if result == model:
                result = Config.self_indicator
            r.set(f'result_{request_id}', pkl.dumps(result))
        except EOFError:
            return
        except Exception as e:
            raise Exception

def initalize_server(self, r, cls_pkl, args, kwargs):
    Q = mp.Manager().Queue()
    p = mp.Process(target=server_process, args=(Q, cls_pkl, args, kwargs))
    p.start()
    build_method(Config.init_message, r, Q)(self)
    return Q

