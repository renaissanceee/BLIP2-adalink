[W socket.cpp:401] [c10d] The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use).
[W socket.cpp:401] [c10d] The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
[E socket.cpp:435] [c10d] The server socket has failed to listen on any local network address.
Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/run.py", line 765, in <module>
    main()
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 345, in wrapper
    return f(*args, **kwargs)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/run.py", line 761, in main
    run(args)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/run.py", line 752, in run
    elastic_launch(
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 131, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 236, in launch_agent
    result = agent.run()
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/metrics/api.py", line 125, in wrapper
    result = f(*args, **kwargs)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 709, in run
    result = self._invoke_run(role)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 844, in _invoke_run
    self._initialize_workers(self._worker_group)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/metrics/api.py", line 125, in wrapper
    result = f(*args, **kwargs)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 678, in _initialize_workers
    self._rendezvous(worker_group)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/metrics/api.py", line 125, in wrapper
    result = f(*args, **kwargs)
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 538, in _rendezvous
    store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py", line 55, in next_rendezvous
    self._store = TCPStore(  # type: ignore[call-arg]
RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
