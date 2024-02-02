/home/stud/zhangya/jiameng/BLIP2-adalink/env/bin/python: can't open file 'train.py': [Errno 2] No such file or directory
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 2) local_rank: 0 (pid: 3169136) of binary: /home/stud/zhangya/jiameng/BLIP2-adalink/env/bin/python
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
  File "/home/stud/zhangya/jiameng/BLIP2-adalink/env/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 245, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-02-02_15:05:45
  host      : worker-6
  rank      : 0 (local_rank: 0)
  exitcode  : 2 (pid: 3169136)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
