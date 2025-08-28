## Launch training

Run the following command to launch torchft lighthouse and replicas using torchtitan on slurm

```bash
$ python runner.py
```

## Test fault tolerance

To inject some failures, you can use the following command

```bash
$ python punisher.py kill_loop
```
