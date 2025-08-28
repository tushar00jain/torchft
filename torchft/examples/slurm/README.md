## Launch lighthouse

Run this command to launch the lighthouse somewhere and make sure other slurm nodes have access to this node


```
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

## Launch training

Run the following command to launch torchft lighthouse and replicas using torchtitan on slurm

```bash
$ # Set the address of the lighthouse server e.g.
$ export TORCHFT_LIGHTHOUSE=http://slurm-head-node-0:29510
$ python runner.py
```

## Test fault tolerance

To inject some failures, you can use the following command

```bash
$ python punisher.py kill_loop
```
