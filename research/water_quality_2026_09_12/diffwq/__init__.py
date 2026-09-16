"""diffwq: record-replay differentiable water quality engine for EPANET 2.3.

Stage A (this package, CPU): exact Python replication of EPANET's Lagrangian
segment transport driven by hydraulics read from the engine's own binary
hydraulics file, producing a transport schedule.
Stage B (torch_replay): PyTorch replay of the schedule under autograd.
"""
