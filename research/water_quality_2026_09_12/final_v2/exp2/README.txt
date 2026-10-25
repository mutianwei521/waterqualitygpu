Exp2 v2 inputs of paper_v2 (build_tables.py, figures/paperfig.py DATA["exp2"]), in the v1 layout {..., "networks": [result]}.
exp2_BWSN2.json <- final_5090_v2/r1_A1_1723390/exp2_v2_bw200x.json (5090, 200 sensors, Slurm job 1723390)
exp2_LTown.json <- final_5090_v2/r1_A2_1723391/exp2_v2_lt10.json (5090, 10 sensors, Slurm job 1723391)
Same settings as the 3090 runs final_v2/exp2_v2_bw200x.json and exp2_v2_lt10.json; detection and GLRT results identical.
Layout mapping: timing.placement_s = v2.placement_v2_s (the v2 placement on 4,096 scenarios); timing.total_s = setup_s + placement_s
+ data_s + identification_s, the whole-study wall time as in v1 (exp2_v2 itself reports total_s without the shared setup).
