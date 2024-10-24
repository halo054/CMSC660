The csv files required the python program and the matlab program are different.

Half of Matlab program should be run first till "M.csv", " Lsymm.csv", and "bsymm.csv" are generated.

Then python program should be run. "x_sliced.csv", and "x_sliced_2.csv" will be generated, as the result of the CG algorithm without and with preconditioning.

Then the rest of matlab program could be run. The rest of the code import the result from CG algorithm, and get the result visualized.