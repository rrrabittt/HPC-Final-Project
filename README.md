# HPC-Final-Project
Project details: [Final Project.pdf](https://github.com/rrrabittt/HPC-Final-Project/files/8596876/Final.Project.pdf)

Options in command line:
`-mesh_size` grid size in spacial domain, default value is `10`, the number of nodes is `mesh_size`+1.

`-kappa` conductivity, default value is `1.0`.

`-rho` density, default value is `1.0`.

`-capacity` heat capacity, default value is `1.0`.

`-time_step` time step size, default value is `0.1`.

`-time_end` the end time of the problem, default value is `1.0`, `time_end` should be divisible by `time_step`.

`-gg_head` prescribed temperature boundary condition at x=0, default value is `0.0`.

`-gg_tail` prescribed temperature boundary condition at x=1, default value is `0.0`.

`-hh_head` heat flux boundary condition at x=0, default value is `1.0`.

`-hh_tail` heat flux boundary condition at x=1, default value is `1.0`.

`-head_bc` boundary condition at x=0: `0` for prescribed temperature, `1` for heat flux, default value is `0`.

`-tail_bc` boundary condition at x=1: `0` for prescribed temperature, `1` for heat flux, default value is `0`.

`-is_explicit` use explicit method if it is true, otherwise use implicit method, default value is `false`.

`-is_record` record current solution into hdf5 file if it is true, default value is `true`.

`-record_frq` frequency of recording, default value is `10`, should be set reasonably if `is_record` is `true`.

`-is_restart` read data from hdf5 file if it is true, default value is `false`, the input hdf5 file should be named as `SOL_re`.
