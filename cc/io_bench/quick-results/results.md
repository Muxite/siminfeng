| pattern | engine | direct | request cap | workers | MB/s | p50 ms | p95 ms | resident before/after | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| random-expert | python | 0 | 262144 | 1 | 81.8 | 2.112 | 9.611 | 0.0/1.0 | ok |
| random-expert | psync | 0 | 262144 | 1 | 32.5 | 0.901 | 8.203 | 0.0/1.0 | ok |
| random-expert | psync | 1 | 262144 | 1 | 84.9 | 1.261 | 9.376 | 0.0/0.0 | ok |
| random-expert | io_uring | 0 | 262144 | 1 | - | - | - | 0.0/? | error |
| random-expert | io_uring | 1 | 262144 | 1 | - | - | - | 0.0/? | error |
| random-expert | python | 0 | 262144 | 16 | 206.0 | 19.747 | 33.110 | 0.0/1.0 | ok |
| random-expert | psync | 0 | 262144 | 16 | 268.4 | 10.592 | 28.763 | 0.0/1.0 | ok |
| random-expert | psync | 1 | 262144 | 16 | 322.6 | 12.238 | 15.036 | 0.0/0.0 | ok |
| random-expert | io_uring | 0 | 262144 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | io_uring | 1 | 262144 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | python | 0 | 2097152 | 1 | 170.7 | 8.121 | 43.012 | 0.0/1.0 | ok |
| random-expert | psync | 0 | 2097152 | 1 | 213.7 | 8.797 | 27.223 | 0.0/1.0 | ok |
| random-expert | psync | 1 | 2097152 | 1 | 246.7 | 4.622 | 21.378 | 0.0/0.0 | ok |
| random-expert | io_uring | 0 | 2097152 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | io_uring | 1 | 2097152 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | python | 0 | 2097152 | 16 | 431.5 | 73.817 | 76.881 | 0.0/1.0 | ok |
| random-expert | psync | 0 | 2097152 | 16 | 60.3 | 526.444 | 554.998 | 0.0/1.0 | ok |
| random-expert | psync | 1 | 2097152 | 16 | 399.5 | 49.189 | 83.562 | 0.0/0.0 | ok |
| random-expert | io_uring | 0 | 2097152 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | io_uring | 1 | 2097152 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | python | 0 | 262144 | 1 | 134.0 | 1.322 | 5.839 | 0.0/1.0 | ok |
| sequential | psync | 0 | 262144 | 1 | 212.4 | 0.719 | 3.149 | 0.0/1.0 | ok |
| sequential | psync | 1 | 262144 | 1 | 108.9 | 1.813 | 4.790 | 0.0/0.0 | ok |
| sequential | io_uring | 0 | 262144 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | io_uring | 1 | 262144 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | python | 0 | 262144 | 16 | 284.7 | 12.956 | 18.708 | 0.0/1.0 | ok |
| sequential | psync | 0 | 262144 | 16 | 329.0 | 10.940 | 21.615 | 0.0/0.98046875 | ok |
| sequential | psync | 1 | 262144 | 16 | 424.7 | 6.849 | 17.757 | 0.0/0.0 | ok |
| sequential | io_uring | 0 | 262144 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | io_uring | 1 | 262144 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | python | 0 | 2097152 | 1 | 64.6 | 35.596 | 64.368 | 0.0/1.0 | ok |
| sequential | psync | 0 | 2097152 | 1 | 220.8 | 9.145 | 22.418 | 0.0/1.0 | ok |
| sequential | psync | 1 | 2097152 | 1 | 162.1 | 12.989 | 25.052 | 0.0/0.0 | ok |
| sequential | io_uring | 0 | 2097152 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | io_uring | 1 | 2097152 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | python | 0 | 2097152 | 16 | 259.0 | 110.632 | 129.138 | 0.0/1.0 | ok |
| sequential | psync | 0 | 2097152 | 16 | 294.3 | 107.493 | 113.031 | 0.0/1.0 | ok |
| sequential | psync | 1 | 2097152 | 16 | 244.9 | 71.374 | 136.634 | 0.0/0.0 | ok |
| sequential | io_uring | 0 | 2097152 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | io_uring | 1 | 2097152 | 16 | - | - | - | ?/? | unsupported (see earlier error) |

ceiling at engine pattern (2 MiB/16 workers buffered): 431.5 MB/s
