# Benchmark data

The assignment uses the public TSPLIB instances `att48`, `ch130`, and `a280`.
`download_data.py` downloads the coordinate files from a public TSPLIB95
mirror and verifies that each file contains a coordinate section. The official
TSPLIB site is:
https://www.math.uwaterloo.ca/tsp/world/tsplib.html

Run:

```bash
python3 data/download_data.py
```

The downloaded `.tsp` files are intentionally ignored by Git. Record the
source URLs and the download date in experiment logs or the final report.
The published best-known tour distance for `att48` is 10628 and for `ch130`
is 6110. The published best-known distance for `a280` is 2579. These values
are used by the summary script for optimality gaps.
