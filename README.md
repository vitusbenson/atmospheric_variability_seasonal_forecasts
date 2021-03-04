
# The role of atmospheric variability for seasonal forecasts

This repo contains one Plotting class that reads ECHAM6 output netCDF files. It currently supports runs saved in a base-directory, which are named `T63_short_r{n}` where `n` in `[1,2,3,4,5,6,8,9,10,11,12]`. From these it uses all `BOT_*.nc` files in the folder `POST`. The files are saved under a given filepath as a PDF.


Run from Commandline:
```
python avsf/plots.py path-to-base-directory path-to-output-file
```

