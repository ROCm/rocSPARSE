#!/usr/bin/env python3

# ########################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################


import argparse
import subprocess
import os
import re # regexp package
import sys
import tempfile
import json



import rocsparse_bench_gnuplot_helper

#
# EXPORT TO PDF WITH GNUPLOT
# arg plot: "all", "gflops", "time", "bandwidth"
#
#
def export_gnuplot(plot, obasename,xargs, yargs, results,verbose = False,debug = False):

    datafile = open(obasename + ".dat", "w+")
    len_xargs = len(xargs)
    for iplot in range(len(yargs)):
        for ixarg  in range(len_xargs):
            isample = iplot * len_xargs + ixarg
            tg = results[isample]["timing"]
            datafile.write(os.path.basename(os.path.splitext(xargs[ixarg])[0]) + " " +
                           tg["time"][0] + " " +
                           tg["time"][1] + " " +
                           tg["time"][2] + " " +
                           tg["flops"][0] + " " +
                           tg["flops"][1] + " " +
                           tg["flops"][2] + " " +
                           tg["bandwidth"][0] + " " +
                           tg["bandwidth"][1] + " "+
                           tg["bandwidth"][2] + "\n")
        datafile.write("\n")
        datafile.write("\n")
    datafile.close();

    if verbose:
        print('//rocsparse-bench-plot  -  write gnuplot file : \'' + obasename + '.gnuplot\'')

    cmdfile = open(obasename + ".gnuplot", "w+")
    # for each plot
    num_curves=len(yargs)
    filetype="pdf"
    filename_extension= "." + filetype
    if plot == "time":
        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Time',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "milliseconds",
                                                 2,3,4,
                                                 yargs)
    elif plot == "gflops":
        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Performance',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GFlops",
                                                 5,6,7,
                                                 yargs)
    elif plot == "bandwidth":
        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + filename_extension,
                                                 'Bandwidth',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GBytes/s",
                                                 8,9,10,
                                                 yargs)
    elif plot == "all":
        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_msec"+ filename_extension,
                                                 'Time',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "milliseconds",
                                                 2,3,4,
                                                 yargs)


        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_gflops"+ filename_extension,
                                                 'Performance',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GFlops",
                                                 5,6,7,
                                                 yargs)

        rocsparse_bench_gnuplot_helper.histogram(cmdfile,
                                                 obasename + "_bandwidth"+ filename_extension,
                                                 'Bandwidth',
                                                 range(num_curves),
                                                 obasename + ".dat",
                                                 [-0.5,len_xargs + 0.5],
                                                 "GBytes/s",
                                                 8,9,10,
                                                 yargs)
    else:
        print("//rocsparse-bench-plot::error invalid plot keyword '"+plot+"', must be 'all' (default), 'time', 'gflops' or 'bandwidth' ")
        exit(1)
    cmdfile.close();

    rocsparse_bench_gnuplot_helper.call(obasename + ".gnuplot")
    if verbose:
        print('//rocsparse-bench-plot CLEANING')

    if not debug:
        os.remove(obasename + '.dat')
        os.remove(obasename + '.gnuplot')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workingdir',     required=False, default = './')
    parser.add_argument('-o', '--obasename',    required=False, default = 'a')
    parser.add_argument('-p', '--plot',    required=False, default = 'all')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    parser.add_argument('-d', '--debug',         required=False, default = False, action = "store_true")
    user_args, unknown_args = parser.parse_known_args()
    verbose=user_args.verbose
    debug=user_args.debug
    obasename = user_args.obasename
    plot = user_args.plot
    if len(unknown_args) > 1:
        print('expecting only one input file.')
    with open(unknown_args[0],"r") as f:
        case=json.load(f)

    cmd = case['cmdline']
    xargs = case['xargs']
    yargs = case['yargs']
    results = case['results']
    num_samples = len(results)
    len_xargs = len(xargs)

    if verbose:
        print('//rocsparse-bench-plot')
        print('//rocsparse-bench-plot  - file : \'' + unknown_args[0] + '\'')

    export_gnuplot(plot, obasename, xargs,yargs, results, verbose,debug)

if __name__ == "__main__":
    main()

