#!/usr/bin/env python
import os
import cProfile
import subprocess
import numpy as np
import json
import logging
from deepdiff import DeepDiff
from deepdiff import diff
from deepdiff.helper import pypy3, py_current_version
logging.basicConfig(level=logging.INFO)

current_pid = str(os.getpid())


def get_profile_of_usage(filename, t1, t2, **params):
    def _run():
        DeepDiff(t1, t2, log_frequency_in_sec=2, **params)
    P = cProfile.Profile()
    P.runcall(_run)
    P.dump_stats(f'results/{filename}.profile')


def plot_resource_usage(filename, t1, t2, freq, **params):
    subprocess.Popen(['psrecord', current_pid, '--interval', str(freq), '--plot', f'results/{filename}.png'], close_fds=True)
    DeepDiff(t1, t2, log_frequency_in_sec=2, **params)


def get_file_name(func_name, **params):
    suffix = 'pypy3' if pypy3 else py_current_version
    filename = f"{func_name}__{suffix}"
    for key, value in params.items():
        filename = f"{filename}__{key}={value}"
    print('-' * 20)
    print(filename)
    return filename


def benchmark_numpy_array(plot_usage, profile_usage):
    """
    Benchmarking numpy arrays and different python versions and cache sizes etc.
    """
    t1 = np.loadtxt('data/mat1.txt')
    t2 = np.loadtxt('data/mat2.txt')

    params = dict(
        ignore_order=True,
        cache_size=5000,
        cache_tuning_sample_size=500,
    )
    filename = get_file_name('benchmark_numpy_array', **params)

    if plot_usage:
        plot_resource_usage(filename, t1, t2, freq='.1', **params)
    elif profile_usage:
        get_profile_of_usage(filename, t1, t2, **params)


def benchmark_array_no_numpy(plot_usage, profile_usage):
    """
    Benchmarking numpy arrays and different python versions and cache sizes etc.
    """
    t1 = np.loadtxt('data/mat1.txt').tolist()
    t2 = np.loadtxt('data/mat2.txt').tolist()

    params = dict(
        ignore_order=True,
        cache_size=10000,
        cache_tuning_sample_size=20000,
    )
    filename = get_file_name('benchmark_array_no_numpy', **params)

    try:
        diff.np = None
        if plot_usage:
            plot_resource_usage(filename, t1, t2, freq='.1', **params)
        elif profile_usage:
            get_profile_of_usage(filename, t1, t2, **params)
    finally:
        diff.np = np


def benchmark_big_jsons(plot_usage, profile_usage):
    """
    Benchmarking numpy arrays and different python versions and cache sizes etc.
    """

    t1 = json.load(open('data/big_actual.json'))
    t2 = json.load(open('data/big_expected.json'))
    params = dict(
        ignore_order=True,
        cache_size=0,
        cache_tuning_sample_size=0,
        max_diffs=300000,
        max_passes=40000,
    )
    filename = get_file_name('benchmark_big_jsons', **params)

    if plot_usage:
        plot_resource_usage(filename, t1, t2, freq='.1', **params)
    elif profile_usage:
        get_profile_of_usage(filename, t1, t2, **params)


def benchmark_deeply_nested_a(plot_usage, profile_usage):
    params = dict(
        ignore_order=True,
        cache_size=500,
        cache_tuning_sample_size=500,
    )
    filename = get_file_name('benchmark_deeply_nested_a', **params)

    t1 = json.load(open('data/nested_a_t1.json'))
    t2 = json.load(open('data/nested_a_t2.json'))

    if plot_usage:
        plot_resource_usage(filename, t1, t2, freq='.1', **params)
    elif profile_usage:
        get_profile_of_usage(filename, t1, t2, **params)


def benchmark_deeply_nested_b(plot_usage, profile_usage):
    params = dict(
        ignore_order=True,
        cache_size=5000,
        cache_tuning_sample_size=0,
    )
    filename = get_file_name('benchmark_deeply_nested_b', **params)

    t1 = json.load(open('data/nested_b_t1.json'))
    t2 = json.load(open('data/nested_b_t2.json'))

    if plot_usage:
        plot_resource_usage(filename, t1, t2, freq='.1', **params)
    elif profile_usage:
        get_profile_of_usage(filename, t1, t2, **params)


if __name__ == '__main__':
    plot_usage = False
    profile_usage = False

    plot_usage = True
    # profile_usage = True

    # benchmark_array_no_numpy(plot_usage, profile_usage)
    # benchmark_big_jsons(plot_usage, profile_usage)
    # benchmark_deeply_nested_a(plot_usage, profile_usage)
    # benchmark_deeply_nested_b(plot_usage, profile_usage)
    benchmark_numpy_array(plot_usage, profile_usage)
