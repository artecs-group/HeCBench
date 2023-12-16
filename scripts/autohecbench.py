#!/usr/bin/env python3
#
# Script to run HeCBench benchmarks and gather results

import re, time, sys, subprocess, multiprocessing, os, shutil
import argparse
import json
import statistics
import random


class Benchmark:
    def __init__(self, args, name, res_regex, run_args = [], binary = "main", invert = False):
        if name.endswith('sycl'):
            self.MAKE_ARGS = ['GCC_TOOLCHAIN="{}"'.format(args.gcc_toolchain)]
            self.MAKE_ARGS = self.MAKE_ARGS + args.sycl_fcompile.split(',')
            if args.sycl_type == 'cuda':
                self.MAKE_ARGS.append('CUDA=yes')
                self.MAKE_ARGS.append('CUDA_ARCH=sm_{}'.format(args.nvidia_sm))
                self.device = 'ngpu'
            elif args.sycl_type == 'hip':
                self.MAKE_ARGS.append('HIP=yes')
                self.MAKE_ARGS.append('HIP_ARCH={}'.format(args.amd_arch))
            elif args.sycl_type == 'opencl':
                self.MAKE_ARGS.append('CUDA=no')
                self.MAKE_ARGS.append('HIP=no')
                self.device = 'igpu'
        elif name.endswith('kokkos'):
            self.MAKE_ARGS = args.kokkos_fcompile.split(',')
            if args.kokkos_type == 'ngpu':
                self.MAKE_ARGS.append('-DDEVICE=ngpu')
                self.device = 'ngpu'
            elif args.kokkos_type == 'igpu':
                self.MAKE_ARGS.append('-DDEVICE=igpu')
                self.device = 'igpu'
            elif args.kokkos_type == 'cpu':
                self.MAKE_ARGS.append('-DDEVICE=cpu')
        elif name.endswith('cuda'):
            self.MAKE_ARGS = ['CUDA_ARCH=sm_{}'.format(args.nvidia_sm)]
            self.MAKE_ARGS = self.MAKE_ARGS + args.cuda_fcompile.split(',')
            self.device = 'ngpu'
        else:
            self.MAKE_ARGS = []
            self.device = ''

        if args.extra_compile_flags:
            flags = args.extra_compile_flags.split(',')
            self.MAKE_ARGS = self.MAKE_ARGS + flags

        if args.bench_dir:
            self.path = os.path.realpath(os.path.join(args.bench_dir, name))
        else:
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', name)

        if name.endswith('kokkos'):
            self.path = os.path.join(self.path, 'build')
            if not os.path.exists(self.path):
                os.makedirs(self.path)

        self.power = args.power
        if self.power != '' and not os.path.exists(self.power):
            os.makedirs(self.power)

        self.name = name
        self.binary = binary
        self.res_regex = res_regex
        self.args = run_args
        self.invert = invert
        self.clean = args.clean
        self.verbose = args.verbose

    def compile(self):
        if self.clean:
            if self.name.endswith('kokkos'):
                shutil.rmtree(os.path.join(self.path))
                os.makedirs(self.path)
            else:
                subprocess.run(["make", "clean"], cwd=self.path).check_returncode()
                time.sleep(1) # required to make sure clean is done before building, despite run waiting on the invoked executable

        out = subprocess.DEVNULL
        if self.verbose:
            out = subprocess.PIPE

        if self.name.endswith('kokkos'):
            proc0 = subprocess.run(["cmake", ".."] + self.MAKE_ARGS, cwd=self.path, stdout=out, stderr=subprocess.STDOUT, encoding="ascii")
            try:
                proc0.check_returncode()
            except subprocess.CalledProcessError as e:
                print(f'Failed compilation in {self.path}.\n{e}')
                if e.stderr:
                    print(e.stderr, file=sys.stderr)
                raise(e)

        command = ["make"] if self.name.endswith('kokkos') else ["make"] + self.MAKE_ARGS
        proc = subprocess.run(command, cwd=self.path, stdout=out, stderr=subprocess.STDOUT, encoding="ascii")
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f'Failed compilation in {self.path}.\n{e}')
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            raise(e)

        if self.verbose:
            print(proc.stdout)

    def run(self):
        powerProc = None
        if self.power != '':
            pcmd = []
            if self.device == 'ngpu':
                pcmd = ['sudo', 'tegrastats', '--interval', '1000']
            elif self.device == 'igpu':
                pcmd = ['sudo', 'turbostat', '--Summary', '--quiet', '--show', 'PkgTmp,PkgWatt,GFXMHz,GFXWatt,RAMWatt,CorWatt', '--interval', '1']
            if len(pcmd) > 0:
                powerProc = subprocess.Popen(pcmd, stdout=subprocess.PIPE)

        cmd = ["./" + self.binary] + self.args
        proc = subprocess.run(cmd, cwd=self.path, stdout=subprocess.PIPE, encoding="ascii")

        if powerProc:
            if self.device == 'igpu':
                powerProc.kill()
            elif self.device == 'ngpu':
                subprocess.run(["sudo", "tegrastats", "--stop"])
            power_out, _ = powerProc.communicate()
            with open(os.path.join(self.power, self.name + '.txt'), 'w') as file:
                file.write(power_out.decode('ascii'))

        out = proc.stdout
        if self.verbose:
            print(" ".join(cmd))
            print(out)
        res = re.findall(self.res_regex, out)
        if not res:
            raise Exception(self.path + ":\nno regex match for " + self.res_regex + " in\n" + out)
        res = sum([float(i) for i in res]) #in case of multiple outputs sum them
        if self.invert:
            res = 1/res
        time.sleep(5)
        return res


def comp(b):
    print("compiling: {}".format(b.name))
    b.compile()

def main():
    parser = argparse.ArgumentParser(description='HeCBench runner')
    parser.add_argument('--output', '-o',
                        help='Output file for csv results')
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='Repeat benchmark run')
    parser.add_argument('--warmup', '-w', type=bool, default=False,
                        help='Run a warmup iteration')
    parser.add_argument('--sycl-type', '-s', choices=['cuda', 'hip', 'opencl'], default='cuda',
                        help='Type of SYCL device to use')
    parser.add_argument('--kokkos-type', '-k', choices=['cpu', 'ngpu', 'igpu'], default='ngpu',
                        help='Type of Kokkos device to use')
    parser.add_argument('--nvidia-sm', type=int, default=60,
                        help='NVIDIA SM version')
    parser.add_argument('--amd-arch', default='gfx908',
                        help='AMD Architecture')
    parser.add_argument('--gcc-toolchain', default='',
                        help='GCC toolchain location')
    parser.add_argument('--extra-compile-flags', '-e', default='',
                        help='Additional compilation flags (inserted before the predefined CFLAGS)')
    parser.add_argument('--clean', '-c', action='store_true',
                        help='Clean the builds')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Clean the builds')
    parser.add_argument('--bench-dir', '-b',
                        help='Benchmark directory')
    parser.add_argument('--bench-data', '-d',
                        help='Benchmark data')
    parser.add_argument('--bench-fails', '-f',
                        help='List of failing benchmarks to ignore')
    parser.add_argument('bench', nargs='+',
                        help='Either specific benchmark name or sycl, cuda, hip or kokkos')
    parser.add_argument('--power', '-p',
                        help='Measure power consumption (requires root)', default='')
    parser.add_argument('--cuda-fcompile',
                        help='Cuda extra compilation flags', default='')
    parser.add_argument('--sycl-fcompile',
                        help='SYCL extra compilation flags', default='')
    parser.add_argument('--kokkos-fcompile',
                        help='Kokkos extra compilation flags', default='')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load benchmark data
    if args.bench_data:
        bench_data = args.bench_data
    else:
        bench_data = os.path.join(script_dir, 'benchmarks', 'subset.json') 

    with open(bench_data) as f:
        benchmarks = json.load(f)

    # Load fail file
    if args.bench_fails:
        bench_fails = os.path.abspath(args.bench_fails)
    else:
        bench_fails = os.path.join(script_dir, 'benchmarks', 'subset-fails.txt')

    with open(bench_fails) as f:
        fails = f.read().splitlines()

    # Build benchmark list
    benches = []
    for b in args.bench:
        if b in ['sycl', 'cuda', 'hip', 'kokkos']:
            benches.extend([Benchmark(args, k, *v)
                            for k, v in benchmarks.items()
                            if k.endswith(b) and k not in fails])
            continue

        benches.append(Benchmark(args, b, *benchmarks[b]))

    t0 = time.time()
    try:
        with multiprocessing.Pool() as p:
            p.map(comp, benches)
    except Exception as e:
        print("Compilation failed, exiting")
        print(e)
        sys.exit(1)

    t_compiled = time.time()

    outfile = sys.stdout
    if args.output:
        outfile = open(args.output, 'w')
        print("benchmark,repetitions,avg-time,median-time,stdev,times", file=outfile)
    
    # randomize order of benchmarks
    random.shuffle(benches)
    results = {b.name: [] for b in benches}

    for i in range(args.repeat):
        for b in benches:
            try:
                print(f"running: {b.name} in iteration {i}")

                if args.warmup:
                    b.run()

                results[b.name].append(float(b.run()))
            except Exception as err:
                print("Error running: ", b.name)
                print(err)

    # save results in csv
    for b in benches:
        avg_time:float = sum([float(i) for i in results[b.name]])/args.repeat
        median_time:float = sorted(results[b.name])[len(results[b.name])//2]
        stdev_time:float = statistics.stdev(results[b.name]) if len(results[b.name]) > 1 else 0.0
        str_results = [str(i) for i in results[b.name]]
        out:str = f"{b.name},{args.repeat},{avg_time},{median_time},{stdev_time},[{','.join(str_results)}]"
        print(out, file=outfile)

    if args.output:
        outfile.close()

    t_done = time.time()
    print("compilation took {} s, runnning took {} s.".format(t_compiled-t0, t_done-t_compiled))

if __name__ == "__main__":
    main()

