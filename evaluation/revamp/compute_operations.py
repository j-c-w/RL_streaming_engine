import json
import argparse
import os
import subprocess

class Benchmark:
    def __init__(self, file, weight):
        self.file = file
        self.weight = weight

def load_benchmarks_from_json(jsonfile):
    benchmarks = []
    with open(jsonfile) as f:
        bench = json.load(f)
        descr = bench['benchmarks']
        for d in descr:
            benchmark_file = d['file']
            weight = float(d['weight'])

            benchmarks.append(Benchmark(benchmark_file, weight=weight))
    return benchmarks

def load_specification_from_json(jsonfile):
    with open(jsonfile) as f:
        spec = json.load(f)
    return spec

# Run the cgra-mapper tool, which will compute the operation
# distribution.
def compute_distribution(description, benchmark):
    subprocess.run(['cgra-mapper', benchmark.file, '--params-file ' + description + ' --frequencies ' + os.getcwd() + '/features_output.json --skip-build'])
    # Load the result into a dict.
    with open('features_output.json', 'r') as f:
        res = json.load(f)

    return res

def compute_actual_frequencies(description, counts):
    total_ops = description['num_ops']
    result_description = {}
    frequencies = {}

    total = 0
    for op in counts:
        total += counts[op]
        frequencies[op] = counts[op]

    for op in frequencies:
        frequencies[op] = float(frequencies[op]) / float(total)

    # Build the total ops now
    for op in frequencies:
        result_description[op] = min(1, int(frequencies[op] * total_ops))

    return result_description


def merge_distributions(distributions):
    result = {}
    for distrib in distributions:
        for op in distrib:
            if op in result:
                result[op] += distrib[op]
            else:
                result[op] = distrib[op]

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cgra_specification', help='This should be a full spec so we can run the CGRA mapper --- the oly things that are relevant though are the rows and columns.  This ALSO needs to have a "num_ops" field to set the target resource utilization. ')
    parser.add_argument('benchmark_description')
    parser.add_argument('output_folder')
    args = parser.parse_args()

    # First, load all the benchmarks.
    benchmarks = load_benchmarks_from_json(args.benchmark_description)
    spec = load_specification_from_json(args.cgra_specification)

    distributions = []
    n = 0
    for b in benchmarks:
        if n > 2:
            break
        n += 1
        distributions.append(compute_distribution(args.cgra_specification, b))

    result_frequencies = merge_distributions(distributions)

    # Now, compute the actual number
    result_dict = compute_actual_frequencies(spec, result_frequencies)
    result_dict['row'] = spec['row']
    result_dict['column'] = spec['column']

    with open(args.output_folder, 'w') as f:
        json.dump(result_dict, f)