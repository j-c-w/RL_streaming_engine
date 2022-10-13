import json
import argparse
import os
import subprocess
import cgra_place

class Benchmark:
    def __init__(self, file, weight):
        self.file = file
        self.weight = weight

def load_benchmarks_from_json(jsonfile):
    benchmarks = []
    print("Loading benchmarks file", jsonfile)
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

def params_file_from_description(args, description, output_file):
    cgra = cgra_place.CGRA([], description['row'], description['column'], args)
    with open(output_file, 'w') as f:
        f.write(json.dumps(cgra.create_cgra_mapper_config()))

def compute_actual_frequencies(description, counts):
    total_ops = description['num_ops']
    result_description = {}
    frequencies = {}

    total = 0
    for op in counts:
        total += counts[op]
        if op in frequencies:
            frequencies[op] += counts[op]
        else:
            frequencies[op] = counts[op]

    print("Total is ", total)
    for op in frequencies:
        print("Number of op is", frequencies[op])
        frequencies[op] = float(frequencies[op]) / float(total)
        print("Fraction for ", op, "is", frequencies[op])

    print("Total ops should be ", total_ops)
    # Build the total ops now
    for op in frequencies:
        result_description[op] = max(1, int(frequencies[op] * total_ops))
        print("Generated", result_description[op], "of", op)

    return result_description

# LIst of skipped ops frmo DFGNode.isTransparentOp
def is_invisible(op):
    if op == "fptosi" or op == "ret" or op == "phi" or op == "bitcast" or op == "trunc" or op == "Constant" or op == "getelementptr" or op == "extractelement" or op == "insertelement" or op == "load" or op == "store":
        return True
    else:
        return False

def merge_distributions(distributions):
    result = {}
    for distrib in distributions:
        for op in distrib:
            if op.startswith('const') or op == 'Constant':
                # Skip these as they are 'free'.
                continue
            if is_invisible(op):
                # likewise
                continue
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

    cgra_description_file = args.output_folder + '/cgra_description.json'
    # First, load all the benchmarks.
    benchmarks = load_benchmarks_from_json(args.benchmark_description)
    spec = load_specification_from_json(args.cgra_specification)
    params_file_from_description(args, spec, cgra_description_file)
    cgra_spec = load_specification_from_json(cgra_description_file)
    cgra_spec['num_ops'] = spec['num_ops']

    distributions = []
    for b in benchmarks:
        distributions.append(compute_distribution(args.cgra_specification, b))

    result_frequencies = merge_distributions(distributions)

    # Now, compute the actual number
    result_dict = compute_actual_frequencies(cgra_spec, result_frequencies)
    result_dict['row'] = spec['row']
    result_dict['column'] = spec['column']
    result_dict['num_ops'] = spec['num_ops']

    with open(args.output_folder + '/cgra.json', 'w') as f:
        json.dump(result_dict, f)
