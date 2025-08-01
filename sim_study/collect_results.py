import csv 
import sys
from math import floor, log
from itertools import accumulate

def cumsum(a: list):
    return list(accumulate(a))

if __name__ == "__main__":
    
    max_ind = int(sys.argv[1])
    likelihood = sys.argv[2]
    print(f"max_ind = {max_ind}")
    
    missing_inds = []
    models = ['BH', 'GlobalCCP', 'NonGlobalCCP', 'NetCP']
    
    results_clean = dict(BH=[], NonGlobalCCP=[], GlobalCCP=[], NetCP=[])
    for i in range(max_ind):
        
        scenario_number = floor(i / (50 * (500 / 10)))
        dataset_number = floor((i % (50 * (500 / 10))) / (500 / 10))
        slice_number = int((i % (50 * (500 / 10))) % (500 / 10))
        start_number = 10 * slice_number + 1
        end_number = 10 * (slice_number + 1) + 1
        
        paths = [f"sim_results/{likelihood}/{model}_{scenario_number+1}_{dataset_number+1}_{start_number}-{end_number}.csv" for model in models]
        for model, path in zip(models, paths):
            try:
                with open(path, 'r') as f:
                    reader = csv.reader(f)
                    vals = [log(float(x[0])) for x in reader]
                    results_clean[model].extend(vals)
            except:
                missing_inds.append(i)
        
        if slice_number == 49:
            
            for k in results_clean.keys():
                results_clean[k] = cumsum(results_clean[k])
            min_ind = min(len(v) for v in results_clean.values()) - 1
            print(min_ind)
            print(q := [(k, v[min_ind]) for k, v in results_clean.items()])
            print(max(q, key=lambda x:x[1]))
            # write current results and reset
            for model in results_clean.keys():
                res_path = f"results_clean/{likelihood}/{model}_{scenario_number}_{dataset_number}.csv"
                print(f"Writing to: {res_path}")
                with open(res_path, 'w') as f:
                    writer = csv.writer(f)                    
                    writer.writerow(results_clean[model])
                results_clean[model] = []
    
    print(f"Number of missing indices: {len(missing_inds)}")
    # print(f"Missing indices: {missing_inds}")