import csv 
from collections import defaultdict

def quantiles(data, quantile_list):
    if not data:
        raise ValueError("Data list is empty.")
    if not all(0 <= q <= 100 for q in quantile_list):
        raise ValueError("Quantiles must be between 0 and 100.")

    sorted_data = sorted(data)
    n = len(sorted_data)
    results = []

    for q in quantile_list:
        pos = q / 100 * (n - 1)
        lower_idx = int(pos)
        upper_idx = min(lower_idx + 1, n - 1)
        weight = pos - lower_idx
        interpolated = sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight
        results.append(interpolated)

    return results

if __name__ == "__main__":

	factors = dict(
		normal_mean = {
			0 : [None]*4,
			1 : [None]*4,
			2 : [None]*4,
			3 : [None]*4,
			4 : [None]*4,
		},
		ar_process = {
			0 : [None]*4,
			1 : [None]*4,
			2 : [None]*4,
			3 : [None]*4,
			4 : [None]*4,
		}
	)

	likelihoods = ['normal_mean', 'ar_process']
	scenarios = list(range(5))
	models = ['NetCP', 'BH', 'GlobalCCP', 'NonGlobalCCP']
	inds = list(range(50))
	for likelihood in likelihoods:
		for scenario in scenarios:
			model_bfs = defaultdict(list)
			for ind in inds:
				NetCP_path = f'results_clean/{likelihood}/NetCP_{scenario}_{ind}.csv'
				with open(NetCP_path, 'r') as file:
					reader=csv.reader(file, delimiter=',')
					NetCP_ev = float(next(iter(reader))[-1])
				
				for model in models:
					path = f'results_clean/{likelihood}/{model}_{scenario}_{ind}.csv' 
					with open(path, 'r') as file:
						reader = csv.reader(file, delimiter=',')
						model_ev = float(next(iter(reader))[-1])
					model_bfs[model].append(model_ev - NetCP_ev)
			for model, BFs in model_bfs.items():
				l, u = quantiles(sorted(BFs), [5, 95])
				
				factors[likelihood][scenario][models.index(model)] = (round(sum(BFs)/50, 1), round(l, 1), round(u, 1))
	print('Normal Mean:')
	for model in models:
		print(model +':' + ' '.join([str(factors['normal_mean'][scenario][models.index(model)]) for scenario in [0, 1, 2, 3, 4]]))
	
	print('')
	print('AR Process')
	for model in models:
		print(model +':' + ' '.join([str(factors['ar_process'][scenario][models.index(model)]) for scenario in [0, 1, 2, 3, 4]]))