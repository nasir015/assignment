from flask import Flask, render_template, request
import numpy as np
import random

app = Flask(__name__)

# Functions for Sampling Techniques

def quadrat_sampling(grid, quadrat_size):
    rows, cols = grid.shape
    q_rows, q_cols = quadrat_size
    sampled_areas = []

    for i in range(0, rows, q_rows):
        for j in range(0, cols, q_cols):
            quadrat = grid[i:i + q_rows, j:j + q_cols]
            sampled_areas.append(np.sum(quadrat))

    return np.mean(sampled_areas)

def capture_recapture(M, C, R):
    return (M * C) / R

def transect_sampling(grid, transect_line):
    return np.sum(grid[transect_line, :])

def adaptive_sampling(grid, threshold):
    sampled = np.zeros_like(grid)
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] > threshold:
                sampled[i, j] = grid[i, j]
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for x, y in neighbors:
                    if 0 <= x < rows and 0 <= y < cols:
                        sampled[x, y] = grid[x, y]

    return sampled


def network_sampling(graph, start_node, depth):
    sampled = set()
    queue = [(start_node, 0)]

    while queue:
        node, d = queue.pop(0)
        if d <= depth:
            sampled.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in sampled:
                    queue.append((neighbor, d + 1))

    return sampled

def composite_sampling(samples):
    return np.mean(samples)

def rank_set_sampling(data, set_size, cycles):
    if not isinstance(data, list):  # Ensure data is a list
        data = list(data)
    if len(data) < set_size:  # Ensure the sample size is valid
        raise ValueError("Set size cannot be larger than the data length.")
    sampled_means = []
    for _ in range(cycles):
        sample = random.sample(data, set_size)
        ranked_sample = sorted(sample)
        sampled_means.append(ranked_sample[set_size // 2])  # Use the median of the sample
    return np.mean(sampled_means)

# Routes for Flask Application
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sampling/<technique>', methods=['GET', 'POST'])
def sampling(technique):
    result = None

    if request.method == 'POST':
        if technique == 'quadrat':
            grid = np.random.randint(0, 10, (10, 10))
            quadrat_size = (int(request.form['rows']), int(request.form['cols']))
            result = quadrat_sampling(grid, quadrat_size)
        elif technique == 'capture_recapture':
            M = int(request.form['M'])
            C = int(request.form['C'])
            R = int(request.form['R'])
            result = capture_recapture(M, C, R)
        elif technique == 'transect':
            grid = np.random.randint(0, 10, (10, 10))
            transect_line = int(request.form['transect_line'])
            result = transect_sampling(grid, transect_line)
        elif technique == 'adaptive':
            grid = np.random.randint(0, 20, (10, 10))  # Adjust range to include values > threshold
            threshold = int(request.form['threshold'])
            result = adaptive_sampling(grid, threshold)

        elif technique == 'network':
            graph = {
                'A': ['B', 'C'],
                'B': ['A', 'D'],
                'C': ['A', 'E'],
                'D': ['B'],
                'E': ['C']
            }
            start_node = request.form['start_node']
            depth = int(request.form['depth'])
            result = network_sampling(graph, start_node, depth)
        elif technique == 'composite':
            samples = [list(map(int, s.split(','))) for s in request.form.getlist('samples')]
            result = composite_sampling(samples)
        elif technique == 'rank_set':
            data = list(map(int, request.form['data'].split(',')))
            set_size = int(request.form['set_size'])
            cycles = int(request.form['cycles'])
            result = rank_set_sampling(data, set_size, cycles)

    return render_template('sampling.html', technique=technique, result=result)

if __name__ == '__main__':
    app.run(debug=True)
