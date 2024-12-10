import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from dataclasses import dataclass

@dataclass
class SimParams:
    # Base mutation rate per base pair per year
    u: float = 1.25e-8
    
    # Number of driver positions in each gene
    n_APC: int = 604
    n_TP53: int = 73
    n_KRAS: int = 20
    
    # Derived mutation rates per year
    r_APC: float = 604 * 1.25e-8
    r_TP53: float = 73 * 1.25e-8
    r_KRAS: float = 20 * 1.25e-8
    r_LOH: float = 1.36e-4  # Rate of loss of heterozygosity
    
    # Division rates per year
    b_APC: float = 0.2
    b_KRAS: float = 0.07
    b_BOTH: float = 0.27  # APC-/-, KRAS+ division rate
    
    # Time parameters
    dt: float = 1.0
    t_max: float = 80.0  # Full lifetime
    
    # Initial conditions
    N_crypts: int = 10**8  # Number of initial crypts


class State:
    """Represents the genetic state of a crypt."""
    def __init__(self, APC: int, TP53: int, KRAS: int):
        self.APC = APC
        self.TP53 = TP53
        self.KRAS = KRAS

    def __hash__(self):
        return hash((self.APC, self.TP53, self.KRAS))
    
    def __eq__(self, other):
        return (self.APC, self.TP53, self.KRAS) == (other.APC, other.TP53, other.KRAS)
    
    def is_malignant(self):
        """Check if state represents a malignant crypt."""
        return (self.APC >= 3 and self.TP53 >= 3 and self.KRAS == 1)


def get_transition_rate(source: State, target: State, params: SimParams) -> float:
    """Calculate transition rate between states."""
    if target.APC > source.APC:
        if source.APC == 0:
            return params.r_LOH if target.APC == 1 else params.r_APC
        elif source.APC == 1:
            return params.r_APC / 2
        elif source.APC == 2:
            return params.r_LOH / 2 if target.APC == 3 else params.r_APC / 2
    
    if target.TP53 > source.TP53:
        if source.TP53 == 0:
            return params.r_LOH if target.TP53 == 1 else params.r_TP53
        elif source.TP53 == 1:
            return params.r_TP53 / 2
        elif source.TP53 == 2:
            return params.r_LOH / 2 if target.TP53 == 3 else params.r_TP53 / 2
    
    if target.KRAS > source.KRAS:
        return params.r_KRAS
    
    return 0.0


def get_division_rate(state: State, params: SimParams) -> float:
    """Get division rate for a given state."""
    if state.APC >= 3:
        if state.KRAS == 1:
            return params.b_BOTH
        return params.b_APC
    elif state.KRAS == 1:
        return params.b_KRAS
    return 0.0


def get_neighbors(state: State):
    """Get all possible next states from current state."""
    neighbors = set()
    if state.APC == 0:
        neighbors.add(State(1, state.TP53, state.KRAS))
        neighbors.add(State(2, state.TP53, state.KRAS))
    elif state.APC == 1:
        neighbors.add(State(3, state.TP53, state.KRAS))
    elif state.APC == 2:
        neighbors.add(State(3, state.TP53, state.KRAS))
        neighbors.add(State(4, state.TP53, state.KRAS))
    
    if state.TP53 == 0:
        neighbors.add(State(state.APC, 1, state.KRAS))
        neighbors.add(State(state.APC, 2, state.KRAS))
    elif state.TP53 == 1:
        neighbors.add(State(state.APC, 3, state.KRAS))
    elif state.TP53 == 2:
        neighbors.add(State(state.APC, 3, state.KRAS))
        neighbors.add(State(state.APC, 4, state.KRAS))
    
    if state.KRAS == 0:
        neighbors.add(State(state.APC, state.TP53, 1))
    
    return neighbors


def tau_leaping_worker(batch_size: int, params: SimParams, batch_id: int):
    """Run a batch of tau-leaping simulations and save results. -yirui 11/19/2024"""
    time_points = np.arange(0, params.t_max + params.dt, params.dt)
    malignant_counts = np.zeros(len(time_points))
    
    for _ in range(batch_size):
        population = {State(0, 0, 0): params.N_crypts}
        had_malignant = False

        for t_idx, t in enumerate(time_points):
            if had_malignant:
                malignant_counts[t_idx:] += 1
                break

            new_events = {}
            for state, count in list(population.items()):
                if count == 0:
                    continue

                division_rate = get_division_rate(state, params)
                if division_rate > 0:
                    p = np.exp(-division_rate * params.dt)
                    n_divisions = np.random.negative_binomial(count, p)
                    if n_divisions > 0:
                        new_events[state] = new_events.get(state, 0) + n_divisions

                for neighbor in get_neighbors(state):
                    rate = get_transition_rate(state, neighbor, params)
                    if rate > 0:
                        p = 1 - np.exp(-rate * params.dt)
                        n_transitions = np.random.binomial(count, p)

                        if n_transitions > 0:
                            new_events[state] = new_events.get(state, 0) - n_transitions
                            new_events[neighbor] = new_events.get(neighbor, 0) + n_transitions

                            if neighbor.is_malignant():
                                had_malignant = True
                                break

            for state, delta in new_events.items():
                population[state] = population.get(state, 0) + delta
                if population[state] < 0:
                    population[state] = 0

            if had_malignant:
                malignant_counts[t_idx:] += 1
                break

    # Save results for this batch 
    #一定要把这一行改成你的path!!!!同时文件名称改成batch_YOURNAME{batch_id}
    #eg: np.savez_compressed(f"YOURPATH/batch_results/batch_alina_{batch_id}.npz", time_points=time_points, probabilities=malignant_counts / batch_size)   
    np.savez_compressed(f"batch_results/batch_{batch_id}.npz", time_points=time_points, probabilities=malignant_counts / batch_size)

    return malignant_counts


def run_large_simulation(params: SimParams, n_runs: int, batch_size: int, max_processes: int = 6): #如果大家的处理器核比较少把max_processes改小一点
    """Run a large simulation in batches with multiprocessing. -yirui 11/19/2024"""
    num_batches = n_runs // batch_size

    print(f"Running {n_runs} simulations in {num_batches} batches, using {max_processes} processes...")

    # Create a multiprocessing pool
    with Pool(processes=max_processes) as pool:
        pool.starmap(
            tau_leaping_worker,
            [(batch_size, params, batch_id) for batch_id in range(num_batches)]
        )


def combine_results(num_batches: int):
    """Combine results from all saved batches. -yirui 11/19/2024"""
    all_probabilities = []

    for batch_id in range(num_batches):
        data = np.load(f"batch_results/batch_{batch_id}.npz")
        all_probabilities.append(data['probabilities'])

    # Combine probabilities across batches
    combined_probabilities = np.mean(all_probabilities, axis=0)
    return combined_probabilities


def plot_results(time_points: np.ndarray, probabilities: np.ndarray):
    """Plot the probability of malignancy over time."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(time_points, probabilities, 'b-', label='Tau-leaping simulation')
    plt.xlabel('Age (years)')
    plt.ylabel('Probability of malignancy')
    plt.title('Probability of Colorectal Cancer Development')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs("batch_results", exist_ok=True)

    params = SimParams(
        dt=1.0,
        t_max=80.0,
        N_crypts=10**8
    )
    n_runs = 8 * 10**6 # 修改为指定的n_runs, 需要大于 10**4且是其整数倍例如 8*10**4
    batch_size = 10**4
    num_batches = n_runs // batch_size

    # Run simulation
    run_large_simulation(params, n_runs, batch_size, max_processes=4)


    #大家跑的时候一定要comment掉下面的代码!
    # Combine and plot results
    probabilities = combine_results(num_batches)
    time_points = np.arange(0, params.t_max + params.dt, params.dt)
    plot_results(time_points, probabilities)
