import numpy as np
import matplotlib.pyplot as plt

# Constants from the paper
def get_parameters():
    """Return the parameters used in the model"""
    # Number of crypts
    N = 1e8
    
    # Base mutation rate per base pair per year
    u = 1.25e-8
    
    # Number of driver positions in each gene
    n_APC = 604
    n_TP53 = 73
    n_KRAS = 20
    
    # Gene-specific mutation rates per year
    r_APC = n_APC * u
    r_TP53 = n_TP53 * u
    r_KRAS = n_KRAS * u
    
    # Rate of LOH (loss of heterozygosity)
    r_LOH = 1.36e-4
    
    # Division rates per year
    b1 = 0.2  # APC-/- crypts
    b2 = 0.07  # KRAS+ crypts
    b12 = b1 + b2  # Double mutant APC-/-KRAS+ crypts
    
    # Correction factors for increased fixation
    c1 = 5.88  # APC correction
    c2 = 3.6   # KRAS correction
    c = c1 * c2  # Total correction
    
    return N, r_APC, r_TP53, r_KRAS, r_LOH, b1, b2, b12, c

def neutral_probability(t):
    """Calculate probability when all mutations are neutral (Equation 1)"""
    N, r_APC, r_TP53, r_KRAS, r_LOH, _, _, _, _ = get_parameters()
    
    return N * r_APC * r_TP53 * r_KRAS * r_LOH**2 * t**5 / 4

def apc_advantage_probability(t):
    """Calculate probability when only APC provides advantage (Equation 2)"""
    N, r_APC, r_TP53, r_KRAS, r_LOH, b1, _, _, c1 = get_parameters()
    
    return (3 * N * r_APC * r_TP53 * r_KRAS * r_LOH**2 * np.exp(b1 * t) * t**2) / (2 * b1**3) * c1

def apc_kras_advantage_probability(t):
    """Calculate probability when both APC and KRAS provide advantage (Equation 3)"""
    N, r_APC, r_TP53, r_KRAS, r_LOH, b1, b2, b12, c = get_parameters()
    
    term1 = 1 / (b12**3 * (b12 - b1))
    term2 = 1 / (b12**3 * (b12 - b2))
    term3 = 1 / (b12**2 * (b12 - b2)**2)
    
    return c * N * r_APC * r_TP53 * r_KRAS * r_LOH**2 * t * np.exp(b12 * t) * (term1 + term2 + term3)

def plot_probabilities():
    """Create plot similar to Figure 2 from the paper"""
    # Set the figure style
    plt.style.use('default')
    
    # Create figure with a white background
    plt.figure(figsize=(8, 10), facecolor='white')
    
    # Create time points (0 to 80 years)
    t = np.linspace(0, 80, 1000)
    
    # Calculate probabilities
    p_neutral = [neutral_probability(ti) for ti in t]
    p_apc = [apc_advantage_probability(ti) for ti in t]
    p_both = [apc_kras_advantage_probability(ti) for ti in t]
    
    # Create the plot
    plt.semilogy(t, p_neutral, 'b-', label='All mutations neutral', linewidth=2)
    plt.semilogy(t, p_apc, 'g-', label='APC advantage only', linewidth=2)
    plt.semilogy(t, p_both, 'r-', label='APC and KRAS advantage', linewidth=2)
    
    # Add reported lifetime risk line
    plt.axhline(y=0.006, color='k', linestyle='--', label='Reported lifetime risk', linewidth=2)
    
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Probability of malignant crypt', fontsize=12)
    plt.title('Probability of CRC Development Under Different Models', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    plt.ylim(1e-12, 1e-1)
    plt.xlim(20, 80)
    
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt

# Generate and show the plot
plot_probabilities()
plt.show()