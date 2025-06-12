import numpy as np
import matplotlib.pyplot as plt


def plot_transe_symmetry():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Points
    author1 = np.array([-1, 0])
    author2 = np.array([1, 0])
    r = np.array([0.8, 0.6])  # relation vector (same for both)

    # Translation results
    a1_plus_r = author1 + r
    a2_plus_r = author2 + r

    # Plot authors
    ax.plot(*author1, 'bo', markersize=10, label='Author1')
    ax.plot(*author2, 'ro', markersize=10, label='Author2')
    ax.text(*(author1 + [-0.1, -0.15]), 'Author1', color='blue', fontsize=12)
    ax.text(*(author2 + [0.05, -0.15]), 'Author2', color='red', fontsize=12)

    # Plot translation results
    ax.plot(*a1_plus_r, 'bs', markersize=8, label='Author1 + r')
    ax.plot(*a2_plus_r, 'rs', markersize=8, label='Author2 + r')
    ax.text(*(a1_plus_r + [-0.1, 0.1]), 'Author1 + r', color='blue', fontsize=11)
    ax.text(*(a2_plus_r + [0.05, 0.1]), 'Author2 + r', color='red', fontsize=11)

    # Draw r vectors
    ax.arrow(*author1, *r, head_width=0.08, head_length=0.12, fc='blue', ec='blue', length_includes_head=True, alpha=0.7)
    ax.arrow(*author2, *r, head_width=0.08, head_length=0.12, fc='red', ec='red', length_includes_head=True, alpha=0.7)
    ax.text(*(author1 + r/2 + [0.05, 0.08]), '$r$', color='blue', fontsize=13)
    ax.text(*(author2 + r/2 + [0.05, 0.08]), '$r$', color='red', fontsize=13)

    # Dashed lines: expected translation
    ax.plot([a1_plus_r[0], author2[0]], [a1_plus_r[1], author2[1]], 'k--', lw=1, alpha=0.7)
    ax.plot([a2_plus_r[0], author1[0]], [a2_plus_r[1], author1[1]], 'k--', lw=1, alpha=0.7)
    ax.text(*(a1_plus_r + author2)/2 + [0, 0.12], 'should match', color='k', fontsize=10, ha='center')
    ax.text(*(a2_plus_r + author1)/2 + [0, -0.12], 'should match', color='k', fontsize=10, ha='center')

    # Algebraic inset
    ax.text(-2.3, 1.2, 'TransE symmetry problem:', fontsize=12, fontweight='bold')
    ax.text(-2.3, 1.05, r'$h_1 + r \neq h_2$', fontsize=12)
    ax.text(-2.3, 0.9, r'$h_2 + r \neq h_1$', fontsize=12)
    ax.text(-2.3, 0.75, r'$\rightarrow$ No $r$ can satisfy both unless $h_1 = h_2$', fontsize=12)

    ax.set_title("TransE's Symmetry Limitation: CollaboratesWith Example", fontsize=14)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('doc/img/transe_symmetry.pdf', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_transe_symmetry() 