import numpy as np
import matplotlib.pyplot as plt

def plot_rotate_symmetry():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Points in complex plane
    author1 = np.array([-1, 0])
    author2 = np.array([1, 0])
    
    # Rotation angle (π radians = 180 degrees)
    theta = np.pi
    
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # Apply rotation to get expected positions
    a1_rotated = R @ author1
    a2_rotated = R @ author2

    # Plot authors
    ax.plot(*author1, 'bo', markersize=10, label='Author1')
    ax.plot(*author2, 'ro', markersize=10, label='Author2')
    ax.text(*(author1 + [-0.1, -0.15]), 'Author1', color='blue', fontsize=12)
    ax.text(*(author2 + [0.05, -0.15]), 'Author2', color='red', fontsize=12)

    # Plot rotated points
    ax.plot(*a1_rotated, 'bs', markersize=8, label='Author1 rotated')
    ax.plot(*a2_rotated, 'rs', markersize=8, label='Author2 rotated')
    ax.text(*(a1_rotated + [-0.1, 0.1]), 'Author1 rotated', color='blue', fontsize=11)
    ax.text(*(a2_rotated + [0.05, 0.1]), 'Author2 rotated', color='red', fontsize=11)

    # Draw rotation arrows
    # For Author1
    t = np.linspace(0, np.pi, 100)
    x1 = -1 + 0.5 * np.cos(t)
    y1 = 0.5 * np.sin(t)
    ax.plot(x1, y1, 'b--', alpha=0.5)
    ax.arrow(x1[-2], y1[-2], x1[-1]-x1[-2], y1[-1]-y1[-2],
             head_width=0.08, head_length=0.12, fc='blue', ec='blue', alpha=0.7)

    # For Author2
    x2 = 1 + 0.5 * np.cos(t)
    y2 = 0.5 * np.sin(t)
    ax.plot(x2, y2, 'r--', alpha=0.5)
    ax.arrow(x2[-2], y2[-2], x2[-1]-x2[-2], y2[-1]-y2[-2],
             head_width=0.08, head_length=0.12, fc='red', ec='red', alpha=0.7)

    # Add rotation angle labels
    ax.text(-0.7, 0.3, r'$\pi$', color='blue', fontsize=14)
    ax.text(0.7, 0.3, r'$\pi$', color='red', fontsize=14)

    # Algebraic inset
    ax.text(-2.3, 1.2, 'RotatE symmetry solution:', fontsize=12, fontweight='bold')
    ax.text(-2.3, 1.05, r'$h_1 \circ r = h_2$', fontsize=12)
    ax.text(-2.3, 0.9, r'$h_2 \circ r = h_1$', fontsize=12)
    ax.text(-2.3, 0.75, r'$\rightarrow$ $r = e^{i\pi}$ achieves symmetry', fontsize=12)

    ax.set_title("RotatE's Symmetry Solution: 180° Rotation", fontsize=14)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('doc/img/rotate_symmetry.pdf', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_rotate_symmetry() 