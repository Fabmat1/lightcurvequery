import numpy as np
import argparse

def find_common_harmonics(P1, P2, tolerance):
    # Generate harmonics (adjust range as needed)
    multiples_P1 = np.arange(1, 100) * P1
    multiples_P2 = np.arange(1, 100) * P2

    # Find common harmonics within the tolerance
    common_harmonics = []
    for m1 in multiples_P1:
        for m2 in multiples_P2:
            if abs(m1 - m2) < tolerance:
                common_harmonics.append((m1 + m2) / 2)  # Average the two values for precision

    # Remove duplicates (if any) and sort
    common_harmonics = sorted(set(common_harmonics))
    return common_harmonics

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Find harmonics caused by two periods and their approximate common multiples."
    )
    parser.add_argument(
        "P1", type=float, help="First period (e.g., 0.33167)"
    )
    parser.add_argument(
        "P2", type=float, help="Second period (e.g., 0.548472)"
    )
    parser.add_argument(
        "-t", "--tolerance", type=float, default=0.001,
        help="Tolerance for matching harmonics (default: 0.001)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Find common harmonics
    common_harmonics = find_common_harmonics(args.P1, args.P2, args.tolerance)

    # Print results
    print("Common Harmonics:")
    for harmonic in common_harmonics:
        print(f"{harmonic:.6f}")

if __name__ == "__main__":
    main()
