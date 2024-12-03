import os
import re
from common_functions import LAMOST_PATTERNS, SDSS_PATTERNS, SOAR_PATTERNS, RVVD_PATH

for i, pattern in enumerate(LAMOST_PATTERNS):
    LAMOST_PATTERNS[i] = pattern.replace("mjd", r"\d+")

for i, pattern in enumerate(SDSS_PATTERNS):
    SDSS_PATTERNS[i] = pattern.replace("mjd", r"\d+")

for i, pattern in enumerate(SOAR_PATTERNS):
    SOAR_PATTERNS[i] = pattern.replace("mjd", r"\d+")

# Directory to search
directory = RVVD_PATH+"/spectra_processed"


def count_matching_files(directory, patterns):
    count = 0
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file matches any of the patterns
        for pattern in patterns:
            if re.fullmatch(pattern, filename):
                count += 1
                break  # If it matches one pattern, no need to check the others
    return count

# Count matching files for each set
lamost_count = count_matching_files(directory, LAMOST_PATTERNS)
sdss_count = count_matching_files(directory, SDSS_PATTERNS)
soar_count = count_matching_files(directory, SOAR_PATTERNS)

# Output the results
print(f"LAMOST matching files: {lamost_count}")
print(f"SDSS matching files: {sdss_count}")
print(f"SOAR matching files: {soar_count}")

print(lamost_count + sdss_count + soar_count)
