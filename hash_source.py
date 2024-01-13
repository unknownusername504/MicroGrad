# Creates a hash of all the source files in the micrograd package.
import hashlib
import os
import sys

# The hash of all the source files
def hash_source():
    # The hash object
    hasher = hashlib.sha256()

    # The source files
    source_files = []

    # Get the current working directory
    cwd = os.getcwd()

    # The micrograd package
    micrograd_package = os.path.join(cwd, "micrograd")

    # The source files in the micrograd package
    for root, dirs, files in os.walk(micrograd_package):
        for file in files:
            if file.endswith(".py"):
                source_files.append(os.path.join(root, file))

    # Sort the source files
    source_files.sort()

    # Hash all the source files
    for file in source_files:
        # Open the file
        with open(file, "rb") as f:
            # Read the file
            data = f.read()
            # Update the hash
            hasher.update(data)

    # Write the .hash file
    with open(os.path.join(cwd, ".hash"), "w") as f:
        # Write the hash
        f.write(hasher.hexdigest())

if __name__ == "__main__":
    hash_source()