
import numpy as np

print("EXPERIMENT 6 - ASSOCIATIVE MEMORY")


# ---------------- AUTOASSOCIATIVE MEMORY ----------------

# Input patterns
inputs = np.array([
    [1,0,0,0],
    [1,1,0,0],
    [0,0,0,1],
    [0,0,1,1]
])

# Create weight matrix
# Each pattern is associated with itself
weights_auto = inputs.T @ inputs

# Test input
test_auto = np.array([1,1,0,0])

# Recall original pattern
output_auto = np.where(test_auto @ weights_auto > 0, 1, 0)

print("\nAutoassociative Output:", output_auto)


# ---------------- HETEROASSOCIATIVE MEMORY ----------------

# Corresponding target patterns
targets = np.array([
    [0,1],
    [0,1],
    [1,0],
    [1,0]
])

# Create weight matrix
# Associate input patterns with target patterns
weights_hetero = inputs.T @ targets

# Test input
test_hetero = np.array([1,1,0,0])

# Recall associated target pattern
output_hetero = np.where(test_hetero @ weights_hetero > 0, 1, 0)

print("Heteroassociative Output:", output_hetero)


# ---------------- BIDIRECTIONAL ASSOCIATIVE MEMORY ----------------

# Bipolar input patterns (E and F)
pattern_e = np.array([
    1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1
])

pattern_f = np.array([
    1,1,1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,1
])

# Corresponding target patterns
target_e = np.array([-1,1])
target_f = np.array([1,1])

# Store both input and target patterns
bam_inputs = np.array([pattern_e, pattern_f])
bam_targets = np.array([target_e, target_f])

# Create BAM weight matrix
weights_bam = bam_inputs.T @ bam_targets


# Forward direction: X -> Y
output_e = np.where(pattern_e @ weights_bam > 0, 1, -1)
output_f = np.where(pattern_f @ weights_bam > 0, 1, -1)

# Reverse direction: Y -> X
reverse_e = np.where(target_e @ weights_bam.T > 0, 1, -1)
reverse_f = np.where(target_f @ weights_bam.T > 0, 1, -1)


print("\nBAM E -> Target:", output_e)
print("BAM F -> Target:", output_f)

print("BAM Target E -> E:", reverse_e)
print("BAM Target F -> F:", reverse_f)
