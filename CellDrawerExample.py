from random import randint
import random
import numpy as np

from PIL import Image


def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c


def complement(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))


# DNA "length" of "triplet code"
bound = 65536

# Initialize a sequence of 1000 random colors. This will be the "dna" code for the "cell", which will be expressed
# phenotypically on the screen.
dna_sequence = []
for i in range(0, bound):
    dna_sequence.append((randint(0, 256), randint(0, 256), randint(0, 256)))

# Set up image to be viewed
w, h = 256, 256
data = np.zeros((h, w, 3), dtype=np.uint8)

index = 0
# Import DNA sequence into two-dimensional array
for i in range(0, w):
    for j in range(0, h):
        data[i][j] = [dna_sequence[index][0], dna_sequence[index][1], dna_sequence[index][2]]
        index += 1

# Display image
img = Image.fromarray(data, 'RGB')
img.save('my.png')
img.show()

# Some questions of fitness: (1) Are colors complementary, (2) Are there objects? (i.e. pixels of at least nearly the
# the same color next to each other?

# Does this image have complementary colors? Formula = summation(complement - x_i) where i = every dna sequence

# Generate sample list since 65536 * 65536 iterations is too much
# IDK what sample size to pick, lets do 100?
rand_smpl = [dna_sequence[k] for k in sorted(random.sample(range(len(dna_sequence)), 100))]

complementary_cost = 0
for x in rand_smpl:
    compl = complement(x[0], x[1], x[2])
    for y in rand_smpl:
        if x == y:
            continue
        cost_0 = abs((y[0] - compl[0]) / 255)
        cost_1 = abs((y[1] - compl[1]) / 255)
        cost_2 = abs((y[2] - compl[2]) / 255)

        distance_from_complement = round(cost_0 + cost_1 + cost_2, 2)
        complementary_cost += distance_from_complement

# Complementary_cost should be high
print("compl_cost of imperf:", complementary_cost)


# Create the "perfect" cell 50/50 complementary colors. See what happens?
compl_sequence = []

for i in range(0, 32768):
    compl_sequence.append((255, 149, 104))
complement_tuple = complement(255, 149, 104)
for i in range(32768, bound):
    compl_sequence.append(complement_tuple)

# Set up image to be viewed
w, h = 256, 256
data = np.zeros((h, w, 3), dtype=np.uint8)

index = 0
# Import DNA sequence into two-dimensional array
for i in range(0, w):
    for j in range(0, h):
        data[i][j] = [compl_sequence[index][0], compl_sequence[index][1], compl_sequence[index][2]]
        index += 1

# Display image
perf_img = Image.fromarray(data, 'RGB')
perf_img.save('compl.png')
perf_img.show()

perf_rand_smpl = [compl_sequence[k] for k in sorted(random.sample(range(len(compl_sequence)), 100))]

complementary_cost = 0
for x in perf_rand_smpl:
    compl = complement(x[0], x[1], x[2])
    for y in perf_rand_smpl:
        if x == y:
            continue
        cost_0 = abs((y[0] - compl[0]) / 255)
        cost_1 = abs((y[1] - compl[1]) / 255)
        cost_2 = abs((y[2] - compl[2]) / 255)

        distance_from_complement = round(cost_0 + cost_1 + cost_2, 2)
        complementary_cost += distance_from_complement

# Complementary_cost should be low
print("compl_cost of perf:", complementary_cost)

# As can be observed, the cell with perfect compl. colors has the lowest cost. This will be a way to determine cell
# fitness

# Now to somehow detect "objects" in RGB pixels, which are really just pixels of nearly the same color adjacent to one
# another.
# Can't use random sample, because I want to look at adjacent pixels
# Maybe, instead of measuring object fitness, measure entropy cost. More entropy, higher cost. But I don't want to
# penalize creativity, I just want to detect objects.

# It's rough as it only takes into account horizontal adjacency, but it's better than nothing.
object_fitness = 0
for r in range(0, len(dna_sequence)):
    if r + 1 < bound:
        if ((dna_sequence[r+1][0] + 10) >= dna_sequence[r][0] >= (dna_sequence[r+1][0] - 10) and
                (dna_sequence[r + 1][1] + 10) >= dna_sequence[r][1] >= (dna_sequence[r + 1][1] - 10) and
                (dna_sequence[r + 1][2] + 10) >= dna_sequence[r][2] >= (dna_sequence[r + 1][2] - 10)):
            object_fitness += 1
        else:
            break
print("object_fitness of imperf:", object_fitness)

object_fitness = 0
for r in range(0, len(compl_sequence)):
    if r + 1 < bound:
        if ((compl_sequence[r+1][0] + 10) >= compl_sequence[r][0] >= (compl_sequence[r+1][0] - 10) and
                (compl_sequence[r + 1][1] + 10) >= compl_sequence[r][1] >= (compl_sequence[r + 1][1] - 10) and
                (compl_sequence[r + 1][2] + 10) >= compl_sequence[r][2] >= (compl_sequence[r + 1][2] - 10)):
            object_fitness += 1
        else:
            break
print("object_fitness of perf:", object_fitness)

