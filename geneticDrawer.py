from random import randint
import numpy as np
import random
import math
import os
from PIL import Image


class Cell:
    """
    A cell that has a DNA sequence of 65536 colors. Supposed to evolve to fit the human definition of "beauty". Or at
    least the Farukh definition of beauty, which to keep things simple will be measured by the following questions:
    1) Are colors complementary?
    2) Are there objects (i.e. pixels of nearly the same color adjacent to each other?)
    """

    def __init__(self, genetic_code=list()):
        """
        Create the artistic cell, with the option to pass genetic information.
        :param genetic_code: A list of 65536 tuples (256*256 screen). Each tuple is a triplet code that encodes the RGB
        of a pixel
        """
        # If no genetic_code was passed, set a random one
        if len(genetic_code) == 0:
            self.dna_sequence = list()
            self.random_sequence()
        else:
            # This should be the bound set in random_sequence
            self.bound = len(genetic_code)
            self.w, self.h = math.sqrt(len(genetic_code)), math.sqrt(len(genetic_code))
            self.dna_sequence = genetic_code

    def random_sequence(self):
        """
        Create a random DNA sequence for the cell. Supposed to be used when creating initial fes of the cell.
        :return: None
        """
        # DNA "length" of "triplet code"
        self.bound = 65536

        self.w, self.h = 256, 256

        # Initialize a sequence of 1000 random colors. This will be the "dna" code for the "cell", which will be
        # expressed phenotypically on the screen.
        for i in range(0, self.bound):
            self.dna_sequence.append((randint(0, 256), randint(0, 256), randint(0, 256)))

    def view_image(self, generation_number=0, cellnum=0, view=False):
        """
        View the image that this cell produces
        :param generation_number: Which generation the cell belongs to
        :param view: Do you want to see this image?
        :param cellnum: index of cell
        :return: None
        """
        # Change directory to "cellPics" to store images
        os.chdir("cellPics")

        # Set up image to be viewed
        data = np.zeros((int(self.h), int(self.w), 3), dtype=np.uint8)

        index = 0
        # Import DNA sequence into two-dimensional array
        for i in range(0, int(self.w)):
            for j in range(0, int(self.h)):
                data[i][j] = [self.dna_sequence[index][0], self.dna_sequence[index][1], self.dna_sequence[index][2]]
                index += 1

        img = Image.fromarray(data, 'RGB')
        img.save('gen{}_{}cell.png'.format(str(generation_number), str(cellnum)))

        # Display image if user wants to view
        if view:
            img.show()

        # Change directory to original
        os.chdir("..")

    def make_gamete(self):
        """
        Randomly get half of dna rows of information from dna sequence to simulate meiosis.
        :return: half of DNA information to be combined with other parent. Length 32768
        """
        # Randomly pull row indices from dna sequence
        random_rows = [list(range(int(self.h)))[k] for k in sorted(random.sample(range(int(self.h)), int(self.h/2)))]

        half_rand_smpl = []
        for row in random_rows:
            index = 0
            while index < self.h:
                half_rand_smpl.append((self.dna_sequence[index * (row - 1) + 1][0],
                                       self.dna_sequence[index * (row - 1) + 1][1],
                                       self.dna_sequence[index * (row - 1) + 1][2]))
                index += 1

        half_rand_smpl = [self.dna_sequence[k] for k in sorted(random.sample(range(len(self.dna_sequence)), int(self.bound/2)))]
        return half_rand_smpl

    def get_bound(self):
        return int(self.bound)

    def get_sequence(self):
        return self.dna_sequence


class ArtWorld:
    """
    Class to generate batches of cells and evolve them
    """
    def __init__(self, batch=10, generations=100, chance_mutation=25):
        # Create batch number of cells and randomly initialize them
        self.generation_num = 0
        self.batch = batch
        self.generations = generations
        self.chance_mutation = chance_mutation
        self.population = []

        for i in range(0, self.batch):
            self.population.append(Cell())

    def make_new_generation(self):
        """
        Select two random parents and breed them. If n is initial population, then 2n offspring are generated
        :return: None
        """
        for i in range(self.get_batch()):
            # Get two random parents from population
            parent_x_ind = randint(0, len(self.get_population()) - 1)
            parent_y_ind = randint(0, len(self.get_population()) - 1)

            # Ensure that parent is combined with a parent that is not itself
            while parent_x_ind == parent_y_ind:
                parent_y_ind = randint(0, len(self.get_population()) - 1)

            parent_x = self.population[parent_x_ind]
            parent_y = self.population[parent_y_ind]

            # Combine "gametes" of the two parents, which are two random rows
            self.population.append(Cell(parent_x.make_gamete() + parent_y.make_gamete()))
        self.generation_num += 1

    def remove_weak(self):
        """
        Measure fitness of each "cell". Fitness = Object_fitness - complementary_cost.
        If fitness is less than 0, I hard code it to be 0.
        :return: None
        """
        population_fitness = {}
        for p in self.get_population():
            fitness = self.measure_fitness(p)
            population_fitness[p] = fitness

        # Delete 10 least cells
        delete_num = 0
        for e in sorted(population_fitness.values(), reverse=True):
            del self.get_population()[int(e)]
            delete_num += 1
            # TODO: FIX GET_BATCH() ISSUE
            if delete_num == 5:
                break

    def start_simulation(self):
        for epoch in range(self.generations):
            i = 0
            for member in self.get_population():
                member.view_image(generation_number=self.generation_num, cellnum=i)
                i += 1

            self.make_new_generation()

            mutation_bool = randint(0, 100)
            if mutation_bool <= self.chance_mutation:
                mutated_ind = randint(self.get_batch(), len(self.get_population()) - 1)
                self.mutation(self.get_population()[mutated_ind])

            self.remove_weak()

    def mutation(self, c):
        """
        Mutation will consist of one triplet code of equal rgb being copied to 10 horizontal adjacent pixel.
        Or a complementary rgb code copied to a pixel that almost has the same complementary code. If no complementary
        code was found, create one randomly on the image.
        :param c: cell
        :return: None
        """
        rand_ind = randint(0, c.get_bound() + 1)
        flip = randint(0, 1)

        if flip:
            for m in range(0, 500):
                c.dna_sequence[(rand_ind + m) % c.get_bound()] = c.dna_sequence[rand_ind]
        else:
            compl_exists = False
            compl = self.complement(c.dna_sequence[rand_ind][0], c.dna_sequence[rand_ind][1], c.dna_sequence[rand_ind][2])

            index = 0
            for d in c.get_sequence():
                if (compl[0] - 10 <= d[0] <= compl[0] + 10) and (compl[1] - 10 <= d[1] <= compl[1] + 10) and (compl[2] - 10 <= d[2] <= compl[2] + 10):
                    compl_exists = True
                    for m in range(0, 500):
                        c.dna_sequence[(index + 1) % c.get_bound()] = compl
                index += 1
            if not compl_exists:
                c.dna_sequence[rand_ind] = compl

    def measure_fitness(self, c):
        """
        Measures the fitness of this cell.
        :param c: Cell
        :return: Tuple (Cost of pixels without complementary colors, Fitness of Pixels of the same color next to each other)
        """

        rand_smpl = [c.dna_sequence[k] for k in sorted(random.sample(range(len(c.dna_sequence)), 100))]

        complementary_cost = 0
        for x in rand_smpl:
            compl = self.complement(x[0], x[1], x[2])
            for y in rand_smpl:
                if x == y:
                    continue
                cost_0 = abs((y[0] - compl[0]) / 255)
                cost_1 = abs((y[1] - compl[1]) / 255)
                cost_2 = abs((y[2] - compl[2]) / 255)

                distance_from_complement = round(cost_0 + cost_1 + cost_2, 2)
                complementary_cost += distance_from_complement

        bound = c.get_bound()
        object_fitness = 0
        for r in range(0, len(c.dna_sequence)):
            if r + 1 < bound:
                if ((c.dna_sequence[r + 1][0] + 10) >= c.dna_sequence[r][0] >= (c.dna_sequence[r + 1][0] - 10) and
                        (c.dna_sequence[r + 1][1] + 10) >= c.dna_sequence[r][1] >= (c.dna_sequence[r + 1][1] - 10) and
                        (c.dna_sequence[r + 1][2] + 10) >= c.dna_sequence[r][2] >= (c.dna_sequence[r + 1][2] - 10)):
                    object_fitness += 1
                else:
                    break
        if object_fitness - complementary_cost < 0:
            return 0
        return object_fitness - complementary_cost

    # TODO: WHAT DOES THIS DO?
    def hilo(self, a, b, c):
        if c < b: b, c = c, b
        if b < a: a, b = b, a
        if c < b: b, c = c, b
        return a + c

    def complement(self, r, g, b):
        k = self.hilo(r, g, b)
        return tuple(k - u for u in (r, g, b))

    def get_sample(self):
        random_cell_i = randint(0, self.batch - 1)

        random_cell = self.population[random_cell_i]
        random_cell.view_image(self.generation_num)

    def get_population(self):
        return self.population

    def get_batch(self):
        return self.batch
