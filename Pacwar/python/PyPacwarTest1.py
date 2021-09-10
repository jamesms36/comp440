import _PyPacwar
import numpy
import random


def geneticAlg():
    # Implement genetic algorithm here
    print()

class Population:
    def __init__(self):
        self.genes = []
        print()

    def add_gene(self, gene):
        self.genes.append(gene)

    def select_next_gen(self):
        print()

    def get_genes(self):
        return self.genes

class Gene:
    def __init__(self, gene_size = 50):
        self.gene_size = gene_size
        self.gene = list(numpy.random.randint(low = 0,high=4,size=self.gene_size))

    def get_gene(self):
        return self.gene

    def print_gene(self):
        separator = ""
        return separator.join(str(x) for x in self.gene)

    def set_gene(self, gene):
        self.gene = gene

    def mutate(self, p):
        replace_list = numpy.random.rand(self.gene_size) < p
        print(replace_list)
        for i in range(self.gene_size):
            if replace_list[i]:
                self.gene[i] = numpy.random.randint(low = 0, high=4)

    def compete(self, competitor):
        return _PyPacwar.battle(self.get_gene(), competitor.get_gene())

    def vs_ones(self):
        ones = [1] * 50
        (rounds, c1, c2) = _PyPacwar.battle(self.get_gene(), ones)
        if c2 == 0:
            score = 1000/rounds + 10000
        else:
            score = rounds + 10*c1
        return score

    def crossover(self, partner, k):
        start = random.randint(0, len(self.gene[i]) / 2)
        for i in range(start, start + len(self.gene[i]) / 2:
            int temp = self.gene[i]
            self.gene[i] = partner[i]
            partner[i] = temp



def main():
    a = Gene()
    ones_gene = Gene()
    ones = [1] * 50
    #threes = [3] * 50
    a.set_gene(ones)
    ones_gene.set_gene(ones)

    population = Population()
    for i in range(0, 1000):
        gene = Gene()
        score = gene.vs_ones()
        print(score)
        if (score > 150):
            population.add_gene(gene)


if __name__ == "__main__":
    main()