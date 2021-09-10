import _PyPacwar
import numpy


def geneticAlg():
    # Implement genetic algorithm here
    pop = Population(size=3)
    pop.print_pop()


class Population:
    def __init__(self, size=100):
        self.size = size
        self.population = [Gene(50) for x in range(size)]

    def select_next_gen(self, size=100):
        print()

    def get_pop(self):
        return self.population

    def print_pop(self):
        for i in self.population:
            print(i.get_gene())

    def crossover(self):
        # crosses over random genes within the population
        print()

    def mutate(self):
        # mulates all the genes within the population
        for i in range(len(self.population)):
            self.population[i] = self.population[i].mutate()

    def reproduce(self):
        # selects the next generation, then crosses over, then mutates
        self.select_next_gen()
        self.crossover()
        self.mutate()
        print()


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
        print("")

def testStuff():
    a = Gene()
    ones_gene = Gene()
    ones = [1] * 50
    # threes = [3] * 50
    a.set_gene(ones)
    ones_gene.set_gene(ones)
    print("a vs ones score:", a.vs_ones())

    for i in range(1, 10000):
        gene = Gene()
        score = gene.vs_ones()
        print(score)
        if (score > 10000):
            print("decent gene found")
            print(gene.get_gene())
            print("decent score:, ", score)
            print("halting execution")
            break

    # (rounds, c1, c2) = a.compete(ones_gene)
    # print("Number of rounds:", rounds)
    # print("Ones PAC-mites remaining:", c1)
    # print("Threes PAC-mites remaining:", c2)


def main():
    geneticAlg()


if __name__ == "__main__":
    main()

