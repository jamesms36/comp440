import _PyPacwar
import numpy
import random


def geneticAlg():
    # Implement genetic algorithm here
    pop = Population(size=3)
    pop.print_pop()


class Scorer:
    def __init__(self):
        self.top_seqs = []

    def add_top_seq(self, seq):
        self.top_seqs.append(seq)

    def score_seq(self, seq):
        score = 0
        for top_seq in self.top_seqs:
            (rounds, c1, c2) = _PyPacwar.battle(seq.get_seq(), top_seq)
            if c2 == 0:
                if rounds < 100:
                    score += 20
                elif rounds < 200:
                    score += 19
                elif rounds < 300:
                    score += 18
                elif rounds < 500:
                    score += 17
                # score += (500 - rounds) * 3 + 10000
            elif c1 == 0:
                if rounds < 100:
                    score += 0
                elif rounds < 200:
                    score += 1
                elif rounds < 300:
                    score += 2
                elif rounds < 500:
                    score += 3
            elif rounds == 500:
                if c1 > c2 * 10:
                    score += 13
                elif c1 > c2 * 3:
                    score += 12
                elif c1 > c2 * 1.5:
                    score += 11
                elif c2 > c1 * 10:
                    score += 7
                elif c2 > c1 * 3:
                    score += 8
                elif c2 > c1 * 1.5:
                    score += 9
                else:
                    score += 10
                # score += rounds + 10 * c1
        return score


class Population:
    def __init__(self, size=-1):
        if (size != -1):
            self.size = size
            self.population = [Gene_Seq(50) for x in range(size)]
        else:
            self.population = []

        self.scorer = Scorer()

    def get_population(self):
        return self.population

    def add_gene_seq(self, gene_seq):
        self.population.append(gene_seq)

    def set_scorer(self, scorer):
        self.scorer = scorer

    def print_population(self):
        for i in self.population:
            print(i.get_seq())

    def reproduce(self):
        temp_population = self.population.copy()
        new_population = []
        random.shuffle(temp_population)

        # Creates new generation from current population
        for i in range(0, len(temp_population), 2):
            j = 0
            seq1 = temp_population[i]
            seq2 = Gene_Seq()
            if i < len(temp_population) - 1:
                seq2 = temp_population[i + 1]

            temp_seqs = []

            # Generate possible sequence modifications
            while j < 10:
                temp_seq1 = seq1.get_seq_copy()

                # If seq1 is not the last in the list, crossover and mutate the lists
                if i < len(temp_population) - 1:
                    temp_seq2 = seq2.get_seq_copy()
                    temp_seq1.crossover(temp_seq2)
                    temp_seq2.mutate(3)
                    temp_seqs.append(temp_seq2)

                temp_seq1.mutate(3)
                temp_seqs.append(temp_seq1)

                j += 1

            max_score = 0
            max = Gene_Seq()
            second_score = 0
            second = Gene_Seq()

            # Selects two best mutations
            for seq in temp_seqs:
                score = self.scorer.score_seq(seq)

                if score > max_score:  # Found a new best sequence
                    second_score = max_score
                    second = max
                    max_score = score
                    max = seq
                elif score > second_score:  # Found a new second best sequence
                    second_score = score
                    second = seq

            new_population.append(max)
            new_population.append(second)

        self.population = new_population

    def get_avg_population_score(self):
        avg = 0
        for seq in self.population:
            avg += self.scorer.score_seq(seq)
        avg /= len(self.population)
        return avg

    def get_max_seq(self):
        max_score = 0
        max_seq = []
        for seq in self.population:
            score = self.scorer.score_seq(seq)
            if score > max_score:
                max_score = score
                max_seq = seq
        return max_score, max_seq


class Gene_Seq:
    def __init__(self, seq_size=50):
        self.seq_size = seq_size
        self.seq = list(numpy.random.randint(low=0, high=4, size=self.seq_size))

    def get_seq(self):
        return self.seq

    def get_seq_copy(self):
        new_seq = Gene_Seq(0)
        new_seq.set_seq(self.seq.copy())
        new_seq.set_seq_size(self.seq_size)
        return new_seq

    def print_seq(self):
        separator = ""
        return separator.join(str(x) for x in self.seq)

    def set_seq_size(self, seq_size):
        self.seq_size = seq_size

    def set_seq(self, seq):
        self.seq = seq

    def set_gene(self, gene_idx, gene):
        self.seq[gene_idx] = gene

    def compete(self, competitor):
        return _PyPacwar.battle(self.get_seq(), competitor.get_seq())

    def vs_ones(self):
        ones = [1] * 50
        (rounds, c1, c2) = _PyPacwar.battle(self.get_seq(), ones)
        if c2 == 0:
            score = 500 - rounds + 10000
        else:
            score = rounds + 10 * c1
        return score

    def crossover(self, partner):
        start = random.randint(0, self.seq_size - 1)
        end = start + 7  # self.seq_size / 2

        for i in range(self.seq_size):
            if ((i >= start and i < end) or ((end % self.seq_size) < end and i < end)):
                temp = self.seq[i]
                self.seq[i] = partner.get_seq()[i]
                partner.set_gene(i, temp)

    def mutate(self, max_replaceable):
        # Determines how many genes to mutate (must be less than or equal to all of them)
        replace_num = min(random.randint(0, max_replaceable), self.seq_size - 1)
        replace_list = []

        # determines the indeces to replace
        while len(replace_list) < replace_num:
            idx = random.randint(0, self.seq_size - 1)
            if idx not in replace_list:
                replace_list.append(idx)

        for idx in replace_list:
            self.seq[idx] = numpy.random.randint(low=0, high=4)


def main():
    scorer = Scorer()
    scorer.add_top_seq([1] * 50)
    scorer.add_top_seq([3] * 50)
    scorer.add_top_seq(
        [0, 3, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 0, 0, 3, 3, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 3, 1, 3, 3,
         2, 1, 3, 0, 1, 3, 2, 3, 2, 1, 1, 3, 1])
    scorer.add_top_seq(
        [0, 3, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 1, 0, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 3, 2,
         1, 1, 2, 1, 1, 2, 0, 2, 2, 1, 1, 3, 1])
    # 10111132101122010001111111111222101111122231002131

    #  10310000111102213003333312333323312333300133213310
    # 01310000100000303103123323323123123333310313223313


    population = Population()
    population.set_scorer(scorer)
    for i in range(0, 1000):
        seq = Gene_Seq()
        score = seq.vs_ones()

        if (score > 400):
            population.add_gene_seq(seq)

    tries = 1
    while tries < 50:
        population.reproduce()
        score, seq = population.get_max_seq()
        print(tries, population.get_avg_population_score(), score, seq.print_seq())
        tries += 1


if __name__ == "__main__":
    main()