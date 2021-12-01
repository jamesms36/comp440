import _PyPacwar
import numpy
import random


def geneticAlg():
    # Implement genetic algorithm here
    pop = Population(size=3)
    pop.print_pop()


def grade_score(rounds, c1, c2):
    score = 0
    if c2 == 0:
        if rounds < 100:
            score += 20.0 + (99 - rounds) / 99
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

    return score

class Scorer:
    def __init__(self):
        self.top_seqs = []

    def add_top_seq(self, seq):
        self.top_seqs.append(seq)

    def add_top_seqs(self, seqs):
        for seq in seqs:
            self.top_seqs.append(seq)

    def remove_oldest_seq(self):
        self.top_seqs.pop(0)

    def score_seq(self, seq):
        score = 0.0
        for top_seq in self.top_seqs:
            (rounds, c1, c2) = _PyPacwar.battle(seq.get_seq(), top_seq)
            score += grade_score(rounds, c1, c2)
        return score

    def battle_eachother(self):

        # Init score map
        score_map = {}
        for i in range(len(self.top_seqs)):
            score_map[i] = 0

        # Battles all the seqs against each other
        for i in range(len(self.top_seqs) - 1):
            for j in range(i + 1, len(self.top_seqs)):
                seq1 = self.top_seqs[i]
                seq2 = self.top_seqs[j]
                (rounds, c1, c2) = _PyPacwar.battle(seq1, seq2)
                score_map[i] += grade_score(rounds, c1, c2)
                score_map[j] += grade_score(rounds, c2, c1)

        for i in range(len(self.top_seqs)):
            score_map[i] = score_map[i] / len(self.top_seqs)


        ret_map = {}
        for key, val in score_map.items():
            ret_map[''.join(str(e) for e in self.top_seqs[key])] = val

        return ret_map

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

        prev_best_seq = Gene_Seq()
        prev_best_score = 0

        # Creates new generation from current population
        for i in range(0, len(temp_population), 2):
            j = 0
            seq1 = temp_population[i]
            seq2 = Gene_Seq()
            if i < len(temp_population) - 1:
                seq2 = temp_population[i + 1]

            temp_seqs = []

            # maintains the old best sequence to prevent losing ground
            test1 = seq1.get_seq_copy()
            test2 = seq2.get_seq_copy()
            score1 = test1.get_score()
            score2 = test2.get_score()
            if score1 == -1:
                score1 = self.scorer.score_seq(test1)
            if score2 == -1:
                score2 = self.scorer.score_seq(test2)

            if score1 > prev_best_score:
                prev_best_score = score1
                prev_best_seq = test1
            if score2 > prev_best_score:
                prev_best_score = score2
                prev_best_seq = test2

            # Generate possible sequence modifications
            while j < 10:
                temp_seq1 = seq1.get_seq_copy()

                # If seq1 is not the last in the list, crossover and mutate the lists
                if i < len(temp_population) - 1:
                    temp_seq2 = seq2.get_seq_copy()
                    temp_seq1.crossover(temp_seq2)
                    temp_seq2.mutate(random.randint(2, 5))
                    temp_seq2.set_score(-1)
                    temp_seqs.append(temp_seq2)

                temp_seq1.set_score(-1)
                temp_seq1.mutate(random.randint(2, 5))
                temp_seqs.append(temp_seq1)

                j += 1

            max_score = 0
            max = Gene_Seq()
            second_score = 0
            second = Gene_Seq()

            # Selects two best mutations
            for seq in temp_seqs:
                if seq.get_score() == -1:
                    score = self.scorer.score_seq(seq)
                    seq.set_score(score)
                else:
                    score = seq.get_score()

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

        new_population.pop(random.randint(0, len(new_population) - 1))
        new_population.append(prev_best_seq)

        self.population = new_population

    def get_avg_population_score(self):
        avg = 0
        for seq in self.population:
            if seq.get_score() == -1:
                score = self.scorer.score_seq(seq)
                seq.set_score(score)
            else:
                score = seq.get_score()

            avg += score

        avg /= len(self.population)
        return avg

    def get_max_seq(self):
        max_score = 0
        max_seq = []
        for seq in self.population:
            if seq.get_score() == -1:
                score = self.scorer.score_seq(seq)
                seq.set_score(score)
            else:
                score = seq.get_score()

            if score > max_score:
                max_score = score
                max_seq = seq
        return max_score, max_seq


class Gene_Seq:
    def __init__(self, seq_size=50):
        self.seq_size = seq_size
        self.seq = list(numpy.random.randint(low=0, high=4, size=self.seq_size))
        self.score = -1

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

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
        score = 0
        if c2 == 0:
            score += (500 - rounds) * 3 + 10000
        else:
            score += rounds + 10 * c1
        return score

    def crossover(self, partner):
        start = random.randint(0, self.seq_size - 1)
        end = start + random.randint(6, 15)  # self.seq_size / 2

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

# formats seq so that it can easily be copy and pasted
def format_seq_as_list(seq):
    separator = ""
    ret_str = "scorer.add_top_seq([" + separator.join(str(x) + "," for x in seq.print_seq()) + "])"
    x = ret_str.index("]")
    return ret_str[:x - 1] + ret_str[x:] + "       " + seq.print_seq()


def main():
    scorer = Scorer()
    scorer.add_top_seq([1] * 50)
    scorer.add_top_seq([3] * 50)
    scorer.add_top_seq(
        [0, 3, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 0, 0, 3, 3, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 3, 1, 3, 3,
         2, 1, 3, 0, 1, 3, 2, 3, 2, 1, 1, 3, 1])
    scorer.add_top_seq(
        [0, 3, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1,
                       1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 3, 1, 1, 3, 2, 1, 3, 1])
    scorer.add_top_seq([0,1,3,1,0,0,0,0,1,1,1,1,2,1,2,3,3,3,3,3,1,2,1,3,2,3,3,2,3,3,2,2,3,2,3,3,2,3,3,1,3,3,1,3,2,0,0,3,1,2])
    scorer.add_top_seq(
        [0, 3, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 3, 0, 0, 3, 0, 3, 3, 3, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 0, 2,
         1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1])

    all_time_scorer = Scorer()
    all_time_scorer.add_top_seqs(scorer.top_seqs)

    all_time_scorer.add_top_seq(
        [1,0,0,0,0,0,0,0,1,1,1,1,2,2,2,3,3,0,0,0,1,1,1,2,1,1,0,1,1,2,1,1,2,1,3,2,1,1,2,3,1,3,1,3,3,0,0,3,1,1])

    #  10310000111102213003333312333323312333300133213310
    # 01310000100000303103123323323123123333310313223313
    # 03100000101100003333311121121111122122111131132131

    # 01300000110100203333133323212121222311313323210312

    # 01310000111121233333121323323322323323313313200312
    # 03100100111123003033321121121111211021131131331111

    # maybe
    # 11020000113120223303312221211211220211230303301311
    # 10000000111122233000111211011211213211231313300311

    f = open("bestseqs.txt", "a")

    restarts = 0
    while restarts < 1:

        # Sets up new population
        population = Population()
        population.set_scorer(scorer)
        for i in range(0, 1000):
            seq = Gene_Seq()
            score = seq.vs_ones()

            if (score > 400):
                population.add_gene_seq(seq)

        # Tries to find a sequence that beats the best sequences
        tries = 1
        while tries < 80:
            population.reproduce()
            score, seq = population.get_max_seq()
            print(tries, population.get_avg_population_score(), score, seq.print_seq())
            tries += 1



        # writes new one to file
        f.write(format_seq_as_list(seq) + '\n')

        # Updates the scorer
        scorer.add_top_seq(seq.get_seq())
        scorer.remove_oldest_seq()

        # Updates the all time scorer
        all_time_scorer.add_top_seq((seq.get_seq()))

        restarts += 1

    f.write("\n\n\n\n~~~~~~~~ SCORING ~~~~~~~~~\n\n")
    score_map = all_time_scorer.battle_eachother()
    for i in sorted(score_map.items(), key=lambda x:x[1], reverse=True):
        print(i[1], i[0])
        f.write(i[1] + " " i[0] + '\n')

    f.close()


if __name__ == "__main__":
    main()