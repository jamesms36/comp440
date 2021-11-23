import random

q = {
    }

def next_state(s, q):
    r = random.random()
    p = 0
    for i in q[s]:
        p = p+q[s][i]
        if r < p:
            return i

def run_gibbs(size):
    s = random.choice(list(q.keys()))
    seq = [s]
    for i in range(size):
        next = next_state(s, q)
        seq.append(next)
        s = next

    print("Size: ", size, "(1,0): ", seq.count((1, 0)) / (size * 1.0), "(0,1): ", seq.count((0, 1)) / (size * 1.0),  "(1,1): ", seq.count((1, 1)) / (size * 1.0), "(0,0): ", seq.count((0, 0)) / (size * 1.0), )

run_gibbs(1000)
run_gibbs(5000)
run_gibbs(10000)