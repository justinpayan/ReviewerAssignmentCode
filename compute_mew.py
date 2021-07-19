from itertools import permutations


def max_egalitarian(val_fns, goods_per_agent):
    n = len(val_fns)
    m = len(val_fns[1])

    goods = range(m)

    max_min_score = -1
    mew_alloc = None

    for p in permutations(goods):
        scores = []
        for i in range(n):
            score = 0
            for g in p[i*goods_per_agent: (i+1)*goods_per_agent]:
                score += val_fns[i][g]
            scores.append(score)
        if min(scores) > max_min_score:
            max_min_score = min(scores)
            mew_alloc = p

    return max_min_score, mew_alloc


def max_usw(val_fns, goods_per_agent):
    n = len(val_fns)
    m = len(val_fns[1])

    goods = range(m)

    max_usw = -1
    usw_alloc = None

    for p in permutations(goods):
        scores = []
        for i in range(n):
            score = 0
            for g in p[i*goods_per_agent: (i+1)*goods_per_agent]:
                score += val_fns[i][g]
            scores.append(score)
        if sum(scores) > max_usw:
            max_usw = sum(scores)
            usw_alloc = p

    return max_usw, usw_alloc


if __name__ == '__main__':
    # valn_fns = {0: [.8, .8, .7, .4, .4, .4],
    #             1: [.9, .7, .1, .9, .6, .5],
    #             2: [.2, .2, .3, .3, .4, .4]}
    valn_fns = {0: [.6, .5, .6, .6, .4, .4],
                1: [.9, .7, 0, .9, .6, .5],
                2: [.1, .1, .2, .2, .4, .4]}

    # Enumerate all solutions where each agent gets 2 goods. Return the one with the max min score.
    # Assume the number of goods is greater than or equal to the number of agents times the number of
    # goods expected by each agent.
    # print(max_egalitarian(valn_fns, 2))
    print(max_usw(valn_fns, 2))