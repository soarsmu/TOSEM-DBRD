"""
Merge the results from multi files
"""

res_1, res_2, res_3, res_4 = 'ranking_result_mozilla-old_0.txt', \
'ranking_result_mozilla-old_1.txt', \
'ranking_result_mozilla-old_2.txt', \
'ranking_result_mozilla-old_3.txt'

def calculate_res(res_1, res_2, res_3, res_4):
    hitsPerK = dict((k, 0) for k in range(1, 11))
    
    with open(res_1, 'r') as f:
        lines1 = f.readlines()
        for line in lines1:
            rank = line.strip().split(',')[1].strip()
            if rank == 'inf':
                continue
            pos = int(rank)
            for i in range(1, 11):
                if i < pos:
                    continue
                hitsPerK[i] += 1
                
    with open(res_2, 'r') as f:
        lines2 = f.readlines()
        for line in lines2:
            rank = line.strip().split(',')[1].strip()
            if rank == 'inf':
                continue
            pos = int(rank)
            for i in range(1, 11):
                if i < pos:
                    continue
                hitsPerK[i] += 1
                
    with open(res_3, 'r') as f:
        lines3 = f.readlines()
        for line in lines3:
            rank = line.strip().split(',')[1].strip()
            if rank == 'inf':
                continue
            pos = int(rank)
            for i in range(1, 11):
                if i < pos:
                    continue
                hitsPerK[i] += 1
                
    with open(res_4, 'r') as f:
        lines4 = f.readlines()
        for line in lines4:
            rank = line.strip().split(',')[1].strip()
            if rank == 'inf':
                continue
            pos = int(rank)
            for i in range(1, 11):
                if i < pos:
                    continue
                hitsPerK[i] += 1
    
    recallRate = {}
    nDuplicate = len(lines1) + len(lines2) + len(lines3) + len(lines4)
    for k, hit in hitsPerK.items():
        rate = float(hit) / nDuplicate
        recallRate[k] = rate
    print(recallRate)
    for rr in recallRate:
        print(recallRate[rr])
    return recallRate

if __name__ == '__main__':
    calculate_res(res_1, res_2, res_3, res_4)