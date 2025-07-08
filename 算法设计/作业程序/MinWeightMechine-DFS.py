
n = 4  # 部件数
m = 3  # 供应商数
c = 10  # 最大价格

prices = [
    [3, 4, 2],
    [2, 3, 4],
    [1, 2, 3],
    [5, 1, 7]
]

weights = [
    [1, 2, 3],
    [4, 1, 6],
    [7, 3, 9],
    [5, 5, 2]
]
min_weight = sum(weights[i][j] for i in range(n) for j in range(m))
total_cost = c

def dfs(index, current_weight, current_cost):
    global min_weight, total_cost

    if current_cost > total_cost:
        return
    
    if index == n:
        if current_weight < min_weight:
            min_weight = current_weight
        return
    
    for j in range(m):
        dfs(index+1, current_weight+weights[index][j], current_cost+prices[index][j])

dfs(0, 0, 0)
print(min_weight)