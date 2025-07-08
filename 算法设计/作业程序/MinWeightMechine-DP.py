# 最小重量机器设计问题。设某一机器由 n 个部件组成，每一种部件都可以从 m 个
# 不同的供应商处购得。设 wij 是从供应商 j 处购得的部件 i 的重量，cij 是相应的价格。试设计
# 一个算法，给出总价格不超过 c 的最小重量机器设计。


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

sum_weights = sum(weights[i][j] for i in range(n) for j in range(m))
dp = [[sum_weights + 1] * (c + 1) for _ in range(n + 1)]
dp[0][0] = 0

for i in range(1, n+1):
    for j in range(c + 1):
        for k in range(m):
            if j < prices[i-1][k]:
                continue
            dp[i][j] = min(dp[i][j], dp[i - 1][j - prices[i-1][k]] + weights[i-1][k])


min_weight = sum_weights
for j in range(c+1):
    min_weight = min(min_weight, dp[n][j])

# print(dp)
print(min_weight)