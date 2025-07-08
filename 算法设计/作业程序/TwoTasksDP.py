n = 6
arr_a = [2, 5, 7, 10, 5, 2]
arr_b = [3, 8, 4, 11, 3, 4]
max_sum = max(sum(arr_a), sum(arr_b))

dp = [[max_sum for _ in range(max_sum + 1)] for _ in range(n + 1)]
# dp = [[max_sum for _ in range(n + 1)] for _ in range(max_sum)]

dp[0][0] = 0

for i in range(1, n + 1):
    for j in range(max_sum + 1):
        dp[i][j] = min(
            dp[i-1][j-arr_a[i-1]],
            dp[i-1][j] + arr_b[i-1]
        )

min_sum_value = max_sum

for i in range(n + 1):
    for j in range(max_sum + 1):
        if dp[i][j] != max_sum:
            print("dp["+str(i)+"]["+str(j)+"]="+str(dp[i][j]), end=' ')

    print()



for j in range(max_sum + 1):
    if dp[n][j] != max_sum:
        min_sum_value = min(min_sum_value, max(dp[n][j], j))
print("The minimum sum of two tasks is:", min_sum_value)
