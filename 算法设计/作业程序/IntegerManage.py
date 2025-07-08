n = 6
b = 10
a = [1, 9, -3, 6, 5, 6]
c = [1, 9, 5, 11, 13, 6]

min_num = 0 # 因为xi可以全部取0
dp = [[min_num for _ in range(b+1)] for _ in range(n)]

for j in range(0, b+1):
    if (j >= a[0]) :
        dp[0][j] = c[0]
    if (j >= a[0] * 2) :
        dp[0][j] = max(dp[0][j], c[0] * 2)

for i in range(1, n):
    # print(i)
    for j in range(0, b+1):
        dp[i][j] = dp[i-1][j]
        index = j - a[i]
        if (index >= 0 and index < b+1) :
            dp[i][j] = max(dp[i][j], dp[i-1][j-a[i]] + c[i])
        index = j - a[i] * 2
        if (index >= 0 and index < b+1) :
            dp[i][j] = max(dp[i][j], dp[i-1][j-a[i]*2] + c[i] * 2)

max_value = 0
for j in range(0, b+1):
    max_value = max(max_value, dp[n-1][j])

print(max_value)