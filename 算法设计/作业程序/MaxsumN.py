arr = [1, 2, -4, 4, -1, 6, -9, 8, 9]

dp = [0] * len(arr)

for i in range(len(arr)):
    dp[i] = max(dp[i-1] + arr[i], arr[i])

print(max(dp))