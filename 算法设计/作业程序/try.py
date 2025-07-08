def multiply(a):
    n = len(a)
    m = [[0 for _ in range(n)] for _ in range(n)]
    for r in range(2, n+1):
        for i in range(n-r+1):
            j = r+i-1
            m[i][j] = m[i][i] + m[i+1][j] + a[i][0] *a[i][1] * a[j][1]
            print(m[i][j])
            for k in range(i+1, j):
                m[i][j] = min(m[i][j], m[i][k] + m[k+1][j] + a[i][0] *a[k][1] * a[j][1])
    for i in range(n):
        print(m[i])
    print(m[0][n-1])
    traceback(m, 0, n-1, a)

def traceback(m, i, j, a):
    if i == j:
        print(f"{i}", end="")
        return
    k = i
    while k < j:
        if m[i][k] + m[k+1][j] + a[i][0] *a[k][1] * a[j][1] == m[i][j]:
            print(f"(", end="")
            traceback(m, i, k, a)
            print(f")*(", end="")
            traceback(m, k+1, j, a)
            print(")", end="")
            return
        k += 1
n = int(input())
a = []
for i in range(n):
    a.append(list(map(int, input().split())))
multiply(a)