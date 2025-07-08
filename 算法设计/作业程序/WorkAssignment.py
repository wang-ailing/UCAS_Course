# 设有 n 件工作需要分配给 n 个人去完成。将工作 i 分配给第 j 个人
# 完成所需要的费用为 cij。试设计一个算法，为每一个人分配一件不同的工作，并使总费用
# 达到最小。

def dfs(current_cost, work_num, work_list, assigned_work, min_assigned_work):
    global min_cost
    if current_cost > min_cost: # 剪枝
        return
    
    if work_num == 0:
        if current_cost < min_cost:
            min_cost = current_cost
            min_assigned_work[:] = assigned_work[:]
        return
    
    for j in range(len(work_list[work_num-1])): # work_list[i][j] 表示第 i 件工作分配给第 j 个人的费用
        if assigned_work[j] == -1: # 如果第 j 个人还没有分配工作
            assigned_work[j] = work_num-1
            dfs (current_cost + work_list[work_num-1][j], work_num-1, work_list, assigned_work, min_assigned_work)
            assigned_work[j] = -1

    return min_cost

if __name__ == '__main__':
    n = 3
    work_list = [
        [1, 2, 3],
        [4, 5, 2],
        [3, 1, 4]
    ]
    min_assigned_work = [-1] * n
    min_cost = 0
    for i in range(n):
        min_cost += work_list[i][i]
    dfs(0, n, work_list,  [-1] * n, min_assigned_work) 
    print(min_assigned_work)
    for i in range(n):
        print(f"第 {i+1} 个人分配的工作为 {min_assigned_work[i]}, 费用为 {work_list[min_assigned_work[i]][i]}")
    print("总费用为", min_cost) # 输出结果为 4