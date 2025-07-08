# 假设有 n 个任务要由 k 个可并行工作的机器来完成。完成任务 i 需
# 要的时间为 ti。试设计一个算法找出完成这 n 个任务的最佳调度，使得完成全部任务的结束
# 时间最早。

n = 5
k = 3
task_time_list = [3, 2, 4, 1, 5]
task_plan_list = [-1] * n
k_time_list = [0] * k
min_total_time = sum(task_time_list)

def dfs(current_task, k_time_list, task_time_list, total_time, current_plan_list):
    global min_total_time

    # print(current_task)
    if total_time >= min_total_time: # 剪枝
        return

    if current_task == n:
        # print(total_time)
        if total_time < min_total_time:
            min_total_time = total_time
            task_plan_list[:] = current_plan_list[:]
        return
    
    current_task_time = task_time_list[current_task]
    for i in range(0, k):
        current_plan_list[current_task] = i
        k_time_list[i] += current_task_time
        dfs (
            current_task + 1,
            k_time_list,
            task_time_list,
            max(total_time, k_time_list[i]),
            current_plan_list,
        )
        k_time_list[i] -= current_task_time
        current_plan_list[current_task] = -1

dfs(0, k_time_list, task_time_list, 0, [-1] * n)
print(task_plan_list)
print(min_total_time)