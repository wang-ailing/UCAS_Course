def min_time_schedule(tasks, machines):
    # 将任务按照所需时间排序
    sorted_tasks = sorted(enumerate(tasks), key=lambda x: x[1])
    
    # 初始化所有机器的完成时间为0
    machine_times = [0] * machines
    
    # 记录每个任务分配给哪个机器
    task_assignment = [-1] * len(tasks)
    
    # 为每个任务找到最早可用的机器
    for task, time in sorted_tasks:
        # 找到最早空闲的机器
        min_time_index = machine_times.index(min(machine_times))
        # 分配任务
        task_assignment[task] = min_time_index
        # 更新机器的完成时间
        machine_times[min_time_index] += time
    
    # 最大的机器完成时间即为所有任务完成的最早时间
    return max(machine_times), task_assignment

# 示例输入
n = 5  # 任务数量
k = 3  # 机器数量
tasks = [4, 5, 2, 1, 3]  # 每个任务所需的时间

# 调用函数
max_time, assignment = min_time_schedule(tasks, k)
print(f"所有任务完成的最早时间为: {max_time}")
print(f"任务分配给机器的方案为: {assignment}")