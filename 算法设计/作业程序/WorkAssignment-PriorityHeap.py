import heapq

class Node:
    def __init__(self, path, machine_times, total_time, depth):
        self.path = path                # 当前任务的分配路径
        self.machine_times = machine_times  # 每台机器的当前运行时间
        self.total_time = total_time    # 当前策略下的最大运行时间
        self.depth = depth              # 当前处理的任务数

    def __lt__(self, other):
        return self.total_time < other.total_time


def best_dispatch(n, k, task_times):
    root = Node(path=[0] * n, machine_times=[0] * k, total_time=0, depth=0)
    heap = []  # 最小堆
    heapq.heappush(heap, root)
    
    best_time = float('inf')  # 当前最优解（初始为无穷大）
    best_result = None        # 最优解对应的节点

    while heap:
        current = heapq.heappop(heap)
        for i in range(k):
            new_path = current.path[:]
            new_path[current.depth] = i + 1
            
            new_machine_times = current.machine_times[:]
            new_machine_times[i] += task_times[current.depth]
            
            new_total_time = max(new_machine_times)  # 更新总时间
            
            # 创建新节点
            child_node = Node(
                path=new_path,
                machine_times=new_machine_times,
                total_time=new_total_time,
                depth=current.depth + 1
            )
            
            # 如果是叶节点，检查是否找到更优解
            if child_node.depth == n:
                if child_node.total_time < best_time:
                    best_time = child_node.total_time
                    best_result = child_node
            else:
                # 如果是中间节点，且有潜力成为更优解，加入堆中
                if child_node.total_time < best_time:
                    heapq.heappush(heap, child_node)

    # 返回最优结果
    return best_time, best_result.path


# 示例运行
if __name__ == "__main__":
    n = 5
    k = 3
    task_times = [4, 5, 2, 1, 3]

    best_time, best_path = best_dispatch(n, k, task_times)
    print("最短完成时间:", best_time)
    print("任务调度方案:", best_path)
