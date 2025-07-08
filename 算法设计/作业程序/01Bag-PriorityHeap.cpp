import random
import heapq

item_num = 20
bag_capacity = 500

item_prices = [random.randint(1, 100) for _ in range(item_num)]
item_capacities = [random.randint(1, 600) for _ in range(item_num)]

# 求 bag_capacity约束下price的最大值

