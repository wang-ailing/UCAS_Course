import random
import heapq


class Node:
    def __init__(self, level, tag, cw, cp, ub, parent=None):
        self.level = level      # Node depth in the solution tree
        self.tag = tag          # Indicates if the item is included (1) or not (0)
        self.cw = cw            # Current weight in the knapsack
        self.cp = cp            # Current profit in the knapsack
        self.ub = ub            # Upper bound of the node
        self.parent = parent    # Pointer to the parent node

    def __lt__(self, other):
        # Compare nodes based on their upper bound for max-heap behavior
        return self.ub > other.ub

class Knap:
    def __init__(self, p, w, C, N):
        self.p = p              # Profit values
        self.w = w              # Weight values
        self.c = C              # Capacity of the knapsack
        self.n = N              # Number of items
        self.best = 0           # Best profit found
        self.best_node = None   # Node leading to the best solution
        self.answer = [0] * N   # Solution vector
        self.sort_items()       # Sort items by profit-to-weight ratio

    def sort_items(self):
        # Sort items by decreasing profit-to-weight ratio
        items = sorted(zip(self.p, self.w, range(self.n)), key=lambda x: x[0] / x[1], reverse=True)
        self.p, self.w, self.indices = zip(*items)

    def compute_upper_bound(self, k, cw, cp):
        remain_c = self.c - cw
        ub = cp
        while k < self.n and self.w[k] <= remain_c:
            remain_c -= self.w[k]
            ub += self.p[k]
            k += 1
        if k < self.n:
            ub += self.p[k] / self.w[k] * remain_c
        return ub

    def print_result(self):
        print("Maximum profit:", self.best)
        current = self.best_node
        while current:
            self.answer[self.indices[current.level]] = current.tag
            current = current.parent
        print("Solution vector:", self.answer)

    def solve(self):
        root = Node(level=-1, tag=0, cw=0, cp=0, ub=self.compute_upper_bound(0, 0, 0))
        heap = []
        heapq.heappush(heap, root)

        while heap:
            current = heapq.heappop(heap)
            if current.level == self.n - 1:
                continue

            next_level = current.level + 1

            # Left child: include the next item
            if current.cw + self.w[next_level] <= self.c:
                new_cw = current.cw + self.w[next_level]
                new_cp = current.cp + self.p[next_level]
                if new_cp > self.best:
                    self.best = new_cp
                    self.best_node = Node(level=next_level, tag=1, cw=new_cw, cp=new_cp, ub=new_cp, parent=current)
                new_ub = self.compute_upper_bound(next_level + 1, new_cw, new_cp)
                if new_ub > self.best:
                    heapq.heappush(heap, Node(level=next_level, tag=1, cw=new_cw, cp=new_cp, ub=new_ub, parent=current))

            # Right child: exclude the next item
            new_ub = self.compute_upper_bound(next_level + 1, current.cw, current.cp)
            if new_ub > self.best:
                heapq.heappush(heap, Node(level=next_level, tag=0, cw=current.cw, cp=current.cp, ub=new_ub, parent=current))

# Example usage
if __name__ == "__main__":
    item_num = 20
    bag_capacity = 500

    item_prices = [random.randint(1, 100) for _ in range(item_num)]
    item_capacities = [random.randint(1, 600) for _ in range(item_num)]


    print("Item prices:", item_prices)
    print("Item capacities:", item_capacities)
    knap = Knap(item_prices, item_capacities, bag_capacity, item_num)
    knap.solve()
    knap.print_result()
