import heapq

class Node:
    def __init__(self, node, visited_cities, distance):
        self.node = node
        self.visited_cities = visited_cities
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance


def travel_priority(distance_matrix, start_city, n, min_distance):
    visited_cities = [start_city]
    Smallest_Travel_Heap = []
    heapq.heappush(Smallest_Travel_Heap, Node(start_city, visited_cities, 0))
    Smallest_Travel_Line_List = []

    while Smallest_Travel_Heap:
        current_node = heapq.heappop(Smallest_Travel_Heap)
        print(current_node.node, current_node.visited_cities, current_node.distance)
        if min_distance != -1 and current_node.distance > min_distance:
            continue
        if len(current_node.visited_cities) == n:
            if current_node.node < start_city:
                round_length = distance_matrix[current_node.node][start_city]
            else:
                round_length = distance_matrix[start_city][current_node.node]
            if min_distance == -1:
                min_distance = current_node.distance + round_length
                Smallest_Travel_Line_List.append(current_node.visited_cities + [start_city])
            elif current_node.distance + round_length == min_distance:
                Smallest_Travel_Line_List.append(current_node.visited_cities + [start_city])
            else:
                if current_node.distance + round_length < min_distance:
                    Smallest_Travel_Line_List = []
                    Smallest_Travel_Line_List.append(current_node.visited_cities + [start_city])
                min_distance = min(min_distance, current_node.distance + round_length)
            continue

        for i in range(n): # node i
            if i not in current_node.visited_cities:
                new_visited_cities = current_node.visited_cities + [i]
                if current_node.node < i:
                    new_distance = current_node.distance + distance_matrix[current_node.node][i]
                else:
                    new_distance = current_node.distance + distance_matrix[i][current_node.node]
                new_node = Node(i, new_visited_cities, new_distance)
                heapq.heappush(Smallest_Travel_Heap, new_node)

    return min_distance, Smallest_Travel_Line_List

if __name__ == '__main__':
    n = 5
    distance_matrix = [
        [0, 20, 30, 10, 11],
        [0, 0, 16, 4, 2],
        [0, 0, 0, 6, 7],
        [0, 0, 0, 0, 12],
        [0, 0, 0, 0, 0]
    ]

    min_distance, Smallest_Travel_Line_List = travel_priority(distance_matrix, 0, n, -1)
    print(min_distance, Smallest_Travel_Line_List) # 最小值为45，共有4条路径
#OUTPUT：45 [[0, 3, 2, 4, 1, 0], [0, 3, 2, 1, 4, 0], [0, 1, 4, 2, 3, 0], [0, 4, 1, 2, 3, 0]]