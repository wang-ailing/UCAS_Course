import random
import time


class ListNode: # 存储index
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def insert_to_list(list_head, array, index):
    current_node = ListNode(val = index)
    # print("=========================" )
    # print(index)
    # print(array[index])
    # print_list(list_head)
    if (list_head == None):
        return current_node
    current_point = list_head
    last_node = None

    while(current_point != None):
        compared_index = current_point.val
        # print(array[compared_index])
        if (array[compared_index] > array[index]):
            if (last_node == None):
                current_node.next = list_head
                return current_node
            last_node.next = current_node
            current_node.next = current_point
            return list_head
        else:
            last_node = current_point
            current_point = current_point.next
    
    last_node.next = current_node
    return list_head

def insertion_sort(array, start, end):
    list_head = None
    for i in range(start, end + 1):
        list_head = insert_to_list(list_head, array, i)
        # print_list(list_head)
    return list_head

def merge_l(node_a, node_b, array):

    new_head = None
    current_node = None
    i = node_a
    j = node_b
    while i and j:
        i_num = array[i.val]
        j_num = array[j.val]
        
        if i_num <= j_num:
            # tmp_i_next = i.next
            if (new_head == None):
                new_head = i
            else :
                current_node.next = i
            current_node = i
            i = i.next
        else:
            if (new_head == None):
                new_head = j
            else :
                current_node.next = j
            current_node = j
            j = j.next
    if i:
        current_node.next = i
    if j:
        current_node.next = j
    
    return new_head

def merge_sort_l(array, start, end):
    if end - start + 1 < 16:
        return insertion_sort(array, start, end)
    else:
        mid = (start + end) >> 1
        node_a = merge_sort_l(array, start, mid)
        node_b = merge_sort_l(array, mid + 1, end)
        return merge_l(node_a, node_b, array)

def print_list(head):
    sorted_arr = []
    current_node = head
    while current_node:
        sorted_arr.append(array[current_node.val])
        current_node = current_node.next

    print("Sorted Array:")
    print(sorted_arr)

if __name__ == '__main__':
    
    # array = [5, 6, 3, 2, 5]
    array = [random.randint(1, 300000) for _ in range(200000)]
    start_time = time.time()
    link_list = merge_sort_l(array, 0, len(array) - 1)
    end_time = time.time()
    print(f"执行时间：{end_time - start_time: .5f} 秒")
    # 
    # sorted_arr = []
    # current_node = link_list
    # while current_node:
    #     sorted_arr.append(array[current_node.val])
    #     current_node = current_node.next

    # print("Sorted Array:")
    # print(sorted_arr)