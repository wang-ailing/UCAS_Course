import random
import time

def quicksort(array, start, end):
    if start < end:
        pi = partition(array, start, end)
        quicksort(array, start, pi - 1)
        quicksort(array, pi + 1, end)

def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] < pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1


array = [random.randint(1, 300000) for _ in range(200000)]
start_time = time.time()
quicksort(array, 0, len(array) - 1)
end_time = time.time()
print(f"执行时间：{end_time - start_time: .5f} 秒")
# print(array)