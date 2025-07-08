def max_crossing_sum(sum_arr, i, mid, j):
    left_sum = 0
    right_sum = 0
    for k in range(i, mid):
        if i == 0:
            if k == i:
                left_sum = sum_arr[k]
            else:
                left_sum = max(left_sum, sum_arr[k])
        else:
            if k == i:
                left_sum = sum_arr[k] - sum_arr[i-1]
            else:
                left_sum = max(left_sum, sum_arr[k] - sum_arr[i-1])
    
    for k in range(mid, j+1):
        if mid == 0:
            if k == mid:
                right_sum = sum_arr[k]
            else:
                right_sum = max(right_sum, sum_arr[k])
        else:
            if k == mid:
                right_sum = sum_arr[k] - sum_arr[mid-1]
            else:
                right_sum = max(right_sum, sum_arr[k] - sum_arr[mid-1])
    
    return left_sum + right_sum


def max_sum(arr, i, j, sum_arr):
    if i == j:
        return arr[i]
    else:
        mid = (i + j) // 2
        left_sum = max_sum(arr, i, mid, sum_arr)
        right_sum = max_sum(arr, mid+1, j, sum_arr)
        cross_sum = max_crossing_sum(sum_arr, i, mid, j)
        return max(left_sum, right_sum, cross_sum)



if __name__ == '__main__':
    arr = [1, 2, -4, 4, -1, 6, -9, 8, 9]
    sum_arr = [arr[0]]
    for i in range(1, len(arr)):
        sum_arr.append(arr[i] + sum_arr[i-1])
    max_sum_value = max_sum(arr, 0, len(arr)-1, sum_arr)
    print(max_sum_value)