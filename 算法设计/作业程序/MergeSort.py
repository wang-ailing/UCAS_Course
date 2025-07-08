

def merge(array, start, end):
    mid = start + end >> 1
    i = start
    j = mid + 1
    result = []
    while (i <= mid and j <= end):
        if (array[i] <= array[j]):
            result.append(array[i])
            i += 1
        else:
            result.append(array[j])
            j += 1
    while (i<=mid):
        result.append(array[i])
        i += 1
    while (j<=end):
        result.append(array[j])
        j += 1
    return result


def merge_sort(array, start, end):
    if (start >= end):
        return
    mid = start + end >> 1
    merge_sort(array, start, mid)
    merge_sort(array, mid+1, end)
    merged_array = merge(array, start, end)
    for i in range(start, end+1):
        array[i] = merged_array[i-start]

if __name__ == '__main__':
    array = [5,6,3,2,5,10,100,-4]
    # for i in array:
    #     print(i)
    merge_sort(array, 0, len(array)-1)

    print(array)