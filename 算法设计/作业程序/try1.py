def solve(nums, target):
    min_num = min(nums)
    if min_num < 0:
        nums = [num+(-1)*min_num for num in nums]
        target += (-1)*min_num
    i, j =0, 0
    count = 0
    while i<len(nums) and j<len(nums):
        num = sum(nums[j:i+1])
        if num == target:
            count += 1
            i += 1
            j = i
        elif num < target:
            i += 1
        else:
            j += 1
    return count
nums = list(map(int, input().split()))
target = int(input())
print(solve(nums, target))