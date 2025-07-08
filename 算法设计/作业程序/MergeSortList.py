class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge(l1, l2):
    dummy = ListNode(0)
    current = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    current.next = l1 if not l2 else l2
    return dummy.next

def merge_sort(head):
    if not head or not head.next:
        return head
    
    # Find the middle of the linked list
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Split the list into two halves
    second_half = slow.next
    slow.next = None
    
    # Recursively sort the two halves
    l1 = merge_sort(head)
    l2 = merge_sort(second_half)
    
    # Merge the sorted halves
    return merge(l1, l2)

def print_list(node):
    while node:
        print(node.val, end=" ")
        node = node.next
    print()

# Helper function to create a linked list from a python list
def create_linked_list(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for value in arr[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

# Helper function to convert linked list to python list
def linked_list_to_array(head):
    array = []
    while head:
        array.append(head.val)
        head = head.next
    return array

# Example usage
array = [5, 6, 3, 2, 5]
linked_list = create_linked_list(array)
print("Original Linked List:")
print_list(linked_list)

sorted_linked_list = merge_sort(linked_list)
print("Sorted Linked List:")
print_list(sorted_linked_list)

# Convert to array to verify the result
sorted_array = linked_list_to_array(sorted_linked_list)
print("Sorted Array:")
print(sorted_array)