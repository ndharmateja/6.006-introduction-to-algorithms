def count_long_subarray(A):
    """
    Input:  A     | Python Tuple of positive integers
    Output: count | number of longest increasing subarrays of A
    """
    n = len(A)
    curr_seq_length = 1
    curr_max = A[0]
    max_length = 1
    count = 1

    for i in range(1, n):
        num = A[i]
        if num > curr_max:
            curr_seq_length += 1
            curr_max = num
        else:
            curr_seq_length = 1
            curr_max = num

        if curr_seq_length > max_length:
            max_length = curr_seq_length
            count = 1
        elif curr_seq_length == max_length:
            count += 1

    return count
