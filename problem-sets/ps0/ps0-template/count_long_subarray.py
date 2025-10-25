def count_long_subarray(A):
    """
    Input:  A     | Python Tuple of positive integers
    Output: count | number of longest increasing subarrays of A
    """

    # Current seq only consists of A[0]
    n = len(A)
    curr_seq_length = 1

    # It is also the max sequence length so far
    max_length = 1
    count = 1

    # For each of the remaining elements
    for i in range(1, n):
        prev, curr = A[i - 1], A[i]

        # If the curr > prev
        # it means the sequence is extended
        # So we increment curr_seq_length
        if curr > prev:
            curr_seq_length += 1
        # If the curr <= prev, it means that
        # it is the start of a new sequence
        # So we update the curr_seq_length to 1
        else:
            curr_seq_length = 1

        # If the curr_seq_length is greater than the max_length
        # it means we have found a new max length
        # so we update max_length and the count would be 1
        # as this is the first such sequence
        if curr_seq_length > max_length:
            max_length = curr_seq_length
            count = 1
        # If the curr_seq_length is equal to the max_length
        # it means we found another sequence with the max_length
        # so we increment the count
        elif curr_seq_length == max_length:
            count += 1

    return count
