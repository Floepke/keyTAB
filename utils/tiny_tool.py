

def key_class_filter(key_class: str) -> list[int]:
    ''' 
        Return list of key numbers matching the given class 
        example 'abcdefg' to get all white key numbers (1-88)
        example 'CDFGA' to get all black key numbers (1-88)
    '''
    wanted = set(key_class)
    out: list[int] = []
    for key_num in range(1, 89):
        pc = (key_num - 1) % 12  # A0-based pitch class
        if pc in (0, 2, 3, 5, 7, 8, 10):  # naturals
            pc_char = {0: 'a', 2: 'b', 3: 'c', 5: 'd', 7: 'e', 8: 'f', 10: 'g'}[pc]
        else:  # sharps
            pc_char = {1: 'A', 4: 'C', 6: 'D', 9: 'F', 11: 'G'}[pc]
        if pc_char in wanted:
            out.append(key_num)
    return sorted(out)
