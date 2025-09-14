import numpy as np

OPERATORS = '+-*/(^scrl'

def get_priority(operator):
    if operator == '+' or operator == '-':
        return 1
    if operator == '*' or operator == '/':
        return 2
    if operator == '^':
        return 3
    if operator == 's' or operator == 'c' or operator == 'r' or operator == 'l':
        return 4
    return 0

def apply_operator(operator, a, b):
    if operator == '+':
        return a + b
    if operator == '-':
        return a - b
    if operator == '*':
        return a * b
    if operator == '/':
        if np.any(b == 0):
            raise ValueError("The function is not defined for all points of the training data.")
        return a / b
    if np.any((a < 0) & (b % 1 != 0)):
        raise ValueError("The function is not defined for all points of the training data.")
    return a**b

def apply_fun(fun, a):
    if fun == 's':
        return np.sin(a)
    if fun == 'c':
        return np.cos(a)
    if fun == 'r':
        if np.any(a < 0):
            raise ValueError("The function is not defined for all points of the training data.")
        return np.sqrt(a)
    if np.any(a <= 0):
        raise ValueError("The function is not defined for all points of the training data.")
    return np.log(a)

def get_num(fun, fun_rpn, index, prev, length, negative = False):
    if not negative and prev not in OPERATORS and prev != '!':
        return -1
    if fun[index].isdigit():
        num = ''
        count_dots = 0
        while index < length and (fun[index].isdigit() or fun[index] == '.' or fun[index] == ','):
            num += fun[index]
            if fun[index] == '.' or fun[index] == ',':
                count_dots += 1
            index += 1
        index -= 1
        if count_dots > 1 or fun[index] == '.' or fun[index] == '.':
            return -1
        fun_rpn.append(num)
    elif fun[index] == 'x':
        fun_rpn.append('x')
    return index

def get_operator(fun, fun_rpn, index, prev, length, stack, count_brackets):
    if fun[index] == '(':
        if prev not in OPERATORS and prev != '!':
            return -1, count_brackets
        stack.append(fun[index])
        count_brackets += 1
    elif fun[index] == ')':
        if len(stack) == 0 or prev in OPERATORS or prev == '!' or count_brackets == 0:
            return -1, count_brackets
        while stack[-1] != '(':
            fun_rpn.append(stack.pop())
        stack.pop()
        count_brackets -= 1
    elif fun[index] == '-' and prev in '!(':
        if index + 1 >= length:
            return -1, count_brackets
        index += 1
        while fun[index] == ' ':
            index += 1
            if index + 1 >= length:
                return -1, count_brackets
        fun_rpn.append('0')
        if fun[index].isdigit() or fun[index] == 'x':
            index = get_num(fun, fun_rpn, index, prev, length, True)
        else:
            return -1, count_brackets
        fun_rpn.append('-')
    else:
        if prev in OPERATORS or prev == '!':
            return -1, count_brackets
        if len(stack) > 0:
            while len(stack) > 0 and get_priority(stack[-1]) >= get_priority(fun[index]):
                fun_rpn.append(stack.pop())
        stack.append(fun[index])
    return index, count_brackets

def get_fun(fun, stack, index, prev, length):
    if prev not in OPERATORS and prev != '!':
        return -1, prev
    if fun[index] == 's':
        if index + 2 < length and ''.join(fun[index:index+3]) == 'sin':
            index += 2
            prev = 's'
        elif index + 3 < length and ''.join(fun[index:index+4]) == 'sqrt':
            index += 3
            prev = 'r'
        else:
            return -1, prev
    elif fun[index] == 'c':
        if index + 2 < length and ''.join(fun[index:index+3]) == 'cos':
            index += 2
        else:
            return -1, prev
        prev = 'c'
    elif fun[index] == 'l':
        if index + 1 < length and ''.join(fun[index:index+2]) == 'ln':
            index += 1
        else:
            return -1, prev
        prev = 'l'
    stack.append(prev)
    return index, prev

def tokenizer(fun, fun_rpn):
    prev = '!'
    count_brackets = 0
    length = len(fun)
    stack = []
    i = 0
    while i < length:
        if fun[i] == ' ':
            i += 1
            continue
        elif fun[i].isdigit() or fun[i] == 'x':
            i = get_num(fun, fun_rpn, i, prev, length)
            prev = fun[i]
        elif fun[i] in OPERATORS and fun[i] not in 'scl' or fun[i] == ')':
            i, count_brackets = get_operator(fun, fun_rpn, i, prev, length, stack, count_brackets)
            prev = fun[i]
        elif fun[i] in 'scl':
            i, prev = get_fun(fun, stack, i, prev, length)
        else:
            return False
        if i == -1:
            return False
        i += 1
    while len(stack) > 0:
        fun_rpn.append(stack.pop())
    if count_brackets > 0 or len(fun_rpn) == 0:
        return False
    return True

def value(fun_rpn, x):
    nums = []
    n = len(x)
    for el in fun_rpn:
        if el in OPERATORS and el not in 'scrl':
            b = nums.pop()
            a = nums.pop()
            nums.append(apply_operator(el, a, b))
        elif el in 'scrl':
            a = nums.pop()
            nums.append(apply_fun(el, a))
        elif el == 'x':
            nums.append(x)
        else:
            nums.append(np.full(n, float(el)))
    return nums[-1]
