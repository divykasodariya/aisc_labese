
def mp_neuron(inputs, weights, threshold):
    total = 0

    for i in range(len(inputs)):
        total += inputs[i] * weights[i]

    if total >= threshold:
        return 1
    else:
        return 0


def AND(a, b):
    return mp_neuron([a, b], [1, 1], 2)


def OR(a, b):
    return mp_neuron([a, b], [1, 1], 1)


def NOT(a):
    return mp_neuron([a], [-1], 0)


def XOR(a, b):
    return int(AND(OR(a, b), NOT(AND(a, b))))


def half_adder(a, b):
    sum = XOR(a, b)
    carry = AND(a, b)

    return sum, carry


def full_adder(a, b, cin):
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)

    carry = c1 or c2

    return s2, int(carry)


if __name__ == "__main__":

    print("EXPERIMENT 3 - McCULLOCH-PITTS + ADDERS")

    print("AND:")
    for a in [0, 1]:
        for b in [0, 1]:
            print(a, b, "=>", AND(a, b))

    print("\nHalf Adder A B | Sum Carry")
    for a in [0, 1]:
        for b in [0, 1]:
            print(a, b, "|", half_adder(a, b))

    print("\nFull Adder A B Cin | Sum Carry")
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                print(a, b, c, "|", full_adder(a, b, c))
