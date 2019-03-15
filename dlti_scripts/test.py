x = 2


def f(x=x):
    return x


print(f(3))
print(f())

res = 0
for i in range(5):
    res += i

print(res, i)

res2 = [j for j in range(10)]

# print(res2, j)

# with open("test.py") as f1, open(f1.read()) as g:
#     a = f1

# print(f)

b = 3

def f():
    try:
        a = 1/0
    except Exception as err:
        c = err
        c = b
        b = 2
        pass
    return c

print("f", f(), b)


def f2():
    global xyz
    xyz = 2
    return xyz


f2()
print(xyz)

x1 = 1


# def a1():
#     def a2():
#         nonlocal x1
#         x1 = 2
#     a2()
#     return x1


# print(a1())
i= 1

if i == 1:
    j3 = 2
else:
    k3 = j3

print(j3)
