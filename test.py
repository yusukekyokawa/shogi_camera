def star(n):
    for i in range(n):
        print(' '*(n-i)+'*'*(2*i+1))


if __name__ == "__main__":
    n = int(input("How many stars column? >>>"))
    star(n)