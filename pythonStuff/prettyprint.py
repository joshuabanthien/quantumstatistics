def printA(a):
    for row in a:
        for col in row:
            print("{:8.3f}".format(col), end=" ")
        print("")

def printL(a):
    for col in a:
        print("{:8.3f}".format(col), end=" ")
    print("")
