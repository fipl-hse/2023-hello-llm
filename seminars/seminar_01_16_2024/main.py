class B:
    def __str__(self):
        return "Instance of class B"

    def __add__(self, other): ...

def main():
    print('Hello!')
    b = B()
    c = C()
    d = b+c
    print (b)

if __name__ == '__main__':
    main()


class Samples:
    _raw_data: DataFrame

    def __str__(self): -> str
        return "Instance of class B"