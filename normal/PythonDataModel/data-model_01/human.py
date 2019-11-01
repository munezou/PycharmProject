class Human:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return "This is Human object"

    def __str__(self):
        return "my name is {0}. my old is {1}.".format(self.name, self.age)

    def __bytes__(self):
        return "{0}{1}".format(self.name, self.age).encode("UTF-8")

    def __hash__(self):
        return hash("{0}{1}".format(self.name, self.age))

    def __bool__(self):
        if self.name != "":
            return True
        else:
            return False

    def __eq__(self, other):
        if self.name == other.name and self.age == other.age:
            return True
        else:
            return False