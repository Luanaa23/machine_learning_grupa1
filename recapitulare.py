my_list = [8, 2, -3, 4, 2, -20.74, True, None]
print(len(my_list))
print(my_list[1:4])




class Student:
    count_stud=0
    def __init__(self, nume, an):
        self.nume=nume
        self.an=an
        Student.count_stud+=1
    def afisareNumarStudenti(self): #metoda statica
        print(self.count_stud)

s1=Student('Andrei',3)
s2=Student('Ana',1)
#Pentru a se apela metoda statica, se foloseste numele clasei.metoda
Student.afisareNumarStudenti()








