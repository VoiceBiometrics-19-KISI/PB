import pandas as pd
import matplotlib.pyplot as plt

my_file = open('Results-China.txt', 'r')
data = my_file.read()

data = data.replace("[", "")
data = data.replace("]", "")
data = data.replace(" ", "")
data = data.replace(")", "")
data = data.replace("(", "")

data = data.split(",")

mylist = []
elemlist = []
counter = 0
for char in data:

    if len(char) == 1:
        elemlist.append(int(char))
    else:
        elemlist.append(float(char))

    counter += 1
    if counter==4:
        counter = 0
        mylist.append(elemlist)
        elemlist = []


df = pd.DataFrame(mylist, columns=["A", "B", "C", "D"])
print(df)

colors = ["paleturquoise", "aquamarine", "turquoise", "teal", "darkslategrey"]
df["A"].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)
plt.title("Percentage distribution of recognizing the same voice in 4 trials")
plt.ylabel("Count of recognized")
plt.show()

colors = ["dodgerblue", "deepskyblue", "lightskyblue", "powderblue", "aliceblue"]
df["C"].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)
plt.title("Percentage distribution of recognizing different voice in 4 trials")
plt.ylabel("Count of recognized")
plt.show()

df["B"].round(1).value_counts().plot(kind='bar')
plt.title("Distribution of average score for the same voice")
plt.ylabel("Count")
plt.xlabel("Score")
plt.show()

df["D"].round(1).value_counts().plot(kind='bar')
plt.title("Distribution of average score for different voice")
plt.ylabel("Count")
plt.xlabel("Score")
plt.show()

#My voice
#A - correctly
#B - average score (higher better)

#Not my voice
#C - correctly
#D - average score (lower better)