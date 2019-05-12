import matplotlib.pyplot as plt

from src.CnnTrain import loss_list, num_epochs

print(list(loss_list))
plt.plot(list(range(1, 17 * num_epochs)), list(loss_list))
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()

