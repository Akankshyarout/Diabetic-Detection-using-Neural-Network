'''Either use tensorflow or keras not both, use pip uninstall xyz to uninstall any package'''

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x=dataset[:,0:8]
y=dataset[:,8]

#Training
model=Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#model.summary
#shows the neural network structure

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=4000, batch_size=10)
#can change the epoch for your training , epoch size increase accuracy will increase
_,accuracy=model.evaluate(x,y)
print('Accuracy: %.2f' % (accuracy*100))


model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Save model to disk")

