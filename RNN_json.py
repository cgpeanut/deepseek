import numpy as np
from tensorflow import keras
import json
from sklearn.preprocessing import MinMaxScaler

def load_and_predict(filepath):
    # Load data from JSON file
    with open(filepath, 'r') as f:
        dataset = json.load(f)
        
    # Convert the list of numbers to a numpy array and reshape it into (n_samples, 1)
    dataset = np.array(dataset).reshape(-1, 1)
    
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    # Split into train and test sets
    train_size = int(len(dataset) * 0.67)
    train, test = dataset[0:train_size], dataset[train_size:]
    
    # Convert an array of values into a dataset matrix
    def create_dataset(data):
        dataX, dataY = [], []
        for i in range(len(data)-1):
            dataX.append(data[i])
            dataY.append(data[i+1])
        return np.array(dataX), np.array(dataY)
    
    # Create train and test datasets
    look_back = 3
    trainX, trainY = create_dataset(train)
    testX, testY = create_dataset(test)
    
    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, look_back))
    testX = np.reshape(testX, (testX.shape[0], 1, look_back))
    
    # Create and fit the LSTM network
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(4, input_shape=(1, look_back)))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=20, batch_size=1)
    
    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    return trainPredict, testPredict