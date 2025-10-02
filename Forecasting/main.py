import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import mean_squared_error



def main():
    df = pd.read_csv('2021_2024.csv', sep=';')
    df['timestamp'] = pd.to_datetime(df['DateTime'], utc=True)
    df.set_index('timestamp', inplace=True)
    df.drop('DateTime', axis=1, inplace=True)

    data = df['Negative imbalance price'].values
    data = data.reshape(-1, 1)  # Maak er een 2D-array van

    # Schalen zodat het netwerk beter traint
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Deze functie converteert de tijdreeks naar input/output paren.
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            # a is een "window" van 'look_back' observaties (bv. de laatste 60 stappen)
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            # b is de observatie direct na het window (de voorspelling)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # We gebruiken de laatste 60 stappen (bv. 60 kwartieren = 15 uur) om de volgende te voorspellen.
    look_back = 60
    X, y = create_dataset(scaled_data, look_back)

    # Reshape de input naar het formaat dat LSTM verwacht: [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print(X)

    # Data set opsplitsen
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size

    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # Model bouwen met tenserflow
    model = keras.Sequential()
    # Voeg een LSTM laag toe met 50 neuronen.
    # input_shape=(look_back, 1) betekent dat elke input sample 60 timesteps heeft met 1 feature.
    model.add(layers.LSTM(50, input_shape=(look_back, 1)))
    # De output laag heeft 1 neuron, omdat we 1 waarde (de volgende prijs) willen voorspellen.
    model.add(layers.Dense(1))

    # Compileer het model. 'adam' is een efficiënte optimizer en 'mean_squared_error' is
    # een goede loss function voor regressieproblemen (het voorspellen van een getal).
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()

    # Model trainen
    print("\nStarten met training...")
    # Dit kan even duren, afhankelijk van je computer en de hoeveelheid data.
    history = model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)
    print("Training voltooid.")

    # Maak voorspellingen op de testdata
    test_predict = model.predict(X_test)

    # De voorspellingen zijn geschaald (tussen 0 en 1). We moeten ze terug transformeren
    # naar de originele schaal (euro's) om ze te kunnen interpreteren.
    test_predict = scaler.inverse_transform(test_predict)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Bereken de Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test_orig, test_predict))
    print(f'\nTest RMSE: {rmse:.2f} €/MWh')

    # --- STAP 8: VISUALISEER DE RESULTATEN ---



    # Plot de originele data en de voorspellingen
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_orig, color='blue', label='Werkelijke Prijs (Testset)')
    plt.plot(test_predict, color='orange', label='Voorspelde Prijs (Testset)')
    plt.title('Model Voorspellingen vs. Werkelijke Data')
    plt.xlabel('Tijdstappen (in de testset)')
    plt.ylabel('Prijs (€/MWh)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
