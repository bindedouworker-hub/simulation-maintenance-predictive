import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Définition du modèle LSTM pour la prédiction de RUL (Remaining Useful Life)
model = Sequential([
    # Couche LSTM avec 100 neurones, retournant des séquences pour empilement
    LSTM(100, return_sequences=True, input_shape=(50, 3)),
    Dropout(0.2), # Prévention du surapprentissage

    # Seconde couche LSTM pour capturer des patterns complexes
    LSTM(50, return_sequences=False),
    Dropout(0.2),

    # Couche Dense de sortie
    # 1 neurone de sortie : Estimation de la durée de vie restante (Régression)
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()