"""
SimpleNeuralNetwork - Ett nätverk av perceptroner för multi-class klassificering.

Detta är ett "platt" nätverk utan gömda lager. Varje output-klass har sin egen
perceptron som tränas att identifiera just den klassen.
"""

import numpy as np
from perceptron import Perceptron, SigmoidPerceptron


class SimpleNeuralNetwork:
    """
    Ett enkelt nätverk med en perceptron per output-klass.

    Exempel: För pingvinarter (3 klasser) skapas 3 perceptroner:
    - En expert på Adelie
    - En expert på Chinstrap
    - En expert på Gentoo

    Nätverket använder SigmoidPerceptron för att få sannolikheter (0-1)
    istället för binära svar, vilket löser konflikter när flera
    perceptroner säger "ja".
    """

    def __init__(self, num_inputs, num_outputs, learning_rate=0.1, n_iterations=10):
        """
        Skapar ett nätverk med en perceptron per output-klass.

        Args:
            num_inputs: Antal input-features
            num_outputs: Antal klasser att klassificera
            learning_rate: Inlärningsfaktor för alla perceptroner
            n_iterations: Antal träningsiterationers
        """
        self.perceptrons = [
            SigmoidPerceptron(num_inputs, learning_rate, n_iterations)
            for _ in range(num_outputs)
        ]
        self.num_outputs = num_outputs

    def predict(self, inputs):
        """
        Frågar alla perceptroner och returnerar deras svar.

        Args:
            inputs: En lista med input-värden

        Returns:
            Lista med sannolikheter, en per klass
            T.ex. [0.92, 0.15, 0.03] betyder 92% säker på klass 0
        """

        # TODO: returnera en lista med prediktioner från varje perceptron
        # loopa alltså över self.perceptrons och anropa predict för varje
        # returnera desras resultat som en lista
        return []

    def train(self, inputs, targets, epochs=None):
        """
        Tränar varje perceptron mot sin motsvarande kolumn i targets.

        Args:
            inputs: Lista av träningsexempel
            targets: One-hot encoded facit, t.ex. [[1,0,0], [0,1,0], ...]
            epochs: Antal epoker (använder n_iterations om None)
        """
        for i, perceptron in enumerate(self.perceptrons):
            # Skapa en lista med facit BARA för denna klass
            # T.ex. för Adelie-perceptronen: [1, 0, 0, 1, 1, 0, ...]
            specific_targets = [target[i] for target in targets]
            # TODO: träna perceptronen med inputs och sina facit


def compute_accuracy(network, inputs, targets):
    """
    Beräknar noggrannhet för ett nätverk med one-hot encoded targets.

    Args:
        network: Ett objekt med en .predict(input) metod
        inputs: Lista eller array med input-data
        targets: One-hot encoded facit

    Returns:
        Noggrannhet som ett tal mellan 0 och 1
    """
    correct = 0
    total = len(inputs)

    for i, x in enumerate(inputs):
        prediction = network.predict(x)
        target = targets[i]

        # Hitta indexet med högst sannolikhet
        predicted_class = np.argmax(prediction)
        # Hitta indexet där 1:an sitter i one-hot target
        actual_class = np.argmax(target)

        if predicted_class == actual_class:
            correct += 1

    return correct / total


# --- Exempel på användning ---
if __name__ == "__main__":
    import seaborn as sns
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    print("Laddar pingvin-data...")
    scaler = MinMaxScaler()

    penguins = sns.load_dataset("penguins").dropna()
    X = penguins[["flipper_length_mm", "bill_length_mm"]].values
    X = scaler.fit_transform(X)

    # One-Hot Encoding av arterna
    y_onehot = pd.get_dummies(penguins["species"], dtype=int).values
    species_names = ["Adelie", "Chinstrap", "Gentoo"]

    print(f"Träningsdata: {len(X)} pingviner")
    print(f"Exempel på one-hot facit: {y_onehot[0]} ({species_names[np.argmax(y_onehot[0])]})")

    # Skapa och träna nätverket
    network = SimpleNeuralNetwork(num_inputs=2, num_outputs=3, learning_rate=0.1, n_iterations=100)
    network.train(X, y_onehot)

    # Utvärdera
    accuracy = compute_accuracy(network, X, y_onehot)
    print(f"\nNoggrannhet: {accuracy * 100:.2f}%")

    # Visa några exempel
    print("\n--- Exempel på prediktioner ---")
    for i in range(5):
        probs = network.predict(X[i])
        predicted = species_names[np.argmax(probs)]
        actual = species_names[np.argmax(y_onehot[i])]
        print(f"Prediktion: {predicted}, Facit: {actual}, Sannolikheter: {[f'{p:.2f}' for p in probs]}")
