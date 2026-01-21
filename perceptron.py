"""
Perceptron - En neuron som kan lära sig.

Perceptronen utökar Neuron med en inlärningsalgoritm (fit-metoden)
som justerar vikter och bias baserat på träningsdata.
"""

import random
import math
from neuron import Neuron


class Perceptron(Neuron):
    """
    En Perceptron som kan lära sig från träningsdata.

    Utökar Neuron med:
    - Slumpmässig initialisering av vikter/bias
    - fit()-metod för träning
    - Konfigurerbar inlärningsfaktor och antal iterationer
    """

    def __init__(self, num_inputs, learning_rate=0.1, n_iterations=10):
        """
        Skapar en Perceptron med slumpmässiga startvärden.

        Args:
            num_inputs: Antal inputs som perceptronen tar emot
            learning_rate: Inlärningsfaktor (alpha), styr stegstorleken
            n_iterations: Antal gånger att gå igenom träningsdatan
        """
        # Slumpmässiga vikter mellan -0.5 och 0.5
        weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        # Slumpmässig bias
        bias = random.uniform(-0.5, 0.5)

        # Anropa förälderns __init__
        super().__init__(weights, bias)

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, inputs, targets):
        """
        Tränar Perceptronen genom att iterera över träningsdatan.

        Använder perceptron-inlärningsregeln:
        - ny_vikt_x = gammal_vikt + α * fel * input_x
        - ny_bias = gammal_bias + α * fel

        Args:
            inputs: Lista av träningsexempel (varje exempel är en lista av inputs)
            targets: Lista av korrekta svar (0 eller 1)
        """
        for iteration in range(self.n_iterations):
            # Gå igenom varje exempel och dess facit
            for input_example, target in zip(inputs, targets):

                # TODO STEG 1: Gissa (använd predict från Neuron, finns i variabeln self)
                # Du behöver använda ditt input_example här

                # TODO STEG 2: Beräkna Felet
                # error = facit - gissning
                # variabeln target är facit, gissning är svaret från din prediction
                # Kan vara: 0 (rätt), 1 (gissade 0 men skulle vara 1), -1 (gissade 1 men skulle vara 0)
                

                # TODO STEG 3: Justera Parametrar (endast om gissningen var fel)

                # Justera bias: ny_bias = gammal_bias + α * fel
                # TODO

                # Justera varje vikt: ny_vikt_x = gammal_vikt + α * fel * input
                for index in range(len(self.weights)):
                    # TODO: uppdatera vikt_x med formeln ovan
                    # index används för att komma åt rätt vikt och input
                    # du behöver använda self.weights[index] och input_example[index]
                    # Kom ihåg, respektive vikt representerar hur mycket dess motsvarande
                    # input påverkar den slutgiltiga gissningen
                    # Därför för att justera en vikt på ett specifikt index
                    # i vår lista med vikter, behöver vi kunna komma åt
                    # både vikten och dess motsvarande input
                    self.weights[index] += 0  # TODO: uppdatera denna rad



# --- Exempel på användning ---
if __name__ == "__main__":
    # Test på "fest-scenariot"
    # Mål: Lär dig att gå på fest om du inte har ett prov

    inputs = [
        [0, 0],  # Inga vänner, inget prov
        [0, 1],  # Inga vänner, prov
        [1, 0],  # Vänner, inget prov
        [1, 1],  # Vänner, prov
    ]
    targets = [1, 0, 1, 0]  # 1 = Gå, 0 = Stanna hemma

    # Skapa och träna perceptronen
    perceptron = Perceptron(num_inputs=2, learning_rate=0.1, n_iterations=5)
    perceptron.fit(inputs, targets)

    print(f"Slutgiltiga vikter: {perceptron.weights}")
    print(f"Slutgiltig bias: {perceptron.bias}")

    print("\n--- Testar den tränade modellen ---")
    for inp, target in zip(inputs, targets):
        pred = perceptron.predict(inp)
        status = "✓" if pred == target else "✗"
        print(f"Input: {inp} -> Prediktion: {pred}, Förväntat: {target} {status}")
