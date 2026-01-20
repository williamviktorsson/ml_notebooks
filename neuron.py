"""
Neuron - Den grundläggande byggstenen för neurala nätverk.

En neuron tar emot inputs, multiplicerar dem med vikter, lägger till bias,
och kör resultatet genom en aktiveringsfunktion för att producera en output.
"""


class Neuron:
    """
    Representerar en enskild, statisk neuron som kan göra en prediktion.
    Denna version kan INTE lära sig. Dess parametrar måste ställas in manuellt.
    """

    def __init__(self, weights, bias):
        """
        Skapar en neuron med givna vikter och bias.

        Args:
            weights: Lista med vikter, en för varje input
            bias: Neuronens grundinställning (tröskel)
        """
        self.weights = weights
        self.bias = bias

    def activate(self, value) -> float:
        """
        Aktiveringsfunktionen: Stegfunktion (Heaviside).
        Om summan är positiv, returnera 1, annars 0.

        Args:
            value: Den viktade summan

        Returns:
            1 om value > 0, annars 0
        """
        if value > 0:
            return 1
        else:
            return 0

    def predict(self, inputs):
        """
        Beräknar den viktade summan och returnerar en prediktion (0 eller 1).

        Formeln: Summa = (x1*w1) + (x2*w2) + ... + b

        Args:
            inputs: Lista med input-värden

        Returns:
            0 eller 1 beroende på aktiveringsfunktionen
        """
        # Börja med summan på 0
        summation = 0

        # Loopa igenom varje input och dess motsvarande vikt
        for i in range(len(self.weights)):
            # TODO: för respektive vikt, multiplicera med respektive input
            summation += inputs[i] * self.weights[i]

        # Lägg till bias
        summation += self.bias

        return self.activate(summation)


# --- Exempel på användning ---
if __name__ == "__main__":
    # Persona 1: Den Studiemotiverade
    # Prioriterar studier över allt annat
    studious_neuron = Neuron(weights=[0.5, -1.0], bias=-0.2)

    # Scenario: Vänner är där (1), men det är ett prov imorgon (1)
    inputs = [1, 1]
    decision = studious_neuron.predict(inputs)
    print(f"Den Studiemotiverade: Input {inputs} -> Beslut: {decision} (Förväntat: 0)")

    # Persona 2: Sociala Festprissen
    # Älskar att umgås, avslappnad inställning till prov
    social_neuron = Neuron(weights=[1.0, -0.1], bias=0.5)

    # Samma scenario
    decision = social_neuron.predict(inputs)
    print(f"Sociala Festprissen: Input {inputs} -> Beslut: {decision} (Förväntat: 1)")
