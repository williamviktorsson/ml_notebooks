"""
NeuralNetwork - Ett fullt neuralt nätverk med backpropagation.

Detta nätverk kan ha godtyckligt antal lager och neuroner per lager,
och tränas med gradient descent och backpropagation.
"""

import random
import math
import numpy as np


def sigmoid(x):
    """Sigmoid-aktiveringsfunktion."""
    if x < -700:
        return 0.0
    if x > 700:
        return 1.0
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(sig_output):
    """
    Derivatan av sigmoid-funktionen.

    Notera: tar sigmoid-OUTPUT som input, inte x.
    Om sigmoid(x) = s, då är sigmoid'(x) = s * (1 - s)
    """
    # TODO: returnera derivatan här
    return 0.0


class Neuron:
    """
    En neuron med sigmoid-aktivering för användning i neurala nätverk.

    Till skillnad från Perceptron har denna neuron ingen egen träningslogik.
    Träningen hanteras av NeuralNetwork-klassen via backpropagation.
    """

    def __init__(self, num_inputs):
        """
        Skapar en neuron med slumpmässiga vikter och bias.

        Args:
            num_inputs: Antal inkommande kopplingar
        """
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def predict(self, inputs):
        """
        Beräknar neuronens output.

        Args:
            inputs: Lista med input-värden

        Returns:
            Sigmoid av viktad summa + bias
        """
        total = np.dot(self.weights, inputs) # ersätt loop med linjär algebra
        total += self.bias
        return sigmoid(total)


class NeuralNetwork:
    """
    Ett neuralt nätverk som kan tränas med backpropagation.

    Arkitekturen definieras av layer_sizes. T.ex. [2, 3, 1] betyder:
    - 2 inputs
    - 3 neuroner i det dolda lagret
    - 1 neuron i output-lagret
    """

    def __init__(self, layer_sizes):
        """
        Skapar ett neuralt nätverk.

        Args:
            layer_sizes: Lista med antal noder per lager
                        T.ex. [2, 3, 1] = 2 inputs, 3 dolda, 1 output
        """
        self.layer_sizes = layer_sizes
        self.layers: list[list[Neuron]] = []

        # Skapa neuroner för varje lager (utom input-lagret)
        for i in range(1, len(layer_sizes)):
            num_inputs = layer_sizes[i - 1]
            num_neurons = layer_sizes[i]
            layer = [Neuron(num_inputs) for _ in range(num_neurons)]
            self.layers.append(layer)

    def predict(self, inputs):
        """
        Kör datan framåt genom nätverket (forward pass).

        Args:
            inputs: Lista med input-värden

        Returns:
            Lista med output-värden från sista lagret
        """
        current_inputs = inputs
        for layer in self.layers:
            # TODO: för varje neuron i detta lager, beräkna dess output
            # använd current_inputs som input till varje neuron
            next_inputs = []

            # svaret från detta lager blir input till nästa lager
            current_inputs = next_inputs

        # TODO: returnera output från sista lagret
        return []


    def train(self, training_inputs, training_targets, epochs, learning_rate):
        """
        Tränar nätverket med backpropagation.

        Args:
            training_inputs: Lista av input-exempel
            training_targets: Lista av one-hot encoded facit
            epochs: Antal gånger att gå igenom all träningsdata
            learning_rate: Inlärningsfaktor (stegstorlek)
        """
        for epoch in range(epochs):
            for inputs, target in zip(training_inputs, training_targets):
                self._train_single(inputs, target, learning_rate)

            # Logga felet ibland
            if epoch % 1000 == 0 or epoch == epochs - 1:
                mse = self._calculate_mse(training_inputs, training_targets)
                print(f"Epoch {epoch}, MSE: {mse:.6f}")

    def _train_single(self, inputs, target, learning_rate):
        """
        Tränar på ett enskilt exempel.

        Detta är hjärtat av backpropagation-algoritmen.
        """
        # ===== STEG 1: FORWARD PASS =====
        # Spara outputs från varje lager (behövs för backprop)
        outputs_by_layer = [inputs] # lager 0 är input-lagret 
        current_inputs = inputs

        for layer in self.layers:
            # TODO: beräkna output för detta lager,
            # använd current_inputs som input till varje neuron
            layer_outputs = [] # denna lista ska alltså inte vara tom
            outputs_by_layer.append(layer_outputs)
            current_inputs = layer_outputs

        # ===== STEG 2: BACKWARD PASS =====
        # Beräkna delta (felansvar) för varje neuron

        # A. Delta för OUTPUT-LAGRET
        final_outputs = outputs_by_layer[-1] # sista lagrets output
        output_deltas = []

        for index,output in enumerate(final_outputs):
            # Felet = Facit - Gissning
            correct_value = target[index]
            error = 0 # TODO: beräkna felet här

            # Delta = Felet × derivatan_av_sigmoid(Gissning)
            delta = 0 # TODO: beräkna delta här
            output_deltas.append(delta)

        deltas = [output_deltas]

        # B. Delta för DOLDA LAGER (baklänges)
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            next_deltas = deltas[0]
            current_deltas = []
            current_outputs = outputs_by_layer[i + 1]

            for j, neuron in enumerate(current_layer):
                # Hur mycket bidrog denna neuron till felen i nästa lager?
                error_contribution = 0
                for k, next_neuron in enumerate(next_layer):
                    # TODO: lägg till bidraget från nästa neurons delta
                    # multiplicerat med vikten som kopplar dem
                    # vikten mellan dem hittar du i next_neuron.weights[j]
                    # deltat för nästa neuron är next_deltas[k]
                    # (Felet hos nästa neuron) * (Vikten som kopplar dem)
       
                    error_contribution += 0

                # Delta = Felbidrag × derivatan_av_sigmoid(Gissning)
                delta = 0 # TODO: beräkna delta här
                current_deltas.append(delta)

            # Lägg till först i listan (vi går baklänges)
            deltas.insert(0, current_deltas)

        # ===== STEG 3: UPPDATERA VIKTER =====
        for i, layer in enumerate(self.layers):
            inputs_to_layer = outputs_by_layer[i]

            for j, neuron in enumerate(layer):
                delta = deltas[i][j] # delta för denna neuron

                # Uppdatera vikter
                for k, input_val in enumerate(inputs_to_layer):
                    # Vikt-justering = inlärningsfaktor × delta × input_val
                    neuron.weights[k] += 0 # TODO: uppdatera vikten här

                # Uppdatera bias
                neuron.bias += learning_rate * delta

    def _calculate_mse(self, inputs, targets):
        """Beräknar Mean Squared Error för utvärdering."""
        total_error = 0
        for x, y in zip(inputs, targets):
            pred = self.predict(x)
            total_error += sum((t - p) ** 2 for t, p in zip(y, pred))
        return total_error / len(inputs)


# --- Exempel på användning: XOR ---
if __name__ == "__main__":
    print("=== Tränar neuralt nätverk för XOR ===\n")

    # XOR-problemet - kan inte lösas av en enskild perceptron!
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[0], [1], [1], [0]]

    # Skapa nätverk: 2 inputs, 2 dolda neuroner, 1 output
    nn = NeuralNetwork(layer_sizes=[2, 2, 1])

    print("Startar träning...")
    nn.train(X_train, y_train, epochs=10000, learning_rate=0.5)

    print("\n=== Testresultat ===")
    for inputs, target in zip(X_train, y_train):
        prediction = nn.predict(inputs)
        rounded = round(prediction[0])
        status = "✓" if rounded == target[0] else "✗"
        print(f"Input: {inputs} -> Gissning: {prediction[0]:.4f} (≈{rounded}), Facit: {target[0]} {status}")
