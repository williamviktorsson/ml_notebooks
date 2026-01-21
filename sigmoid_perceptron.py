from perceptron import Perceptron

class SigmoidPerceptron(Perceptron):
    """
    En Perceptron med sigmoid-aktiveringsfunktion istället för stegfunktion.

    Sigmoid ger mjuka sannolikheter (0.0 - 1.0) istället för hårda beslut (0 eller 1).
    Detta är användbart för att jämföra konfidensen mellan flera klassificerare.
    """

    def activate(self, value):
        """
        Sigmoid-aktiveringsfunktion.
        Klämmer ihop vilket värde som helst till intervallet (0, 1).

        Args:
            value: Den viktade summan

        Returns:
            Ett värde mellan 0 och 1
        """
        # Skydd mot overflow
        if value < -700:
            return 0.0
        if value > 700:
            return 1.0
        
        # TODO implementera sigmoid-funktionen här
        return 0.5