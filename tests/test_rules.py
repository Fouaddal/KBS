from engine.rules import FeatureEngineeringExpertSystem
from engine.facts import DatasetFact


def test_missing_values_rule():

    engine = FeatureEngineeringExpertSystem()

    engine.reset()

    engine.declare(DatasetFact(has_missing_values=True))

    engine.run()