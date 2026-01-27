from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment


data = get_data('diabetes')

# print(data.head())
# print(data.columns.tolist())

exp = ClassificationExperiment()
exp.setup(data=data, target='Class variable', session_id=123)

# timer.start()
best_model = exp.compare_models()
print(best_model)
# timer.stop()
