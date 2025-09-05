import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
print("FiftyOne app running at:", session.url)
input("Press Enter to quit")